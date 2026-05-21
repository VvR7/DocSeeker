"""
消融实验：仅使用 text_retrieval 工具

与主 pipeline 相同的多轮对话框架，但只开放 text_retrieval 工具。
若模型尝试调用 page_retrieval，会收到工具不可用的提示。

每轮调用均统计并记录输入 token 数。

Usage
-----
    python evaluate_ablation.py [--output_dir DIR] [--limit N] [--start_idx I]

Output
------
<output_dir>/
    eval_<timestamp>.log   — 每题详细日志
    eval_<timestamp>.json  — 每题结果汇总 JSON
"""

import argparse
import json
import logging
import os
import re
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ── 允许从父目录导入项目模块 ─────────────────────────────────────────────────
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_DIR)

import config
import inference as llm   # backend selected by config.INFERENCE_ENGINE / config.MODEL_BACKEND
from embedding import EmbeddingEncoder
from pdf_processor import PDFProcessor
from retrieval import TextRetriever
from constant import LAST_TURN_PROMPT

# ── 路径常量 ──────────────────────────────────────────────────────────────────
BENCHMARK_DIR = os.path.join(_PROJECT_DIR, "Mybenchmark")
DATA_DIR      = os.path.join(BENCHMARK_DIR, "data")
QUESTION_FILE = os.path.join(BENCHMARK_DIR, "question.json")

# ── 仅文本检索的 System Prompt（来自实验设计）────────────────────────────────
SYSTEM_PROMPT_TEXT_ONLY = """\
You are an intelligent document question-answering assistant. You have access to a PDF document and must answer the user's question by strategically retrieving relevant information from it.

## Inputs
- The user's query based on the document. You could only access the document by using retrieval tools.

## Available Tools
### Text retrieval
Retrieves the top-{k1} semantically similar **text chunks** from the document based on your query.

<tools>
{{
  "type": "function",
  "function": {{
    "name_for_human": "text_retrieval",
    "name": "text_retrieval",
    "description": "Retrieve text chunks from the document based on a concise interest description.",
    "parameters": {{
      "type": "object",
      "properties": {{
        "text_input": {{
          "type": "string",
          "description": "Short, specific description to retrieve."
        }}
      }},
      "required": ["text_input"]
    }}
  }}
}}
</tools>



## Decision Policy
Think step by step before each tool call. Follow this general process:

1. **Analyze the question**: What type of information is needed? Is it textual, visual, or both?
2. **Formulate a retrieval query**: Your `text_input` should be a precise, self-contained query — not the user's raw question verbatim. Rephrase it to maximize retrieval relevance.
3. **Execute the tool**: You may call tools multiple times, in any order.Use `text_retrieval` for text-relevant information. Call **ONLY ONE** tool per turn. 
4. **Synthesize the results**: After retrieval, integrate the text chunks and/or page images to compose a grounded, accurate answer.
5. **Answer clearly**: Respond in the same language as the user's question. 

## Rules you must follow
- There is not information from the document provided for you at the beginning. You must access it by tool use. **It's not allow to directly answer the question without any tool use**.
- It's adverise for multi turn tool calls when the information are not sufficient.

## Tool Call Format (example)
<tool_call>
{{"name": "text_retrieval", "arguments": {{"text_input": "The attention mechanism"}}}}
</tool_call>

## Output format
- When you want to call a tool, provide your thinking process ends with your tool call request **wrapped in <tool_call></tool_call> tags**.
`Your thinking process.<tool_call>Your tool request</tool_call>`
- Once you are ready to answer: provide your thinking process ends with your final answer **wrapped in <answer></answer> tags**.
`Your thinking process.<answer></answer>`\
"""

USER_PROMPT_TEMPLATE         = "The user's question is\n{question}"
TEXT_RETRIEVAL_RESP_TEMPLATE = "The most top-{k1} relevant text chunks are below:\n{text_chunks}\n"

# ── 正则 ──────────────────────────────────────────────────────────────────────
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_ANSWER_RE    = re.compile(r"<answer>(.*?)</answer>",              re.DOTALL)
_OPTION_RE    = re.compile(r"^\s*([A-Da-d])\s*[.):\s]",           re.IGNORECASE)


# ── 日志 ──────────────────────────────────────────────────────────────────────

def _setup_logging(log_path: str) -> logging.Logger:
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handlers: List[logging.Handler] = [
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, force=True)
    return logging.getLogger("ablation_text")


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _extract_option_letter(answer: str) -> str:
    stripped = answer.strip()
    if len(stripped) == 1 and stripped.upper() in "ABCD":
        return stripped.upper()
    m = _OPTION_RE.match(stripped)
    if m:
        return m.group(1).upper()
    return stripped


def _format_question(item: Dict[str, Any]) -> str:
    question = item["question"]
    options: Optional[Dict[str, str]] = item.get("options")
    if options:
        opts_str = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
        return f"{question}\n{opts_str}"
    return question


def _parse_tool_call(response: str) -> Optional[Dict[str, Any]]:
    match = _TOOL_CALL_RE.search(response)
    if match is None:
        return None
    raw_json = match.group(1).strip()
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    if "name" not in parsed or "arguments" not in parsed:
        return None
    return parsed


def _parse_answer(response: str) -> Optional[str]:
    match = _ANSWER_RE.search(response)
    if match is None:
        return None
    return match.group(1).strip()


def _text_page_ids(records: List[Dict[str, Any]]) -> List[Optional[int]]:
    seen, result = set(), []
    for rec in records:
        if rec["tool"] == "text_retrieval":
            for r in rec["results"]:
                pid = r.get("page_id")
                if pid not in seen:
                    seen.add(pid)
                    result.append(pid)
    return result


# ── 仅文本检索的 Pipeline ─────────────────────────────────────────────────────

def run_pipeline_text_only(
    text_retriever: TextRetriever,
    question: str,
    logger: logging.Logger,
) -> Tuple[str, List[Dict[str, Any]], List[int]]:
    """
    仅使用 text_retrieval 工具的多轮对话 pipeline。

    Returns:
        (answer, retrieval_records, input_tokens_per_turn)
        answer                  : <answer>...</answer> 中的内容
        retrieval_records       : 每次工具调用的元数据
        input_tokens_per_turn   : 每轮的输入 token 数列表
    """
    system_prompt = SYSTEM_PROMPT_TEXT_ONLY.format(k1=config.K1)
    user_prompt   = USER_PROMPT_TEMPLATE.format(question=question)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    retrieval_records: List[Dict[str, Any]] = []
    input_tokens_per_turn: List[int]        = []

    logger.info(
        f"Pipeline (text_only) started. Question={question!r}  "
        f"MAX_TURN={config.MAX_TURN}  K1={config.K1}"
    )

    for turn_idx in range(config.MAX_TURN):
        turn_num     = turn_idx + 1
        is_last_turn = turn_num == config.MAX_TURN

        if is_last_turn:
            messages.append({"role": "user", "content": LAST_TURN_PROMPT})
            logger.info(f"[Turn {turn_num}] Last-turn notice injected.")

        # 统计当前轮输入 token 数
        input_tokens = llm.count_tokens(messages)
        input_tokens_per_turn.append(input_tokens)
        logger.info(f"[Turn {turn_num}] 输入 tokens: {input_tokens}")

        logger.info(f"[Turn {turn_num}] Calling model ...")
        raw_response: str = llm.get_response(messages)

        logger.info(
            f"[Turn {turn_num}] 模型完整输出:\n"
            f"{'=' * 60}\n{raw_response}\n{'=' * 60}"
        )

        messages.append({"role": "assistant", "content": raw_response})

        answer = _parse_answer(raw_response)
        if answer is not None:
            logger.info(f"[Turn {turn_num}] Answer extracted: {answer!r}")
            return answer, retrieval_records, input_tokens_per_turn

        if is_last_turn:
            logger.warning(f"[Turn {turn_num}] 最后一轮未找到 <answer>，返回空字符串。")
            return "", retrieval_records, input_tokens_per_turn

        tool_call = _parse_tool_call(raw_response)
        if tool_call is None:
            logger.warning(f"[Turn {turn_num}] 未发现 <tool_call> 或 <answer>，终止 pipeline。")
            return "", retrieval_records, input_tokens_per_turn

        name       = tool_call.get("name", "")
        arguments  = tool_call.get("arguments", {})
        text_input = arguments.get("text_input", "")

        logger.info(
            f"[Turn {turn_num}] Tool call: name={name!r}  text_input={text_input!r}"
        )

        if name == "text_retrieval":
            if not text_input:
                logger.warning(f"[Turn {turn_num}] text_retrieval 调用缺少 text_input。")
                messages.append({"role": "user", "content": "Error: text_input is required for text_retrieval."})
                continue

            results = text_retriever.retrieve(text_input, k=config.K1)
            chunks_str = "\n\n".join(
                f"[Chunk {r['chunk_id']}]\n{r['text']}" for r in results
            )
            content = TEXT_RETRIEVAL_RESP_TEMPLATE.format(
                k1=config.K1, text_chunks=chunks_str
            )
            logger.info(
                f"[Turn {turn_num}] text_retrieval executed. "
                f"chunk_ids={[r['chunk_id'] for r in results]}  "
                f"page_ids={[r.get('page_id') for r in results]}"
            )
            record: Dict[str, Any] = {
                "tool":    "text_retrieval",
                "query":   text_input,
                "results": [
                    {
                        "chunk_id": r["chunk_id"],
                        "text":     r["text"],
                        "page_id":  r.get("page_id"),
                        "score":    r["score"],
                    }
                    for r in results
                ],
            }
            retrieval_records.append(record)
            messages.append({"role": "user", "content": content})

        else:
            # 该工具在此消融实验中不可用
            logger.warning(
                f"[Turn {turn_num}] 模型调用了不可用的工具: {name!r}。"
                "仅 text_retrieval 可用。"
            )
            messages.append({
                "role":    "user",
                "content": (
                    f"Tool '{name}' is not available in this setting. "
                    "Only 'text_retrieval' is available. Please use text_retrieval instead."
                ),
            })

    logger.warning("Pipeline 循环结束未返回，返回空字符串。")
    return "", retrieval_records, input_tokens_per_turn


# ── 主评估循环 ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="消融实验：仅 text_retrieval 工具")
    parser.add_argument(
        "--output_dir",
        default=os.path.join(BENCHMARK_DIR, "results_ablation_text_only"),
        help="结果输出目录",
    )
    parser.add_argument("--limit",     type=int, default=None, help="最多评测题数")
    parser.add_argument("--start_idx", type=int, default=0,    help="起始题目索引（0-based）")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.join(args.output_dir, f"eval_{timestamp}.log")
    json_path = os.path.join(args.output_dir, f"eval_{timestamp}.json")

    logger = _setup_logging(log_path)
    logger.info("=== 消融实验：仅 text_retrieval 工具 ===")
    logger.info(f"Log: {log_path}  JSON: {json_path}")
    logger.info(f"K1={config.K1}  MAX_TURN={config.MAX_TURN}")

    with open(QUESTION_FILE, encoding="utf-8") as f:
        all_questions: List[Dict[str, Any]] = json.load(f)

    subset = all_questions[args.start_idx:]
    if args.limit is not None:
        subset = subset[: args.limit]

    logger.info(
        f"总题数: {len(all_questions)}  "
        f"本次评测: {len(subset)} (start_idx={args.start_idx}, limit={args.limit})"
    )

    # ── PDF 缓存：paper_name → {"text_retriever"} ─────────────────────────
    pdf_cache: Dict[str, Dict[str, Any]] = {}
    processor = PDFProcessor()
    encoder   = EmbeddingEncoder()

    def _get_text_retriever(paper_name: str) -> TextRetriever:
        if paper_name in pdf_cache:
            return pdf_cache[paper_name]["text_retriever"]
        pdf_path = os.path.join(DATA_DIR, f"{paper_name}.pdf")
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        logger.info(f"Processing PDF for paper: {paper_name}")
        pdf_data = processor.process(pdf_path)
        tr = TextRetriever(pdf_data["text_chunks"], encoder)
        pdf_cache[paper_name] = {"text_retriever": tr}
        return tr

    summary_records: List[Dict[str, Any]] = []
    correct = 0
    total   = 0

    for q_idx, item in enumerate(subset, start=args.start_idx):
        paper_name    = item["paper_name"]
        full_question = _format_question(item)
        ground_truth  = item.get("ground_truth", "")

        logger.info(
            f"\n{'#' * 70}\n"
            f"[Q{q_idx}] paper={paper_name}  type={item['question_type']}  "
            f"level={item.get('question_level')}  page_idx={item.get('page_idx')}\n"
            f"{'#' * 70}"
        )

        model_answer          = ""
        retrieval_records: List[Dict[str, Any]] = []
        input_tokens_per_turn: List[int]        = []

        try:
            text_retriever = _get_text_retriever(paper_name)
            model_answer, retrieval_records, input_tokens_per_turn = run_pipeline_text_only(
                text_retriever, full_question, logger
            )
        except FileNotFoundError as exc:
            logger.error(f"[Q{q_idx}] PDF 不存在，跳过。{exc}")
            model_answer = "ERROR: PDF not found"
        except Exception as exc:
            logger.error(f"[Q{q_idx}] Pipeline 错误: {exc}\n{traceback.format_exc()}")
            model_answer = f"ERROR: {exc}"

        model_answer_letter = _extract_option_letter(model_answer) if model_answer else ""
        is_correct = model_answer_letter.upper() == ground_truth.strip().upper()

        # ── 详细日志 ──────────────────────────────────────────────────────
        sep = "=" * 70
        logger.info(sep)
        logger.info(
            f"[Q{q_idx}] paper={item['paper_name']}  "
            f"page_idx={item.get('page_idx')}  "
            f"type={item['question_type']}  level={item.get('question_level')}"
        )
        logger.info(f"[Q{q_idx}] QUESTION:\n{full_question}")
        logger.info(f"[Q{q_idx}] GROUND TRUTH: {ground_truth}")
        logger.info(
            f"[Q{q_idx}] MODEL ANSWER (raw): {model_answer!r}  "
            f"→ extracted letter: {model_answer_letter!r}"
        )
        logger.info(f"[Q{q_idx}] CORRECT: {is_correct}")
        logger.info(
            f"[Q{q_idx}] 最终轮输入 tokens: {input_tokens_per_turn[-1] if input_tokens_per_turn else 0}  "
            f"各轮: {input_tokens_per_turn}"
        )
        if retrieval_records:
            logger.info(f"[Q{q_idx}] 检索调用 ({len(retrieval_records)} 次):")
            for call_idx, rec in enumerate(retrieval_records, 1):
                logger.info(f"  [{call_idx}] tool={rec['tool']}  query={rec['query']!r}")
                for r in rec["results"]:
                    pid_str = f"  [page_id={r['page_id']}]" if r.get("page_id") is not None else ""
                    logger.info(
                        f"      chunk_id={r['chunk_id']}{pid_str}  score={r['score']:.4f}"
                    )
        else:
            logger.info(f"[Q{q_idx}] 无检索调用。")
        logger.info(sep)

        if ground_truth:
            total += 1
            if is_correct:
                correct += 1

        summary_records.append({
            "q_idx":                  q_idx,
            "paper_name":             paper_name,
            "question_type":          item["question_type"],
            "question_level":         item.get("question_level"),
            "page_idx":               item.get("page_idx"),
            "question":               item["question"],
            "options":                item.get("options"),
            "ground_truth":           ground_truth,
            "model_answer":           model_answer,
            "model_answer_letter":    model_answer_letter,
            "is_correct":             is_correct,
            "total_input_tokens":     input_tokens_per_turn[-1] if input_tokens_per_turn else 0,
            "input_tokens_per_turn":  input_tokens_per_turn,
            "retrieved_text_page_ids": _text_page_ids(retrieval_records),
            "retrieved_page_nums":    [],  # 无 page_retrieval
        })

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_records, f, ensure_ascii=False, indent=2)

    # ── 最终统计 ──────────────────────────────────────────────────────────────
    acc = correct / total if total > 0 else 0.0
    logger.info(f"\n{'=' * 70}")
    logger.info(f"评测完成。评测题数: {len(subset)}")
    logger.info(f"准确率: {correct}/{total} = {acc:.4f}")
    logger.info(f"Log:  {log_path}")
    logger.info(f"JSON: {json_path}")
    logger.info(f"{'=' * 70}")

    with open(json_path, "r", encoding="utf-8") as f:
        existing = json.load(f)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": {"correct": correct, "total": total, "acc": acc},
                "results":  existing,
            },
            f, ensure_ascii=False, indent=2,
        )


if __name__ == "__main__":
    main()
