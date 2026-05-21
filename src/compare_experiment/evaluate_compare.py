"""
对比实验：直接多图 QA（无检索工具）

将 PDF 所有页面渲染为 PIL.Image，连同问题一起放入模型上下文，
让模型直接输出选项字母（A/B/C/D）。

每张图 max_pixels = 1024 * 32 * 32

Usage
-----
    python evaluate_compare.py [--output_dir DIR] [--limit N] [--start_idx I]

Output
------
<output_dir>/
    eval_<timestamp>.log   — 每题详细日志（含每次输入 token 数、模型完整输出）
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
from typing import Any, Dict, List, Optional

# ── 允许从父目录导入项目模块 ─────────────────────────────────────────────────
_PROJECT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COMPARE_DIR  = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, _COMPARE_DIR)

import fitz  # PyMuPDF
import PIL.Image

import config
import inference as llm   # backend selected by config.INFERENCE_ENGINE / config.MODEL_BACKEND
from constant import SYSTEM_PROMPT_TEMPLATE

# ── 路径常量 ──────────────────────────────────────────────────────────────────
BENCHMARK_DIR = os.path.join(_PROJECT_DIR, "Mybenchmark")
DATA_DIR      = os.path.join(BENCHMARK_DIR, "data")
QUESTION_FILE = os.path.join(BENCHMARK_DIR, "question.json")

MAX_PIXELS = 1536 * 32 * 32
MIN_PIXELS = 64  * 32 * 32

SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE


# ── 日志 ──────────────────────────────────────────────────────────────────────

def _setup_logging(log_path: str) -> logging.Logger:
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handlers: List[logging.Handler] = [
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, force=True)
    return logging.getLogger("compare_eval")


# ── 答案提取 ──────────────────────────────────────────────────────────────────

_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_OPTION_RE     = re.compile(r"[A-D]", re.IGNORECASE)


def _extract_option_letter(raw: str) -> str:
    """
    仅从 <answer>...</answer> 标签中提取第一个大写字母(A/B/C/D)。
    若无 <answer> 标签或标签内找不到 A/B/C/D，返回空字符串（视为答错）。
    """
    tag_match = _ANSWER_TAG_RE.search(raw)
    if not tag_match:
        return ""

    opt = _OPTION_RE.search(tag_match.group(1))
    if opt:
        return opt.group(0).upper()

    return ""


def _format_question(item: Dict[str, Any]) -> str:
    question = item["question"]
    options: Optional[Dict[str, str]] = item.get("options")
    if options:
        opts_str = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
        return f"{question}\n{opts_str}"
    return question


# ── PDF 渲染 ──────────────────────────────────────────────────────────────────

def _pdf_to_images(pdf_path: str) -> List[PIL.Image.Image]:
    """将 PDF 每页渲染为 PIL Image，使用 config.PAGE_RENDER_DPI。"""
    doc = fitz.open(pdf_path)
    zoom   = config.PAGE_RENDER_DPI / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    images: List[PIL.Image.Image] = []
    for page_index in range(len(doc)):
        pix = doc[page_index].get_pixmap(matrix=matrix, alpha=False)
        img = PIL.Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        images.append(img)
    doc.close()
    return images


# ── 消息构建 ──────────────────────────────────────────────────────────────────

def _build_messages(images: List[PIL.Image.Image], question: str) -> list:
    """构建包含所有页面图片和问题的消息列表。"""
    user_content: List[Dict[str, Any]] = []
    user_content.append({"type": "text", "text": "Below are all pages of the document:\n"})
    for i, img in enumerate(images):
        user_content.append({"type": "text", "text": f"[Page {i}]\n"})
        user_content.append({
            "type":       "image",
            "image":      img,
            "min_pixels": MIN_PIXELS,
            "max_pixels": MAX_PIXELS,
        })
        user_content.append({"type": "text", "text": "\n"})
    user_content.append({
        "type": "text",
        "text": (
            f"\nQuestion:\n{question}\n\n"
            "Think step by step, then provide your final answer wrapped in <answer></answer> tags."
        ),
    })
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


# ── 主评估循环 ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="对比实验：直接多图 QA")
    parser.add_argument(
        "--output_dir",
        default=os.path.join(BENCHMARK_DIR, "results_compare"),
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
    logger.info(f"=== 对比实验（直接多图 QA）===")
    logger.info(f"Log: {log_path}  JSON: {json_path}")
    logger.info(f"MAX_PIXELS per image: {MAX_PIXELS}  ({MAX_PIXELS // (32*32)} tokens/image budget)")

    with open(QUESTION_FILE, encoding="utf-8") as f:
        all_questions: List[Dict[str, Any]] = json.load(f)

    subset = all_questions[args.start_idx:]
    if args.limit is not None:
        subset = subset[: args.limit]

    logger.info(
        f"总题数: {len(all_questions)}  "
        f"本次评测: {len(subset)} (start_idx={args.start_idx}, limit={args.limit})"
    )

    # ── PDF 图片缓存：paper_name → List[PIL.Image] ─────────────────────────
    pdf_image_cache: Dict[str, List[PIL.Image.Image]] = {}

    def _get_images(paper_name: str) -> List[PIL.Image.Image]:
        if paper_name in pdf_image_cache:
            return pdf_image_cache[paper_name]
        pdf_path = os.path.join(DATA_DIR, f"{paper_name}.pdf")
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        logger.info(f"Rendering PDF pages for: {paper_name}")
        images = _pdf_to_images(pdf_path)
        pdf_image_cache[paper_name] = images
        logger.info(f"  渲染完成：{len(images)} 页")
        return images

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

        model_answer  = ""
        input_tokens  = 0
        num_pages     = 0

        try:
            images    = _get_images(paper_name)
            num_pages = len(images)
            messages  = _build_messages(images, full_question)

            # 统计输入 token 数
            input_tokens = llm.count_tokens(messages)
            logger.info(f"[Q{q_idx}] 页数: {num_pages}  输入 tokens: {input_tokens}")

            raw_response = llm.get_response(messages)
            logger.info(
                f"[Q{q_idx}] 模型完整输出:\n"
                f"{'=' * 60}\n{raw_response}\n{'=' * 60}"
            )
            model_answer = raw_response.strip()

        except FileNotFoundError as exc:
            logger.error(f"[Q{q_idx}] PDF 不存在，跳过。{exc}")
            model_answer = "ERROR: PDF not found"
        except Exception as exc:
            logger.error(f"[Q{q_idx}] 推理错误: {exc}\n{traceback.format_exc()}")
            model_answer = f"ERROR: {exc}"

        model_answer_letter = _extract_option_letter(model_answer) if model_answer else ""
        is_correct = model_answer_letter.upper() == ground_truth.strip().upper()

        sep = "=" * 70
        logger.info(sep)
        logger.info(f"[Q{q_idx}] QUESTION:\n{full_question}")
        logger.info(f"[Q{q_idx}] GROUND TRUTH: {ground_truth}")
        logger.info(
            f"[Q{q_idx}] MODEL ANSWER (raw): {model_answer!r}  "
            f"→ extracted letter: {model_answer_letter!r}"
        )
        logger.info(f"[Q{q_idx}] CORRECT: {is_correct}")
        logger.info(sep)

        if ground_truth:
            total += 1
            if is_correct:
                correct += 1

        summary_records.append({
            "q_idx":               q_idx,
            "paper_name":          paper_name,
            "question_type":       item["question_type"],
            "question_level":      item.get("question_level"),
            "page_idx":            item.get("page_idx"),
            "question":            item["question"],
            "options":             item.get("options"),
            "ground_truth":        ground_truth,
            "model_answer":        model_answer,
            "model_answer_letter": model_answer_letter,
            "is_correct":          is_correct,
            "input_tokens":        input_tokens,
            "num_pages":           num_pages,
            # 对比实验无检索，保持字段一致
            "retrieved_text_page_ids": [],
            "retrieved_page_nums":     [],
        })

        # 增量写入 JSON，防止中途崩溃丢失数据
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
