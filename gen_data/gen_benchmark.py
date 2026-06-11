#!/usr/bin/env python3
"""
文档 QA 多选题 Benchmark 数据生成脚本

Pipeline（每篇论文 × 每页）:
  1. 参考文献页检测  → 是则跳过
  2. MM 题目生成    (图片 / 表格 / 公式)
  3. 文本题目生成
  4. Review（有图 vs. 无图），实时写入 JSON
  5. 通过 filter 的题目聚合到全局 filter.json

用法:
  cd Doc/gen_data
  python gen_benchmark.py

环境变量覆盖配置:
  QWEN_MODEL_PATH   模型路径
  PAPER_IMG_DIR     论文图片根目录
  OUTPUT_DIR        输出根目录
  MAX_MODEL_LEN     vLLM KV cache 长度 (默认 32768)
  GPU_MEMORY_UTILIZATION  显存利用率 (默认 0.85)
"""
from __future__ import annotations

import base64
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

# ── 将当前目录加入 sys.path，确保 vllm_local.py 内 `import config` 能找到 config.py ──
_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import config
from qwenvl.vllm_local import get_response  # noqa: E402

from prompt_mm import SYSTEM_PROMPT as MM_SYSTEM_PROMPT
from prompt_mm import USER_PROMPT as MM_USER_PROMPT
from prompt_text import SYSTEM_PROMPT as TEXT_SYSTEM_PROMPT
from prompt_text import USER_PROMPT as TEXT_USER_PROMPT

# ─── 辅助 Prompts ──────────────────────────────────────────────────────────────

_REFS_CHECK_SYSTEM = """\
You are a document page classifier. Determine whether the given page is a \
references or bibliography page.

A references/bibliography page primarily contains a list of cited works \
(authors, titles, venues, years, DOIs, etc.) with little to no original content.

Output strictly valid JSON and nothing else — no markdown, no explanation."""

_REFS_CHECK_USER = """\
Is this page a references or bibliography page?

Respond with exactly one of:
{"is_references": true}
{"is_references": false}"""

_REVIEW_SYSTEM = """\
You are an expert at answering academic multiple-choice questions.
Select the single best answer for the question provided.
Output strictly valid JSON and nothing else — no markdown, no explanation."""


def _build_review_user(q: dict) -> str:
    opts = q["options"]
    return (
        "Answer the following multiple-choice question.\n\n"
        f"Question: {q['question']}\n"
        f"A: {opts['A']}\n"
        f"B: {opts['B']}\n"
        f"C: {opts['C']}\n"
        f"D: {opts['D']}\n\n"
        'Respond with JSON: {"answer": "X"} where X is A, B, C, or D.'
    )


# ─── 日志配置 ──────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


def _setup_logging(output_dir: Path) -> None:
    """同时输出到终端（INFO）和日志文件（DEBUG）。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"gen_benchmark_{ts}.log"

    fmt = "%(asctime)s [%(levelname)-8s] %(message)s"
    logging.basicConfig(
        level=logging.DEBUG,
        format=fmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    # 将 vllm / transformers 的冗余日志压低到 WARNING
    for noisy in ("vllm", "transformers", "torch", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger.info(f"日志文件: {log_path}")


# ─── 工具函数 ──────────────────────────────────────────────────────────────────

def _encode_image(image_path: str | Path) -> str:
    """将图片编码为 base64 data URL（vllm_local.py 接受此格式）。"""
    path = Path(image_path)
    ext = path.suffix.lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _msgs_with_image(system: str, user_text: str, image_b64: str) -> list:
    """构造含图片的消息列表（Qwen-VL 格式）。"""
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_b64},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def _msgs_text_only(system: str, user_text: str) -> list:
    """构造纯文本消息列表。"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ]


def _parse_json(response: str, ctx: str = "") -> dict | list | None:
    """
    从模型输出中提取 JSON，容忍 markdown 代码块包裹。
    解析失败时记录警告并返回 None。
    """
    text = response.strip()

    # 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 提取 ```json ... ``` 或 ``` ... ```
    for pat in (r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"):
        m = re.search(pat, text)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                pass

    # 贪婪提取第一个 JSON 数组或对象
    for pat in (r"(\[[\s\S]*\])", r"(\{[\s\S]*\})"):
        m = re.search(pat, text)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

    tag = f" [{ctx}]" if ctx else ""
    logger.warning(f"JSON 解析失败{tag} | 原始输出前200字符: {text[:200]!r}")
    return None


def _extract_answer_letter(parsed: dict | list | None) -> str | None:
    """从解析结果中提取答案字母 A/B/C/D，解析失败返回 None。"""
    if not isinstance(parsed, dict):
        return None
    raw = parsed.get("answer", "")
    if isinstance(raw, str):
        m = re.search(r"[ABCD]", raw.upper())
        if m:
            return m.group(0)
    return None


def _save_json(data: list | dict, path: Path) -> None:
    """原子写入 JSON（先写临时文件再重命名，防止断电丢失）。"""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    tmp.replace(path)


# ─── 各步骤函数 ────────────────────────────────────────────────────────────────

def _is_references_page(image_b64: str, paper: str, page: int) -> bool:
    """调用模型判断当前页是否为参考文献页。"""
    msgs = _msgs_with_image(_REFS_CHECK_SYSTEM, _REFS_CHECK_USER, image_b64)
    raw = get_response(
        msgs,
        max_new_tokens=config.MAX_NEW_TOKENS_REFS,
        max_model_len=config.MAX_MODEL_LEN,
        gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
        dtype=config.DTYPE,
    )
    parsed = _parse_json(raw, f"{paper}/p{page}/refs_check")
    if parsed is None:
        logger.warning(
            f"[{paper}] 第 {page} 页 参考文献检测解析失败，默认视为非参考文献页继续"
        )
        return False
    result = bool(parsed.get("is_references", False))
    logger.debug(f"[{paper}] 第 {page} 页 参考文献检测={result} | raw={raw[:80]!r}")
    return result


def _generate_mm_questions(image_b64: str, paper: str, page: int) -> list[dict]:
    """生成图片 / 表格 / 公式相关题目。"""
    msgs = _msgs_with_image(MM_SYSTEM_PROMPT, MM_USER_PROMPT, image_b64)
    raw = get_response(
        msgs,
        max_new_tokens=config.MAX_NEW_TOKENS_GENERATE,
        max_model_len=config.MAX_MODEL_LEN,
        gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
        dtype=config.DTYPE,
    )
    parsed = _parse_json(raw, f"{paper}/p{page}/mm")
    if parsed is None:
        logger.warning(f"[{paper}] 第 {page} 页 MM 题目解析失败，跳过")
        return []

    if not isinstance(parsed, list):
        parsed = [parsed]

    questions: list[dict] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue

        # result=NO 表示该页无视觉元素
        if item.get("result") == "NO":
            logger.info(f"[{paper}] 第 {page} 页 无视觉元素，跳过 MM 题目")
            continue

        missing = [k for k in ("question", "options", "answer") if k not in item]
        if missing:
            logger.warning(
                f"[{paper}] 第 {page} 页 MM 条目缺少字段 {missing}，跳过该条目"
            )
            continue

        elem_type = item.get("element_type", "figure")
        q_level = item.get("question_level")
        questions.append(
            {
                "paper_name": paper,
                "page_idx": page,
                "question_type": elem_type,
                "element_label": item.get("element_label", ""),
                "question_level": q_level,
                "question": item["question"],
                "options": item["options"],
                "ground_truth": item["answer"],
                "rationale": item.get("rationale", ""),
                "grounding": item.get("grounding", ""),
            }
        )

    level_summary = ", ".join(
        f"L{q['question_level']}" for q in questions if q.get("question_level")
    )
    logger.info(
        f"[{paper}] 第 {page} 页 MM 题目生成 {len(questions)} 道"
        + (f" [levels: {level_summary}]" if level_summary else "")
    )
    return questions


def _generate_text_questions(image_b64: str, paper: str, page: int) -> list[dict]:
    """生成文本内容相关题目。"""
    msgs = _msgs_with_image(TEXT_SYSTEM_PROMPT, TEXT_USER_PROMPT, image_b64)
    raw = get_response(
        msgs,
        max_new_tokens=config.MAX_NEW_TOKENS_GENERATE,
        max_model_len=config.MAX_MODEL_LEN,
        gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
        dtype=config.DTYPE,
    )
    parsed = _parse_json(raw, f"{paper}/p{page}/text")
    if parsed is None:
        logger.warning(f"[{paper}] 第 {page} 页 文本题目解析失败，跳过")
        return []

    if not isinstance(parsed, list):
        parsed = [parsed]

    questions: list[dict] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        missing = [k for k in ("question", "options", "answer") if k not in item]
        if missing:
            logger.warning(
                f"[{paper}] 第 {page} 页 文本条目缺少字段 {missing}，跳过该条目"
            )
            continue
        q_level = item.get("question_level")
        # 提取每个干扰项的 distractor_type（文本 prompt 要求模型在 rationale 中标注）
        distractor_types = item.get("distractor_type", {})
        questions.append(
            {
                "paper_name": paper,
                "page_idx": page,
                "question_type": "text",
                "question_level": q_level,
                "question": item["question"],
                "options": item["options"],
                "ground_truth": item["answer"],
                "rationale": item.get("rationale", ""),
                "distractor_type": distractor_types,
                "source_text": item.get("source_text", ""),
            }
        )

    level_summary = ", ".join(
        f"L{q['question_level']}" for q in questions if q.get("question_level")
    )
    logger.info(
        f"[{paper}] 第 {page} 页 文本题目生成 {len(questions)} 道"
        + (f" [levels: {level_summary}]" if level_summary else "")
    )
    return questions


def _review_question(q: dict, image_b64: str) -> dict:
    """
    双路 Review:
      review_1 — 图片 + 题目  → 模型作答
      review_2 — 仅题目       → 模型作答

    结果写入 q 的以下字段:
      review_with_image_answer    : str | None
      review_without_image_answer : str | None
      review_with_image_correct   : bool
      review_without_image_correct: bool
      in_filter                   : bool  (review_1 对 且 review_2 错)
    """
    paper = q["paper_name"]
    page = q["page_idx"]
    review_user = _build_review_user(q)
    gt = q.get("ground_truth", "").strip().upper()

    # Review 1: 有图
    msgs1 = _msgs_with_image(_REVIEW_SYSTEM, review_user, image_b64)
    raw1 = get_response(
        msgs1,
        max_new_tokens=config.MAX_NEW_TOKENS_REVIEW,
        max_model_len=config.MAX_MODEL_LEN,
        gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
        dtype=config.DTYPE,
    )
    ans1 = _extract_answer_letter(_parse_json(raw1, f"{paper}/p{page}/review_with_img"))

    # Review 2: 无图
    msgs2 = _msgs_text_only(_REVIEW_SYSTEM, review_user)
    raw2 = get_response(
        msgs2,
        max_new_tokens=config.MAX_NEW_TOKENS_REVIEW,
        max_model_len=config.MAX_MODEL_LEN,
        gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
        dtype=config.DTYPE,
    )
    ans2 = _extract_answer_letter(_parse_json(raw2, f"{paper}/p{page}/review_no_img"))

    correct1 = (ans1 == gt) if (ans1 and gt) else False
    correct2 = (ans2 == gt) if (ans2 and gt) else False

    q["review_with_image_answer"] = ans1
    q["review_without_image_answer"] = ans2
    q["review_with_image_correct"] = correct1
    q["review_without_image_correct"] = correct2
    q["in_filter"] = correct1 and not correct2

    flag_img = f"{ans1}({'✓' if correct1 else '✗'})" if ans1 else "N/A"
    flag_noimg = f"{ans2}({'✓' if correct2 else '✗'})" if ans2 else "N/A"
    level_tag = f"L{q['question_level']} " if q.get("question_level") else ""
    logger.info(
        f"[{paper}] 第 {page} 页 review | "
        f"{level_tag}"
        f"type={q.get('question_type', '?')}  "
        f"有图={flag_img}  无图={flag_noimg}  "
        f"过滤={'是' if q['in_filter'] else '否'} | "
        f"题目: {q['question'][:60]!r}"
    )
    return q


# ─── 论文级处理 ────────────────────────────────────────────────────────────────

def _process_paper(
    paper_name: str,
    paper_dir: Path,
    output_dir: Path,
    filter_questions: list[dict],
) -> list[dict]:
    """
    处理单篇论文，返回所有生成的题目列表。
    支持断点续跑：已处理的页面（以 page_idx 为键）不会重复处理。
    """
    paper_out = output_dir / paper_name
    paper_out.mkdir(parents=True, exist_ok=True)
    questions_file = paper_out / "questions.json"

    # 加载已有进度
    all_questions: list[dict] = []
    processed_pages: set[int] = set()
    if questions_file.exists():
        with open(questions_file, encoding="utf-8") as fh:
            all_questions = json.load(fh)
        processed_pages = {q["page_idx"] for q in all_questions}
        logger.info(
            f"[{paper_name}] 加载已有进度: {len(all_questions)} 道题目，"
            f"已处理页面: {sorted(processed_pages)}"
        )

    # 枚举页面图片，按页码升序排列
    page_files = sorted(
        paper_dir.glob("page*.jpg"),
        key=lambda p: int(re.search(r"page(\d+)", p.stem).group(1)),
    )
    if not page_files:
        logger.warning(f"[{paper_name}] 未找到任何页面图片（page*.jpg），跳过该论文")
        return all_questions

    logger.info(f"[{paper_name}] 共 {len(page_files)} 页待处理")

    for page_path in page_files:
        page_idx = int(re.search(r"page(\d+)", page_path.stem).group(1))

        if page_idx in processed_pages:
            logger.debug(f"[{paper_name}] 第 {page_idx} 页 已处理，跳过")
            continue

        logger.info(f"[{paper_name}] ── 第 {page_idx} 页 开始 ──────────────────")

        # 编码图片
        try:
            image_b64 = _encode_image(page_path)
        except Exception as exc:
            logger.error(f"[{paper_name}] 第 {page_idx} 页 图片读取失败: {exc}，跳过")
            continue

        # Step 1: 参考文献页检测
        if _is_references_page(image_b64, paper_name, page_idx):
            logger.info(f"[{paper_name}] 第 {page_idx} 页 判定为参考文献页，跳过题目生成")
            processed_pages.add(page_idx)
            continue

        # Step 2 & 3: 生成题目
        mm_qs = _generate_mm_questions(image_b64, paper_name, page_idx)
        text_qs = _generate_text_questions(image_b64, paper_name, page_idx)
        new_qs = mm_qs + text_qs

        logger.info(
            f"[{paper_name}] 第 {page_idx} 页 共生成 {len(new_qs)} 道题目 "
            f"(MM:{len(mm_qs)}, 文本:{len(text_qs)})"
        )

        # Step 4: Review 并实时持久化
        for q in new_qs:
            q = _review_question(q, image_b64)
            all_questions.append(q)

            if q["in_filter"]:
                filter_questions.append(q)

            # 实时写入，保障断点后数据不丢失
            _save_json(all_questions, questions_file)

        processed_pages.add(page_idx)
        logger.info(
            f"[{paper_name}] 第 {page_idx} 页 处理完成，"
            f"累计题目: {len(all_questions)} 道"
        )

    logger.info(
        f"[{paper_name}] 全部页面处理完成，共 {len(all_questions)} 道题目"
    )
    return all_questions


# ─── 主入口 ────────────────────────────────────────────────────────────────────

def main() -> None:
    paper_img_dir = Path(config.PAPER_IMG_DIR)
    output_dir = Path(config.OUTPUT_DIR)

    _setup_logging(output_dir)

    logger.info("=" * 70)
    logger.info("文档 QA Benchmark 数据生成 开始")
    logger.info(f"  论文图片目录 : {paper_img_dir.resolve()}")
    logger.info(f"  输出目录     : {output_dir.resolve()}")
    logger.info(f"  模型路径     : {config.QWEN_MODEL_PATH}")
    logger.info(f"  MAX_MODEL_LEN: {config.MAX_MODEL_LEN}")
    logger.info(f"  GPU_MEM_UTIL : {config.GPU_MEMORY_UTILIZATION}")
    logger.info("=" * 70)

    if not paper_img_dir.exists():
        logger.error(f"论文图片目录不存在: {paper_img_dir.resolve()}")
        sys.exit(1)

    paper_dirs = sorted(
        [d for d in paper_img_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )
    if not paper_dirs:
        logger.error(f"论文图片目录下没有子目录: {paper_img_dir.resolve()}")
        sys.exit(1)

    logger.info(f"共发现 {len(paper_dirs)} 篇论文: {[d.name for d in paper_dirs]}")

    # 加载全局 filter.json（支持断点续跑）
    filter_file = output_dir / "filter.json"
    filter_questions: list[dict] = []
    if filter_file.exists():
        with open(filter_file, encoding="utf-8") as fh:
            filter_questions = json.load(fh)
        logger.info(f"加载已有 filter.json: {len(filter_questions)} 道题目")

    total_questions = 0
    for paper_dir in paper_dirs:
        paper_name = paper_dir.name
        logger.info(f"\n{'━' * 70}")
        logger.info(f"处理论文: {paper_name}")
        logger.info(f"{'━' * 70}")

        questions = _process_paper(paper_name, paper_dir, output_dir, filter_questions)
        total_questions += len(questions)

        # 每处理完一篇论文即更新 filter.json
        _save_json(filter_questions, filter_file)
        logger.info(
            f"filter.json 已更新: {len(filter_questions)} 道题目 → {filter_file}"
        )

    logger.info("\n" + "=" * 70)
    logger.info("全部处理完成！")
    logger.info(f"  总生成题目数    : {total_questions}")
    logger.info(f"  通过 filter 题目: {len(filter_questions)}")
    logger.info(f"  filter.json     : {filter_file.resolve()}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
