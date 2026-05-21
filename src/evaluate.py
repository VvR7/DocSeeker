"""
Evaluation script for the document QA pipeline on Mybenchmark.

Usage
-----
    python evaluate.py [--output_dir OUTPUT_DIR] [--limit N] [--start_idx I]

Output
------
<output_dir>/
    eval_<timestamp>.log   — detailed log per question (question text, ground
                             truth, question type, page_idx, model raw outputs,
                             full retrieved text chunks with page_id, retrieved
                             page numbers)
    eval_<timestamp>.json  — brief summary list (question, ground_truth,
                             model_answer, retrieved_text_page_ids,
                             retrieved_page_nums)
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

# ─── Paths ────────────────────────────────────────────────────────────────────

BENCHMARK_DIR = os.path.join(os.path.dirname(__file__), "Mybenchmark")
DATA_DIR = os.path.join(BENCHMARK_DIR, "data")
QUESTION_FILE = os.path.join(BENCHMARK_DIR, "question.json")


# ─── Logging helpers ──────────────────────────────────────────────────────────

def _setup_logging(log_path: str) -> logging.Logger:
    """Configure a file+console logger dedicated to the evaluation run."""
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handlers: List[logging.Handler] = [
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
    return logging.getLogger("evaluate")


# ─── Answer extraction ───────────────────────────────────────────────────────

_OPTION_RE = re.compile(r"^\s*([A-Da-d])\s*[.):\s]", re.IGNORECASE)

def _extract_option_letter(answer: str) -> str:
    """
    Try to extract a single A/B/C/D letter from a free-form answer string.

    Handles formats such as:
        "B"            → "B"
        "B."           → "B"
        "B. some text" → "B"
        "B) some text" → "B"
        "B: some text" → "B"
        "  b  "        → "B"

    If no recognisable option letter is found, the stripped answer is returned
    unchanged so the caller can still store the raw text.
    """
    stripped = answer.strip()
    # Single-letter answer (possibly with trailing punctuation)
    if len(stripped) == 1 and stripped.upper() in "ABCD":
        return stripped.upper()
    # Letter followed by separator then optional text
    m = _OPTION_RE.match(stripped)
    if m:
        return m.group(1).upper()
    return stripped


# ─── Question formatting ──────────────────────────────────────────────────────

def _format_question(item: Dict[str, Any]) -> str:
    """
    Build the full question string (question + options if present) to feed
    into the pipeline.
    """
    question = item["question"]
    options: Optional[Dict[str, str]] = item.get("options")
    if options:
        opts_str = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
        return f"{question}\n{opts_str}"
    return question


# ─── Retrieval record summarisers ─────────────────────────────────────────────

def _text_page_ids(records: List[Dict[str, Any]]) -> List[Optional[int]]:
    """Collect all page_ids from text_retrieval records (deduplicated, sorted)."""
    seen = set()
    result = []
    for rec in records:
        if rec["tool"] == "text_retrieval":
            for r in rec["results"]:
                pid = r.get("page_id")
                if pid not in seen:
                    seen.add(pid)
                    result.append(pid)
    return result


def _page_nums(records: List[Dict[str, Any]]) -> List[int]:
    """Collect all page_num values from page_retrieval records (deduplicated, sorted)."""
    seen = set()
    result = []
    for rec in records:
        if rec["tool"] == "page_retrieval":
            for r in rec["results"]:
                pn = r["page_num"]
                if pn not in seen:
                    seen.add(pn)
                    result.append(pn)
    return result


# ─── Detailed log writer ──────────────────────────────────────────────────────

def _log_question_detail(
    logger: logging.Logger,
    q_idx: int,
    item: Dict[str, Any],
    full_question: str,
    model_answer: str,
    model_answer_letter: str,
    records: List[Dict[str, Any]],
    input_tokens_per_turn: List[int],
) -> None:
    """Write the detailed per-question block to the log."""
    sep = "=" * 70
    logger.info(sep)
    logger.info(f"[Q{q_idx}] paper={item['paper_name']}  page_idx={item.get('page_idx')}  "
                f"type={item['question_type']}  level={item.get('question_level')}")
    logger.info(f"[Q{q_idx}] QUESTION:\n{full_question}")
    logger.info(f"[Q{q_idx}] GROUND TRUTH: {item['ground_truth']}")
    logger.info(
        f"[Q{q_idx}] MODEL ANSWER (raw): {model_answer!r}  "
        f"→ extracted letter: {model_answer_letter!r}"
    )
    logger.info(
        f"[Q{q_idx}] INPUT TOKENS: last_turn={input_tokens_per_turn[-1] if input_tokens_per_turn else 0}  "
        f"per_turn={input_tokens_per_turn}"
    )

    if records:
        logger.info(f"[Q{q_idx}] RETRIEVAL HISTORY ({len(records)} call(s)):")
        for call_idx, rec in enumerate(records, 1):
            logger.info(
                f"  [{call_idx}] tool={rec['tool']}  query={rec['query']!r}"
            )
            if rec["tool"] == "text_retrieval":
                for r in rec["results"]:
                    pid_str = f"  [page_id={r['page_id']}]" if r.get("page_id") is not None else ""
                    logger.info(
                        f"      chunk_id={r['chunk_id']}{pid_str}  score={r['score']:.4f}\n"
                        # f"      TEXT: {r['text']}"
                    )
            elif rec["tool"] == "page_retrieval":
                for r in rec["results"]:
                    logger.info(
                        f"      page_num={r['page_num']} (0-indexed)  score={r['score']:.4f}"
                    )
    else:
        logger.info(f"[Q{q_idx}] No retrieval calls made.")
    logger.info(sep)


# ─── Main evaluation loop ─────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the QA pipeline on Mybenchmark")
    parser.add_argument(
        "--output_dir", default=os.path.join(BENCHMARK_DIR, "results"),
        help="Directory where log and JSON results are saved."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of questions to evaluate (default: all)."
    )
    parser.add_argument(
        "--start_idx", type=int, default=0,
        help="Index of the first question to evaluate (0-based, default: 0)."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.output_dir, f"eval_{timestamp}.log")
    json_path = os.path.join(args.output_dir, f"eval_{timestamp}.json")

    logger = _setup_logging(log_path)
    logger.info(f"Evaluation started. Log: {log_path}  JSON: {json_path}")
    logger.info(f"Benchmark data dir: {DATA_DIR}")

    # ── Load questions ────────────────────────────────────────────────────────
    with open(QUESTION_FILE, encoding="utf-8") as f:
        all_questions: List[Dict[str, Any]] = json.load(f)

    subset = all_questions[args.start_idx:]
    if args.limit is not None:
        subset = subset[: args.limit]

    logger.info(
        f"Total questions in file: {len(all_questions)}  "
        f"Evaluating: {len(subset)} (start_idx={args.start_idx}, limit={args.limit})"
    )

    # ── Import pipeline components (heavy imports deferred until here) ────────
    # Inference backend is selected via config.INFERENCE_ENGINE / config.MODEL_BACKEND.
    from embedding import EmbeddingEncoder
    from pdf_processor import PDFProcessor
    from retrieval import PageRetriever, TextRetriever
    from pipeline import run_pipeline_with_retrievers

    # ── PDF cache: paper_name → {"text_retriever", "page_retriever"} ─────────
    pdf_cache: Dict[str, Dict[str, Any]] = {}
    processor = PDFProcessor()
    encoder = EmbeddingEncoder()

    def _get_retrievers(paper_name: str):
        if paper_name in pdf_cache:
            return pdf_cache[paper_name]["text_retriever"], pdf_cache[paper_name]["page_retriever"]
        pdf_path = os.path.join(DATA_DIR, f"{paper_name}.pdf")
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        logger.info(f"Processing PDF for paper: {paper_name}")
        pdf_data = processor.process(pdf_path)
        tr = TextRetriever(pdf_data["text_chunks"], encoder)
        pr = PageRetriever(pdf_data["pages"])
        pdf_cache[paper_name] = {"text_retriever": tr, "page_retriever": pr}
        return tr, pr

    # ── Evaluation loop ───────────────────────────────────────────────────────
    summary_records: List[Dict[str, Any]] = []
    correct = 0
    total = 0

    for q_idx, item in enumerate(subset, start=args.start_idx):
        paper_name = item["paper_name"]
        full_question = _format_question(item)
        ground_truth = item.get("ground_truth", "")

        logger.info(
            f"\n{'#' * 70}\n"
            f"[Q{q_idx}] paper={paper_name}  type={item['question_type']}  "
            f"level={item.get('question_level')}  page_idx={item.get('page_idx')}\n"
            f"{'#' * 70}"
        )

        model_answer = ""
        records: List[Dict[str, Any]] = []
        input_tokens_per_turn: List[int] = []

        try:
            text_retriever, page_retriever = _get_retrievers(paper_name)
            model_answer, records, input_tokens_per_turn = run_pipeline_with_retrievers(
                text_retriever, page_retriever, full_question
            )
        except FileNotFoundError as exc:
            logger.error(f"[Q{q_idx}] PDF missing — skipping. {exc}")
            model_answer = "ERROR: PDF not found"
        except Exception as exc:
            logger.error(
                f"[Q{q_idx}] Pipeline error: {exc}\n{traceback.format_exc()}"
            )
            model_answer = f"ERROR: {exc}"

        # ── Extract option letter for accuracy check ─────────────────────────
        model_answer_letter = _extract_option_letter(model_answer) if model_answer else ""

        # ── Detailed log ─────────────────────────────────────────────────────
        _log_question_detail(
            logger, q_idx, item, full_question,
            model_answer, model_answer_letter, records,
            input_tokens_per_turn,
        )

        # ── Accuracy bookkeeping ─────────────────────────────────────────────
        is_correct = (model_answer_letter.upper() == ground_truth.strip().upper())
        if ground_truth:
            total += 1
            if is_correct:
                correct += 1

        # ── Summary record ───────────────────────────────────────────────────
        summary_records.append({
            "q_idx": q_idx,
            "paper_name": paper_name,
            "question_type": item["question_type"],
            "question_level": item.get("question_level"),
            "page_idx": item.get("page_idx"),
            "question": item["question"],
            "options": item.get("options"),
            "ground_truth": ground_truth,
            "model_answer": model_answer,
            "model_answer_letter": model_answer_letter,
            "is_correct": is_correct,
            "total_input_tokens": input_tokens_per_turn[-1] if input_tokens_per_turn else 0,
            "input_tokens_per_turn": input_tokens_per_turn,
            "retrieved_text_page_ids": _text_page_ids(records),
            "retrieved_page_nums": _page_nums(records),
        })

        # Write JSON incrementally so partial results survive crashes
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_records, f, ensure_ascii=False, indent=2)

    # ── Final statistics ──────────────────────────────────────────────────────
    acc = correct / total if total > 0 else 0.0
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Evaluation complete. Questions evaluated: {len(subset)}")
    logger.info(f"Accuracy (exact match): {correct}/{total} = {acc:.4f}")
    logger.info(f"Detailed log  : {log_path}")
    logger.info(f"Summary JSON  : {json_path}")
    logger.info(f"{'=' * 70}")

    # Append accuracy summary to JSON
    with open(json_path, "r", encoding="utf-8") as f:
        existing = json.load(f)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": {"correct": correct, "total": total, "acc": acc},
                "results": existing,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
