"""
Document QA pipeline.

Entry points:
    run_pipeline(pdf_path, question)
        Full pipeline including PDF pre-processing.  Returns (answer, records).

    run_pipeline_with_retrievers(text_retriever, page_retriever, question)
        Dialogue loop only (PDF already processed).  Returns (answer, records).

Flow
----
1. Pre-process the PDF → text chunks + page images, each with embeddings.
2. Build an initial messages list: [system, user-question].
3. Repeatedly call Qwen3-VL (up to MAX_TURN times):
   - Log the full raw response.
   - If the model emits <answer>...</answer> → extract and return.
   - If the model emits a <tool_call> → execute the tool, append the result
     as the next user message.
   - If neither pattern is found → treat as empty answer and stop.
4. Before the final allowed call, inject the "last turn" prompt so the model
   knows to produce a conclusive answer.

Return value (answer, records)
------------------------------
answer  : str   — content inside the <answer> tag (empty string if not found).
records : list  — one entry per tool call:
    {
        "tool":    "text_retrieval" | "page_retrieval",
        "query":   str,
        "results": [
            # text_retrieval:
            {"chunk_id": int, "text": str, "page_id": int|None, "score": float}
            # page_retrieval:
            {"page_num": int, "score": float}
        ]
    }
"""

import json
import logging
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import config
import inference as llm   # backend selected by config.INFERENCE_ENGINE / config.MODEL_BACKEND
from embedding import EmbeddingEncoder
from pdf_processor import PDFProcessor
from retrieval import PageRetriever, TextRetriever
from constant import (
    SYSTEM_PROMPT_TEMPLATE,
    USER_PROMPT_TEMPLATE,
    TEXT_RETRIEVAL_RESPONSE_TEMPLATE,
    LAST_TURN_PROMPT,
)

# ─── Logging setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─── Parsing helpers ──────────────────────────────────────────────────────────

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL
)
_ANSWER_RE = re.compile(
    r"<answer>(.*?)</answer>", re.DOTALL
)


def parse_tool_call(response: str) -> Optional[Dict[str, Any]]:
    """
    Extract the first <tool_call>…</tool_call> block and parse the JSON inside.

    Returns a dict with keys ``name`` (str) and ``arguments`` (dict),
    or None if no valid tool call is found.
    """
    match = _TOOL_CALL_RE.search(response)
    if match is None:
        return None
    raw_json = match.group(1).strip()
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        logger.warning(f"Failed to parse tool call JSON: {exc}\nRaw: {raw_json}")
        return None

    if not isinstance(parsed, dict):
        logger.warning(f"Tool call JSON is not a dict: {parsed}")
        return None
    if "name" not in parsed or "arguments" not in parsed:
        logger.warning(f"Tool call JSON missing 'name' or 'arguments': {parsed}")
        return None

    return parsed


def parse_answer(response: str) -> Optional[str]:
    """
    Extract the content inside the first <answer>…</answer> tag.

    Returns the stripped string, or None if the tags are absent.
    """
    match = _ANSWER_RE.search(response)
    if match is None:
        return None
    return match.group(1).strip()


# ─── Tool execution ───────────────────────────────────────────────────────────

def _execute_text_retrieval(
    text_input: str,
    text_retriever: TextRetriever,
) -> Tuple[str, Dict[str, Any]]:
    """
    Run text retrieval.

    Returns:
        content : str          — user-message text to append to the conversation.
        record  : dict         — retrieval metadata for logging.
    """
    results = text_retriever.retrieve(text_input, k=config.K1)
    chunks_str = "\n\n".join(
        f"[Chunk {r['chunk_id']}]\n{r['text']}" for r in results
    )
    content = TEXT_RETRIEVAL_RESPONSE_TEMPLATE.format(
        k1=config.K1, text_chunks=chunks_str
    )
    logger.info(
        f"text_retrieval executed. "
        f"Retrieved chunk_ids: {[r['chunk_id'] for r in results]}, "
        f"page_ids: {[r['page_id'] for r in results]}"
    )
    record: Dict[str, Any] = {
        "tool": "text_retrieval",
        "query": text_input,
        "results": [
            {
                "chunk_id": r["chunk_id"],
                "text": r["text"],
                "page_id": r.get("page_id"),
                "score": r["score"],
            }
            for r in results
        ],
    }
    return content, record


def _execute_page_retrieval(
    text_input: str,
    page_retriever: PageRetriever,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run page retrieval.

    Returns:
        content : list[dict]   — multi-modal user-message content items.
        record  : dict         — retrieval metadata for logging.
    """
    results = page_retriever.retrieve(text_input, k=config.K2)

    page_nums = [r["page_num"] for r in results]
    logger.info(
        f"page_retrieval executed. "
        f"Retrieved page numbers (0-indexed): {page_nums}, "
        f"scores: {[round(r['score'], 4) for r in results]}"
    )

    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": f"The most top-{config.K2} relevant document page images are below:\n",
        }
    ]
    for result in results:
        content.append(
            {"type": "text", "text": f"[Page {result['page_num']}]\n"}
        )
        content.append({
            "type": "image",
            "image": result["image"],
            "min_pixels": 64 * 32 * 32,
            "max_pixels": 1536 * 32 * 32,
        })
        content.append({"type": "text", "text": "\n"})

    record: Dict[str, Any] = {
        "tool": "page_retrieval",
        "query": text_input,
        "results": [
            {"page_num": r["page_num"], "score": r["score"]}
            for r in results
        ],
    }
    return content, record


def _execute_tool(
    tool_call: Dict[str, Any],
    text_retriever: TextRetriever,
    page_retriever: PageRetriever,
) -> Optional[Tuple[Union[str, List[Dict]], Dict[str, Any]]]:
    """
    Dispatch a parsed tool call to the appropriate retriever.

    Returns (content, record) or None if the tool name is unrecognised.
    """
    name: str = tool_call.get("name", "")
    arguments: Dict = tool_call.get("arguments", {})
    text_input: str = arguments.get("text_input", "")

    if not text_input:
        logger.warning(f"Tool '{name}' called with empty text_input.")
        return None

    if name == "text_retrieval":
        return _execute_text_retrieval(text_input, text_retriever)
    elif name == "page_retrieval":
        return _execute_page_retrieval(text_input, page_retriever)
    else:
        logger.warning(f"Unknown tool name: '{name}'")
        return None


# ─── Core dialogue loop ───────────────────────────────────────────────────────

def run_pipeline_with_retrievers(
    text_retriever: TextRetriever,
    page_retriever: PageRetriever,
    question: str,
) -> Tuple[str, List[Dict[str, Any]], List[int]]:
    """
    Multi-turn dialogue loop given pre-built retrievers.

    Args:
        text_retriever: Initialised TextRetriever for the document.
        page_retriever: Initialised PageRetriever for the document.
        question:       The user's natural-language question.

    Returns:
        (answer, retrieval_records, input_tokens_per_turn)
        answer                : str       — text inside <answer>…</answer>.
        retrieval_records     : list      — one dict per tool call (see module docstring).
        input_tokens_per_turn : list[int] — input token count for each model call.
    """
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(k1=config.K1, k2=config.K2)
    user_prompt = USER_PROMPT_TEMPLATE.format(question=question)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    retrieval_records: List[Dict[str, Any]] = []
    input_tokens_per_turn: List[int] = []

    logger.info(
        f"Pipeline started. Question={question!r}  "
        f"MAX_TURN={config.MAX_TURN}  K1={config.K1}  K2={config.K2}"
    )

    for turn_idx in range(config.MAX_TURN):
        turn_num = turn_idx + 1
        is_last_turn = turn_num == config.MAX_TURN

        if is_last_turn:
            messages.append({"role": "user", "content": LAST_TURN_PROMPT})
            logger.info(f"[Turn {turn_num}] Last-turn notice injected.")

        input_tokens = llm.count_tokens(messages)
        input_tokens_per_turn.append(input_tokens)
        logger.info(f"[Turn {turn_num}] Input tokens: {input_tokens}")

        logger.info(f"[Turn {turn_num}] Calling model ({config.MODEL_BACKEND}/{config.INFERENCE_ENGINE}) ...")
        raw_response: str = llm.get_response(messages)

        logger.info(
            f"[Turn {turn_num}] Raw model response:\n"
            f"{'=' * 60}\n{raw_response}\n{'=' * 60}"
        )

        messages.append({"role": "assistant", "content": raw_response})

        answer = parse_answer(raw_response)
        if answer is not None:
            logger.info(f"[Turn {turn_num}] Answer extracted:\n{answer}")
            return answer, retrieval_records, input_tokens_per_turn

        if is_last_turn:
            logger.warning(
                f"[Turn {turn_num}] Last turn reached but no <answer> tag found. "
                "Returning empty string."
            )
            return "", retrieval_records, input_tokens_per_turn

        tool_call = parse_tool_call(raw_response)
        if tool_call is None:
            logger.warning(
                f"[Turn {turn_num}] No <tool_call> or <answer> found. "
                "Stopping pipeline."
            )
            return "", retrieval_records, input_tokens_per_turn

        logger.info(
            f"[Turn {turn_num}] Tool call detected: "
            f"name={tool_call['name']!r}  "
            f"text_input={tool_call['arguments'].get('text_input', '')!r}"
        )

        result = _execute_tool(tool_call, text_retriever, page_retriever)
        if result is None:
            logger.warning(
                f"[Turn {turn_num}] Tool execution returned None. Stopping pipeline."
            )
            return "", retrieval_records, input_tokens_per_turn

        tool_content, record = result
        retrieval_records.append(record)
        messages.append({"role": "user", "content": tool_content})

    logger.warning("Pipeline loop exited without returning. Returning empty string.")
    return "", retrieval_records, input_tokens_per_turn


# ─── Full pipeline (PDF processing + dialogue) ────────────────────────────────

def run_pipeline(pdf_path: str, question: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Full document QA pipeline: process the PDF then run the dialogue loop.

    Args:
        pdf_path: Path to the input PDF file.
        question: The user's natural-language question about the document.

    Returns:
        (answer, retrieval_records)  — see run_pipeline_with_retrievers.
    """
    processor = PDFProcessor()
    pdf_data = processor.process(pdf_path)

    encoder = EmbeddingEncoder()
    text_retriever = TextRetriever(pdf_data["text_chunks"], encoder)
    page_retriever = PageRetriever(pdf_data["pages"])

    logger.info(f"PDF pre-processing complete. Starting dialogue. PDF={pdf_path!r}")
    return run_pipeline_with_retrievers(text_retriever, page_retriever, question)


# ─── Callback-based dialogue loop (for streaming) ────────────────────────────

def _run_pipeline_dialogue_with_callback(
    text_retriever: TextRetriever,
    page_retriever: PageRetriever,
    question: str,
    callback,
) -> None:
    """
    Same logic as run_pipeline_with_retrievers, but fires ``callback(event_dict)``
    at each meaningful step instead of returning values.  Used by
    run_pipeline_streaming to drive the SSE event stream.
    """
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(k1=config.K1, k2=config.K2)
    user_prompt = USER_PROMPT_TEMPLATE.format(question=question)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for turn_idx in range(config.MAX_TURN):
        turn_num = turn_idx + 1
        is_last_turn = turn_num == config.MAX_TURN

        callback({"type": "turn_start", "turn": turn_num, "total": config.MAX_TURN})

        if is_last_turn:
            messages.append({"role": "user", "content": LAST_TURN_PROMPT})

        callback({"type": "model_thinking", "turn": turn_num})
        raw_response: str = llm.get_response(messages)
        messages.append({"role": "assistant", "content": raw_response})

        answer = parse_answer(raw_response)
        if answer is not None:
            callback({"type": "answer", "content": answer})
            return

        if is_last_turn:
            callback({"type": "answer", "content": ""})
            return

        tool_call = parse_tool_call(raw_response)
        if tool_call is None:
            callback({"type": "answer", "content": ""})
            return

        tool_name: str = tool_call["name"]
        query: str = tool_call["arguments"].get("text_input", "")

        # Extract the model's reasoning text that precedes the <tool_call> tag
        thinking_text = re.sub(r"<tool_call>.*", "", raw_response, flags=re.DOTALL).strip()

        callback({
            "type": "tool_call",
            "turn": turn_num,
            "tool": tool_name,
            "query": query,
            "thinking": thinking_text[:400] if thinking_text else "",
        })

        result = _execute_tool(tool_call, text_retriever, page_retriever)
        if result is None:
            callback({"type": "answer", "content": ""})
            return

        tool_content, record = result

        # Build a lightweight results summary — skip raw PIL images
        summary: List[Dict[str, Any]] = []
        for r in record.get("results", []):
            if "chunk_id" in r:  # text_retrieval
                summary.append({
                    "chunk_id": r["chunk_id"],
                    "page_id": r.get("page_id"),
                    "score": round(float(r["score"]), 4),
                    "snippet": r["text"][:150].strip(),
                })
            else:  # page_retrieval
                summary.append({
                    "page_num": r["page_num"],
                    "score": round(float(r["score"]), 4),
                })

        callback({
            "type": "tool_result",
            "turn": turn_num,
            "tool": tool_name,
            "results": summary,
        })

        messages.append({"role": "user", "content": tool_content})

    callback({"type": "answer", "content": ""})


def run_pipeline_streaming(pdf_path: str, question: str):
    """
    Generator-based pipeline that yields event dicts as processing progresses.

    Runs the blocking pipeline in a daemon thread and bridges results back via
    a queue so the caller can iterate lazily without blocking the event loop.

    Yielded event shapes
    --------------------
    {"type": "status",        "message": str}
    {"type": "turn_start",    "turn": int, "total": int}
    {"type": "model_thinking","turn": int}
    {"type": "tool_call",     "turn": int, "tool": str, "query": str, "thinking": str}
    {"type": "tool_result",   "turn": int, "tool": str, "results": list}
    {"type": "answer",        "content": str}
    {"type": "error",         "message": str}
    """
    import queue as _queue
    import threading as _threading

    q: _queue.Queue = _queue.Queue()
    _DONE = object()

    def _cb(event: dict) -> None:
        q.put(event)

    def _worker() -> None:
        try:
            _cb({"type": "status", "message": "Processing PDF document..."})
            processor = PDFProcessor()
            pdf_data = processor.process(pdf_path)

            encoder = EmbeddingEncoder()
            text_retriever = TextRetriever(pdf_data["text_chunks"], encoder)
            page_retriever = PageRetriever(pdf_data["pages"])

            _cb({"type": "status", "message": "PDF processed. Starting inference..."})
            _run_pipeline_dialogue_with_callback(
                text_retriever, page_retriever, question, _cb
            )
        except Exception as exc:
            logger.exception("run_pipeline_streaming worker error")
            _cb({"type": "error", "message": str(exc)})
        finally:
            q.put(_DONE)

    thread = _threading.Thread(target=_worker, daemon=True)
    thread.start()

    while True:
        item = q.get()
        if item is _DONE:
            return
        yield item


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Document QA pipeline")
    parser.add_argument("--pdf_path", help="Path to the input PDF file")
    parser.add_argument("--question", help="Question to answer about the document")
    args = parser.parse_args()

    final_answer, _ = run_pipeline(args.pdf_path, args.question)
    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print(final_answer)
    print("=" * 60)
