import os
import sys
import json
import time
import asyncio
import logging
import shutil
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# ── Set MOCK_MODE = True to run without GPU (fake pipeline for UI testing) ──
MOCK_MODE = True

if not MOCK_MODE:
    # Make project root importable when running from service/ or project root
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from pipeline import run_pipeline, run_pipeline_streaming  # noqa: E402

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("chat-app")


# ── Mock pipeline (no GPU required) ─────────────────────────────────────────

def _mock_pipeline_streaming(pdf_path: str, question: str):
    """
    Fake pipeline generator that returns realistic events with artificial delays.
    Simulates: PDF processing → Turn 1 (text_retrieval) → Turn 2 (page_retrieval)
               → Turn 3 (answer).
    """
    pdf_name = Path(pdf_path).name

    time.sleep(0.5)
    yield {"type": "status", "message": f"Processing PDF document: {pdf_name}"}

    time.sleep(1.2)
    yield {"type": "status", "message": "PDF processed. Starting inference..."}

    # ── Turn 1: text_retrieval ──
    time.sleep(0.4)
    yield {"type": "turn_start", "turn": 1, "total": 3}

    time.sleep(0.6)
    yield {"type": "model_thinking", "turn": 1}

    time.sleep(1.5)
    yield {
        "type": "tool_call",
        "turn": 1,
        "tool": "text_retrieval",
        "query": question[:80] if len(question) > 80 else question,
        "thinking": (
            "The user is asking about specific content in the document. "
            "I need to retrieve relevant text chunks first to understand the context "
            "before formulating a grounded answer."
        ),
    }

    time.sleep(0.8)
    yield {
        "type": "tool_result",
        "turn": 1,
        "tool": "text_retrieval",
        "results": [
            {"chunk_id": 3,  "page_id": 2,    "score": 0.8921, "snippet": "This section introduces the proposed methodology for multimodal document understanding..."},
            {"chunk_id": 7,  "page_id": 4,    "score": 0.8734, "snippet": "The experimental results demonstrate a significant improvement over baseline models..."},
            {"chunk_id": 12, "page_id": 7,    "score": 0.8501, "snippet": "Table 2 summarizes the ablation study results across all evaluation datasets..."},
            {"chunk_id": 1,  "page_id": 1,    "score": 0.8123, "snippet": "Abstract: We present a novel approach to document question answering that combines..."},
            {"chunk_id": 19, "page_id": 11,   "score": 0.7896, "snippet": "In conclusion, our framework effectively leverages both textual and visual cues..."},
        ],
    }

    # ── Turn 2: page_retrieval ──
    time.sleep(0.4)
    yield {"type": "turn_start", "turn": 2, "total": 3}

    time.sleep(0.6)
    yield {"type": "model_thinking", "turn": 2}

    time.sleep(1.8)
    yield {
        "type": "tool_call",
        "turn": 2,
        "tool": "page_retrieval",
        "query": f"figures and tables related to {question[:50]}",
        "thinking": (
            "The retrieved text chunks mention Table 2 and figures. "
            "I should also retrieve the relevant page images to verify the visual content "
            "and obtain precise data from tables or diagrams."
        ),
    }

    time.sleep(0.9)
    yield {
        "type": "tool_result",
        "turn": 2,
        "tool": "page_retrieval",
        "results": [
            {"page_num": 4,  "score": 0.9312},
            {"page_num": 7,  "score": 0.8847},
        ],
    }

    # ── Turn 3: answer ──
    time.sleep(0.4)
    yield {"type": "turn_start", "turn": 3, "total": 3}

    time.sleep(0.6)
    yield {"type": "model_thinking", "turn": 3}

    time.sleep(2.0)
    answer = (
        f"Based on the retrieved content from \"{pdf_name}\", here is a comprehensive answer "
        f"to your question: \"{question}\"\n\n"
        "The document presents a multimodal retrieval-augmented generation framework that "
        "combines dense text retrieval with page-level visual retrieval. According to the "
        "experimental results (Table 2, Page 7), the proposed method achieves state-of-the-art "
        "performance on multiple document QA benchmarks, with a notable improvement of +4.3% "
        "on DocVQA and +3.1% on MP-DocVQA compared to the previous best model.\n\n"
        "The key findings are:\n"
        "1. Joint text and visual retrieval consistently outperforms text-only baselines.\n"
        "2. The multi-turn tool-calling strategy allows the model to iteratively refine its "
        "retrieval queries, leading to more focused and accurate answers.\n"
        "3. Page image retrieval is especially beneficial for questions involving tables, "
        "figures, and mathematical formulas.\n\n"
        "(Note: this is a mock answer for UI testing — no real model was invoked.)"
    )
    yield {"type": "answer", "content": answer, "turn_count": 3}


def _mock_get_response(pdf_path: str, question: str) -> str:
    events = list(_mock_pipeline_streaming(pdf_path, question))
    for e in reversed(events):
        if e.get("type") == "answer":
            return e["content"]
    return ""

app = FastAPI(title="Document Chat Interface")

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def get_response(pdf_path: str, question: str) -> str:
    """Blocking call to the pipeline (real or mock). Returns the answer string."""
    if MOCK_MODE:
        return _mock_get_response(pdf_path, question)
    answer, _records, _tokens = run_pipeline(pdf_path, question)
    return answer


def _get_streaming_generator(pdf_path: str, question: str):
    """Return the appropriate streaming generator based on MOCK_MODE."""
    if MOCK_MODE:
        return _mock_pipeline_streaming(pdf_path, question)
    return run_pipeline_streaming(pdf_path, question)


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Frontend not found</h1>")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse(
            status_code=400,
            content={"error": "Only PDF files are supported"},
        )
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    log.info(f"File uploaded: {file_path}")
    return JSONResponse(content={"filename": file.filename, "path": str(file_path)})


@app.post("/chat")
async def chat(pdf_path: str = Form(...), question: str = Form(...)):
    """Non-streaming endpoint — returns the final answer as JSON."""
    if not os.path.exists(pdf_path):
        return JSONResponse(
            status_code=400,
            content={"error": "PDF file not found, please upload again"},
        )
    if not question or not question.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "Question cannot be empty"},
        )
    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(None, get_response, pdf_path, question)
    return JSONResponse(content={"answer": answer})


@app.post("/chat/stream")
async def chat_stream(pdf_path: str = Form(...), question: str = Form(...)):
    """
    SSE streaming endpoint.  Yields ``data: <json>\\n\\n`` lines.
    The client reads them with fetch + ReadableStream (not EventSource,
    since EventSource only supports GET).
    """
    if not os.path.exists(pdf_path):
        async def _err():
            yield f'data: {json.dumps({"type": "error", "message": "PDF file not found"}, ensure_ascii=False)}\n\n'
        return StreamingResponse(_err(), media_type="text/event-stream")

    if not question or not question.strip():
        async def _err():
            yield f'data: {json.dumps({"type": "error", "message": "Question cannot be empty"}, ensure_ascii=False)}\n\n'
        return StreamingResponse(_err(), media_type="text/event-stream")

    def _safe_next(gen):
        """Advance the generator; return None on exhaustion."""
        try:
            return next(gen)
        except StopIteration:
            return None

    async def event_generator():
        loop = asyncio.get_event_loop()
        gen = _get_streaming_generator(pdf_path, question)
        while True:
            event = await loop.run_in_executor(None, _safe_next, gen)
            if event is None:
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    uvicorn.run("chat_app:app", host="0.0.0.0", port=8000, reload=False)
