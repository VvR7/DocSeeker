import os
import logging
import shutil
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("chat-app")

app = FastAPI(title="Document Chat Interface")

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def get_response(pdf_path: str, question: str) -> str:
    return "response from zhouyi"


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
    answer = get_response(pdf_path, question)
    return JSONResponse(content={"answer": answer})


if __name__ == "__main__":
    uvicorn.run("chat_app:app", host="0.0.0.0", port=8000, reload=True)