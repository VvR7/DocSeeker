"""
ColQwen2.5 embedding server
Usage:
    python colqwen_server.py --model /data3/zdw/Doc/colqwen2.5-v0.2 --device cuda:0 --port 8787
"""

import argparse
import base64
import io
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers.utils.import_utils import is_flash_attn_2_available

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("colqwen-server")

# ── global state ──────────────────────────────────────────────────────────────
_model = None
_processor = None


def load_model(model_path: str, device: str):
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

    global _model, _processor
    log.info(f"Loading model from {model_path} on {device} …")
    _model = ColQwen2_5.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation=(
            "flash_attention_2" if is_flash_attn_2_available() else None
        ),
    ).eval()
    _processor = ColQwen2_5_Processor.from_pretrained(model_path)
    log.info("Model ready.")


# ── request / response schemas ────────────────────────────────────────────────
class ImageEmbedRequest(BaseModel):
    # Each entry is a base64-encoded image (PNG/JPEG/…)
    images_b64: List[str]
    batch_size: Optional[int] = 4       # process this many images per forward pass


class QueryEmbedRequest(BaseModel):
    queries: List[str]
    batch_size: Optional[int] = 16


class EmbedResponse(BaseModel):
    # shape: [N, seq_len, dim]  – multi-vector ColPali embeddings
    embeddings: List[List[List[float]]]
    shape: List[int]


# ── helpers ───────────────────────────────────────────────────────────────────
def _decode_b64_image(b64: str) -> Image.Image:
    try:
        data = base64.b64decode(b64)
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad base64 image: {e}")


def _embed_in_batches(items, process_fn, batch_size: int) -> torch.Tensor:
    """Run inference in chunks to avoid OOM on large lists."""
    all_vecs = []
    for i in range(0, len(items), batch_size):
        chunk = items[i : i + batch_size]
        batch = process_fn(chunk).to(_model.device)
        with torch.no_grad():
            vecs = _model(**batch)           # (B, seq_len, dim)
        all_vecs.append(vecs.cpu().float())
    return torch.cat(all_vecs, dim=0)       # (N, seq_len, dim)


# ── lifespan (load model before first request) ────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    args = app.state.args
    load_model(args.model, args.device)
    yield


app = FastAPI(title="ColQwen2.5 Embedding Service", lifespan=lifespan)


# ── routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/embed/images", response_model=EmbedResponse)
def embed_images(req: ImageEmbedRequest):
    if _model is None:
        raise HTTPException(503, "Model not loaded yet")

    images = [_decode_b64_image(b) for b in req.images_b64]
    vecs = _embed_in_batches(
        images,
        lambda imgs: _processor.process_images(imgs),
        req.batch_size,
    )
    return EmbedResponse(
        embeddings=vecs.tolist(),
        shape=list(vecs.shape),
    )


@app.post("/embed/queries", response_model=EmbedResponse)
def embed_queries(req: QueryEmbedRequest):
    if _model is None:
        raise HTTPException(503, "Model not loaded yet")

    vecs = _embed_in_batches(
        req.queries,
        lambda qs: _processor.process_queries(qs),
        req.batch_size,
    )
    return EmbedResponse(
        embeddings=vecs.tolist(),
        shape=list(vecs.shape),
    )


@app.post("/score")
def score(image_req: ImageEmbedRequest, query_req: QueryEmbedRequest):
    """Convenience: embed both and return the ColPali multi-vector score matrix."""
    img_resp = embed_images(image_req)
    qry_resp = embed_queries(query_req)

    img_t = torch.tensor(img_resp.embeddings)   # (N_img, seq, dim)
    qry_t = torch.tensor(qry_resp.embeddings)   # (N_qry, seq, dim)
    scores = _processor.score_multi_vector(qry_t, img_t)
    return {"scores": scores.tolist(), "shape": list(scores.shape)}


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="/HOME/sysu_gbli2/sysu_gbli2xy_1/HDD_POOL/zdw/Docproject/colqwen2.5-v0.2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port",   type=int, default=8787)
    parser.add_argument("--host",   default="0.0.0.0")
    args = parser.parse_args()

    app.state.args = args
    uvicorn.run(app, host=args.host, port=args.port)