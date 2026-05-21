"""
ColPali embedding interface implemented via the ColQwen HTTP service.

The service (service/server.py) must be running before this module is used.
Start it with:
    python service/server.py --model <model_path> --device cuda:0 --port 8787

Both public functions return a *list* of per-sample 2-D arrays so callers
(pdf_processor, retrieval) can iterate over them naturally:
    image embeddings : List[np.ndarray]  each (seq_len, dim)
    query embeddings : List[np.ndarray]  each (seq_len, dim)
"""

import logging
import os
import sys
from typing import List

import numpy as np
import PIL.Image

# Make `service/` importable regardless of the working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "service"))
from client import ColQwenClient  # noqa: E402

import config

logger = logging.getLogger(__name__)

# Module-level singleton — created once on first import
_client: ColQwenClient | None = None


def _get_client() -> ColQwenClient:
    global _client
    if _client is None:
        logger.info(
            f"Connecting to ColPali service at {config.COLPALI_SERVER_URL} ..."
        )
        _client = ColQwenClient(
            base_url=config.COLPALI_SERVER_URL,
            timeout=config.COLPALI_TIMEOUT,
        )
    return _client


# ─── Public interfaces ────────────────────────────────────────────────────────

def get_colpali_image_embedding(
    images: List[PIL.Image.Image],
) -> List[np.ndarray]:
    """
    Compute ColPali patch-level embeddings for a batch of images via the
    HTTP service.

    Args:
        images: List of PIL images (one per document page).

    Returns:
        List of float32 numpy arrays, one per image.
        Each array has shape (seq_len, dim).
    """
    client = _get_client()
    logger.info(f"Requesting image embeddings for {len(images)} page(s) ...")
    # embed_images returns (N, seq_len, dim)
    batch: np.ndarray = client.embed_images(images)
    result = [batch[i] for i in range(batch.shape[0])]
    logger.info(
        f"Image embeddings received: {len(result)} arrays, "
        f"shape per array = {result[0].shape if result else 'n/a'}"
    )
    return result


def get_colpali_query_embedding(
    queries: List[str],
) -> List[np.ndarray]:
    """
    Compute ColPali token-level embeddings for a batch of query strings via
    the HTTP service.

    Args:
        queries: List of query strings.

    Returns:
        List of float32 numpy arrays, one per query.
        Each array has shape (seq_len, dim).
    """
    client = _get_client()
    logger.info(f"Requesting query embeddings for {len(queries)} query/ies ...")
    # embed_queries returns (N, seq_len, dim)
    batch: np.ndarray = client.embed_queries(queries)
    result = [batch[i] for i in range(batch.shape[0])]
    logger.info(
        f"Query embeddings received: {len(result)} arrays, "
        f"shape per array = {result[0].shape if result else 'n/a'}"
    )
    return result


# ─── Scoring helper ───────────────────────────────────────────────────────────

def colpali_maxsim_score(
    query_embedding: np.ndarray,
    image_embedding: np.ndarray,
) -> float:
    """
    MaxSim late-interaction score between a single query and a single page.

    Mirrors the normalisation logic in ColQwenClient._maxsim so that scores
    are consistent whether computed here or via client.score().

    Args:
        query_embedding: float32 array of shape (seq_len_q, dim).
        image_embedding: float32 array of shape (seq_len_i, dim).

    Returns:
        Scalar relevance score (higher = more relevant).
    """
    # L2-normalise each token vector (ColPali model output is usually already
    # normalised, but this makes the function robust to any upstream change)
    q = query_embedding / (
        np.linalg.norm(query_embedding, axis=-1, keepdims=True) + 1e-9
    )
    d = image_embedding / (
        np.linalg.norm(image_embedding, axis=-1, keepdims=True) + 1e-9
    )
    # (seq_len_q, seq_len_i)
    sim_matrix = q @ d.T
    # For each query token: max similarity over all image tokens; then sum
    return float(sim_matrix.max(axis=-1).sum())
