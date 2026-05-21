"""
Retrieval utilities for text chunks and document pages.

TextRetriever  — cosine similarity over Qwen3-Embedding vectors.
PageRetriever  — MaxSim late-interaction score over ColPali patch embeddings.
"""

import logging
from typing import Any, Dict, List

import numpy as np

import config
from colpali_embedding import colpali_maxsim_score, get_colpali_query_embedding
from embedding import EmbeddingEncoder

logger = logging.getLogger(__name__)


# ─── Text retriever ───────────────────────────────────────────────────────────

class TextRetriever:
    """
    Retrieve the top-k most relevant text chunks for a query string.

    Each chunk is expected to be a dict with at least:
        chunk_id  (int)
        text      (str)
        page_id   (int, 0-indexed, optional — included when present)
        embedding (np.ndarray, L2-normalised, shape (hidden_dim,))
    """

    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        encoder: EmbeddingEncoder,
    ) -> None:
        self._chunks = chunks
        self._encoder = encoder
        # Pre-stack embeddings for fast batch dot-product
        if chunks:
            self._matrix = np.stack(
                [c["embedding"] for c in chunks], axis=0
            )  # (num_chunks, dim)
        else:
            self._matrix = np.empty((0,), dtype=np.float32)

    def retrieve(self, query: str, k: int = config.K1) -> List[Dict[str, Any]]:
        """
        Return the top-*k* chunks most similar to *query*, sorted by
        descending cosine similarity.

        Returns a list of dicts:
            chunk_id   (int)
            text       (str)
            page_id    (int or None)  — 0-indexed source page, if tracked
            score      (float)
        """
        if not self._chunks:
            logger.warning("TextRetriever has no chunks.")
            return []

        query_emb = self._encoder.encode(query)  # (dim,)
        # query_emb and chunk embeddings are L2-normalised → dot product = cosine sim
        scores = self._matrix @ query_emb  # (num_chunks,)

        k = min(k, len(self._chunks))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            chunk = self._chunks[idx]
            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "page_id": chunk.get("page_id"),
                    "score": float(scores[idx]),
                }
            )

        logger.info(
            f"TextRetriever: query={repr(query[:60])}, "
            f"top-{k} chunk_ids={[r['chunk_id'] for r in results]}, "
            f"scores={[round(r['score'], 4) for r in results]}"
        )
        return results


# ─── Page retriever ───────────────────────────────────────────────────────────

class PageRetriever:
    """
    Retrieve the top-k most relevant document pages for a query string.

    Each page is expected to be a dict with at least:
        page_num   (int, 0-indexed)
        image      (PIL.Image.Image)
        embedding  (np.ndarray shape (num_patches, dim), or None if unavailable)
    """

    def __init__(self, pages: List[Dict[str, Any]]) -> None:
        self._pages = pages

    def retrieve(self, query: str, k: int = config.K2) -> List[Dict[str, Any]]:
        """
        Return the top-*k* pages most relevant to *query*, sorted by
        descending MaxSim score.

        Returns a list of dicts:
            page_num  (int, 0-indexed)
            image     (PIL.Image.Image)
            score     (float)

        Raises:
            RuntimeError: if ColPali embeddings are unavailable (embedding is None).
        """
        if not self._pages:
            logger.warning("PageRetriever has no pages.")
            return []

        # Validate that embeddings are available
        if self._pages[0]["embedding"] is None:
            raise RuntimeError(
                "ColPali page embeddings are not available. "
                "Implement get_colpali_image_embedding() in colpali_embedding.py."
            )

        # Compute ColPali query embedding
        query_embeddings = get_colpali_query_embedding([query])
        query_emb = query_embeddings[0]  # (num_tokens, dim)

        scores = np.array(
            [
                colpali_maxsim_score(query_emb, page["embedding"])
                for page in self._pages
            ]
        )

        k = min(k, len(self._pages))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            page = self._pages[idx]
            results.append(
                {
                    "page_num": page["page_num"],
                    "image": page["image"],
                    "score": float(scores[idx]),
                }
            )

        logger.info(
            f"PageRetriever: query={repr(query[:60])}, "
            f"top-{k} page_nums={[r['page_num'] for r in results]}, "
            f"scores={[round(r['score'], 4) for r in results]}"
        )
        return results
