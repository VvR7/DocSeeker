"""
ColQwen2.5 client  –  drop into any conda env (only needs: requests, Pillow, numpy)

Quick start
-----------
from colqwen_client import ColQwenClient
from PIL import Image

client = ColQwenClient("http://localhost:8788")

# Embed images
imgs = [Image.open("page1.jpg"), Image.open("page2.jpg")]
img_embs = client.embed_images(imgs)          # np.ndarray (N, seq_len, dim)

# Embed text queries
qry_embs = client.embed_queries(["What is figure 2 about?"])

# Retrieve – multi-vector MaxSim score (ColPali style)
scores = client.score(imgs, ["What is figure 2 about?"])   # (n_queries, n_images)
best_page = scores[0].argmax()
"""

import base64
import io
from typing import List, Union

import numpy as np
import requests
from PIL import Image


class ColQwenClient:
    def __init__(self, base_url: str = "http://localhost:8788", timeout: int = 120):
        """
        Parameters
        ----------
        base_url : URL of the running colqwen_server.py
        timeout  : HTTP timeout in seconds (increase for many/large images)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ── public API ────────────────────────────────────────────────────────────

    def embed_images(
        self,
        images: List[Union[Image.Image, str]],
        batch_size: int = 4,
    ) -> np.ndarray:
        """
        Embed a list of images.

        Parameters
        ----------
        images     : list of PIL.Image.Image  –or–  file paths (str)
        batch_size : server-side batch size (tune to your GPU memory)

        Returns
        -------
        np.ndarray of shape (N, seq_len, dim), dtype float32
        """
        pil_images = [self._to_pil(img) for img in images]
        b64_list = [self._pil_to_b64(img) for img in pil_images]

        resp = self._post("/embed/images", {"images_b64": b64_list, "batch_size": batch_size})
        return np.array(resp["embeddings"], dtype=np.float32)

    def embed_queries(
        self,
        queries: List[str],
        batch_size: int = 16,
    ) -> np.ndarray:
        """
        Embed a list of text queries.

        Returns
        -------
        np.ndarray of shape (N, seq_len, dim), dtype float32
        """
        resp = self._post("/embed/queries", {"queries": queries, "batch_size": batch_size})
        return np.array(resp["embeddings"], dtype=np.float32)

    def score(
        self,
        images: List[Union[Image.Image, str]],
        queries: List[str],
        batch_size_img: int = 4,
        batch_size_qry: int = 16,
    ) -> np.ndarray:
        """
        Compute ColPali multi-vector MaxSim scores.

        Returns
        -------
        np.ndarray of shape (n_queries, n_images), dtype float32
        Higher = more relevant.
        """
        img_embs = self.embed_images(images, batch_size=batch_size_img)
        qry_embs = self.embed_queries(queries, batch_size=batch_size_qry)
        return self._maxsim(qry_embs, img_embs)

    def health(self) -> dict:
        r = requests.get(f"{self.base_url}/health", timeout=10)
        r.raise_for_status()
        return r.json()

    # ── internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _to_pil(img: Union[Image.Image, str]) -> Image.Image:
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        return Image.open(img).convert("RGB")

    @staticmethod
    def _pil_to_b64(img: Image.Image, fmt: str = "JPEG") -> str:
        buf = io.BytesIO()
        img.save(buf, format=fmt, quality=92)
        return base64.b64encode(buf.getvalue()).decode()

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        try:
            r = requests.post(url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot reach ColQwen server at {self.base_url}. "
                "Is colqwen_server.py running?"
            )
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Server error {r.status_code}: {r.text}") from e

    @staticmethod
    def _maxsim(query_embs: np.ndarray, image_embs: np.ndarray) -> np.ndarray:
        """
        ColPali multi-vector score: for each query token, find the max similarity
        over all image tokens, then sum across query tokens.

        query_embs : (Q, Sq, D)
        image_embs : (I, Si, D)
        returns    : (Q, I)
        """
        # Normalize
        q = query_embs / (np.linalg.norm(query_embs, axis=-1, keepdims=True) + 1e-9)
        d = image_embs / (np.linalg.norm(image_embs, axis=-1, keepdims=True) + 1e-9)

        # (Q, Sq, D) x (I, Si, D)^T → (Q, I, Sq, Si)
        # Use einsum for clarity: score[q,i] = sum_sq max_si dot(q[q,sq], d[i,si])
        # Build (Q, I, Sq, Si) sim matrix
        scores = np.einsum("qsd,imd->qism", q, d)   # (Q, I, Sq, Si)
        scores = scores.max(axis=-1).sum(axis=-1)    # (Q, I)
        return scores.astype(np.float32)