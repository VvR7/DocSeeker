"""
Qwen3-Embedding-0.6B text encoder.
Uses last-token pooling + L2 normalisation, consistent with the official recommendation.
"""

import logging

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

import config

logger = logging.getLogger(__name__)


class EmbeddingEncoder:
    def __init__(self, model_name_or_path: str = config.EMBEDDING_MODEL_PATH):
        logger.info(f"Loading embedding model from {model_name_or_path} ...")
        # padding_side='left' is the standard choice for last-token pooling
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, padding_side="left"
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path, torch_dtype=torch.float16
        )
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        logger.info("Embedding model loaded.")

    @staticmethod
    def _last_token_pool(
        last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Take the hidden state of the last valid token as the sentence vector."""
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_state[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_state.shape[0]
        return last_hidden_state[
            torch.arange(batch_size, device=last_hidden_state.device),
            sequence_lengths,
        ]

    def encode(self, text: str, max_length: int = 512) -> np.ndarray:
        """
        Encode a text string into an L2-normalised embedding vector.

        Returns:
            float32 numpy array of shape (hidden_dim,)
        """
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded)
            embedding = self._last_token_pool(
                outputs.last_hidden_state, encoded["attention_mask"]
            )
            embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding.cpu().float().numpy()[0]

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two already-normalised vectors."""
        return float(np.dot(a, b))
