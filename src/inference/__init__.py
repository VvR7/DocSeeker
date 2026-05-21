"""
Inference package — routes to the appropriate backend.

Backend is selected by config.INFERENCE_ENGINE:
    "transformers" – Hugging Face Transformers (single-process, low overhead)
    "vllm"         – vLLM engine (higher throughput, PagedAttention)

The active model is selected by config.MODEL_BACKEND:
    "qwen3vl"  – Qwen3-VL
    "qwen25vl" – Qwen2.5-VL
    "llava_ov" – LLaVA-OneVision-1.5 (transformers engine only)

Public API (identical for all backends):
    get_response(messages, max_new_tokens, repetition_penalty) -> str
    count_tokens(messages) -> int
"""

import config

if config.INFERENCE_ENGINE == "transformers":
    from .transformers_backend import get_response, count_tokens
elif config.INFERENCE_ENGINE == "vllm":
    from .vllm_backend import get_response, count_tokens
else:
    raise ValueError(
        f"Unknown INFERENCE_ENGINE {config.INFERENCE_ENGINE!r}. "
        "Choose from: 'transformers', 'vllm'."
    )

__all__ = ["get_response", "count_tokens"]
