"""
vLLM backend for local VLM inference.

Supported models (set via config.MODEL_BACKEND):
    "qwen3vl"  – Qwen3-VL  → config.QWEN_MODEL_PATH
    "qwen25vl" – Qwen2.5-VL → config.QWEN25_MODEL_PATH

vLLM-specific tunables (set in config):
    VLLM_MAX_MODEL_LEN          – KV-cache pre-allocation length (default 32768)
    VLLM_GPU_MEMORY_UTILIZATION – Fraction of GPU memory used by vLLM (default 0.85)

The LLM singleton is lazily initialised and cached; it is re-created only when
the underlying configuration changes.

Public API:
    get_response(messages, max_new_tokens, repetition_penalty) -> str
    count_tokens(messages) -> int
"""

from __future__ import annotations

import logging
import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from vllm.v1.engine.exceptions import EngineDeadError

import config

logger = logging.getLogger(__name__)

# ─── Singleton cache ──────────────────────────────────────────────────────────

_llm: LLM | None = None
_processor: AutoProcessor | None = None
_init_kwargs_key: tuple | None = None


# ─── GPU count helper ────────────────────────────────────────────────────────

def _resolve_tensor_parallel_size() -> int:
    """
    Determine the number of GPUs for tensor parallelism.

    Priority order:
      1. config.VLLM_TENSOR_PARALLEL_SIZE  — explicit override (recommended when
         using CUDA_VISIBLE_DEVICES, because torch.cuda.device_count() may return
         the full machine GPU count inside vLLM worker processes spawned via
         multiprocessing, bypassing the CUDA_VISIBLE_DEVICES restriction).
      2. CUDA_VISIBLE_DEVICES env var       — parse the comma-separated list.
      3. torch.cuda.device_count()          — fallback (may be unreliable in
         vLLM workers, see above).
    """
    explicit = getattr(config, "VLLM_TENSOR_PARALLEL_SIZE", None)
    if explicit is not None:
        logger.info(f"tensor_parallel_size: {explicit} (from config.VLLM_TENSOR_PARALLEL_SIZE)")
        return int(explicit)

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible:
        n = len(cuda_visible.split(","))
        logger.info(
            f"tensor_parallel_size: {n} (parsed from CUDA_VISIBLE_DEVICES={cuda_visible!r})"
        )
        return n

    n = torch.cuda.device_count()
    logger.info(f"tensor_parallel_size: {n} (from torch.cuda.device_count())")
    return n


# ─── Model path helper ────────────────────────────────────────────────────────

def _model_path() -> str:
    backend = config.MODEL_BACKEND
    if backend == "qwen3vl":
        return config.QWEN_MODEL_PATH
    if backend == "qwen25vl":
        return config.QWEN25_MODEL_PATH
    if backend == "llava_ov":
        raise ValueError(
            "'llava_ov' is not supported with the vllm engine. "
            "Set INFERENCE_ENGINE = 'transformers' to use LLaVA-OneVision."
        )
    raise ValueError(
        f"Unknown MODEL_BACKEND {backend!r} for vllm engine. "
        "Choose from: 'qwen3vl', 'qwen25vl'."
    )


# ─── Lazy loader ─────────────────────────────────────────────────────────────

def _get_llm_and_processor() -> tuple[LLM, AutoProcessor]:
    """
    Lazily initialise and cache the LLM + AutoProcessor singleton.

    Key parameters:
      VLLM_MAX_MODEL_LEN           – KV cache is pre-allocated at this length,
                                     which is the primary driver of VRAM usage.
                                     Qwen3-VL default is 128K (OOM on many setups);
                                     32768 covers most evaluation contexts.
      VLLM_GPU_MEMORY_UTILIZATION  – vLLM's share of VRAM (weights + KV cache).
                                     0.85 leaves headroom vs. the 0.90 default.
      mm_processor_cache_gb=0      – Disables the multimodal LRU processor cache
                                     to avoid an ABA race in vLLM v1 that causes
                                     AssertionError when items are evicted between
                                     is_cached and get_and_update_item.
    """
    global _llm, _processor, _init_kwargs_key

    model_path = _model_path()
    max_model_len = config.VLLM_MAX_MODEL_LEN
    gpu_memory_utilization = config.VLLM_GPU_MEMORY_UTILIZATION
    dtype = "bfloat16"

    key = (model_path, max_model_len, gpu_memory_utilization, dtype)
    if _llm is None or _init_kwargs_key != key:
        logger.info(
            f"Initializing vLLM  model={model_path}  "
            f"max_model_len={max_model_len}  "
            f"gpu_memory_utilization={gpu_memory_utilization}  "
            f"dtype={dtype}"
        )
        tensor_parallel_size = _resolve_tensor_parallel_size()
        _processor = AutoProcessor.from_pretrained(model_path)
        _llm = LLM(
            model=model_path,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            seed=0,
            mm_processor_cache_gb=0,
        )
        _init_kwargs_key = key
        logger.info("vLLM initialized.")

    return _llm, _processor


# ─── Input preparation ────────────────────────────────────────────────────────

def _prepare_inputs(messages: list, processor: AutoProcessor) -> dict:
    """
    Convert internal message format to the input dict expected by llm.generate().

    Dynamic-resolution patch logic is handled inside vLLM; pre-processing with
    image_patch_size here would cause multimodal cache hash mismatches in
    vLLM v1 (AssertionError).
    """
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    mm_data: dict = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    return {
        "prompt": text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }


# ─── Public interface ─────────────────────────────────────────────────────────

def get_response(
    messages: list,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
    repetition_penalty: float = config.REPETITION_PENALTY,
) -> str:
    """
    Run vLLM inference and return the generated text.

    Args:
        messages:           OpenAI/Qwen chat-format message list.
        max_new_tokens:     Maximum tokens to generate.
        repetition_penalty: Repetition penalty (1.0 = disabled).

    Returns:
        The model's text response as a plain string.
    """
    global _llm, _init_kwargs_key

    _MAX_ENGINE_RETRIES = 2
    for attempt in range(1, _MAX_ENGINE_RETRIES + 1):
        llm, processor = _get_llm_and_processor()
        inp = _prepare_inputs(messages, processor)

        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=max_new_tokens,
            top_k=-1,
            stop_token_ids=[],
            repetition_penalty=repetition_penalty,
        )

        try:
            outputs = llm.generate([inp], sampling_params=sampling_params)
            input_token_count = len(outputs[0].prompt_token_ids)
            logger.info(f"Input tokens: {input_token_count}")
            return outputs[0].outputs[0].text
        except EngineDeadError:
            logger.error(
                f"vLLM EngineDeadError on attempt {attempt}/{_MAX_ENGINE_RETRIES}. "
                "Resetting LLM singleton and reinitializing ..."
            )
            _llm = None
            _init_kwargs_key = None
            if attempt == _MAX_ENGINE_RETRIES:
                raise


def count_tokens(messages: list) -> int:
    """
    Return the exact input token count (text + image tokens) for the given messages.

    Uses image_patch_size from the processor for accurate vision token counting,
    consistent with the Transformers backend.
    """
    _, processor = _get_llm_and_processor()
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    return int(inputs.input_ids.shape[1])
