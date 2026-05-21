"""
Hugging Face Transformers backend for local VLM inference.

Supported models (set via config.MODEL_BACKEND):
    "qwen3vl"  – Qwen3-VL  (Qwen3VLForConditionalGeneration)
    "qwen25vl" – Qwen2.5-VL (Qwen2_5_VLForConditionalGeneration)
    "llava_ov" – LLaVA-OneVision-1.5 (AutoModelForCausalLM + trust_remote_code)

The selected model is lazily initialised on the first call.

Public API:
    get_response(messages, max_new_tokens, repetition_penalty) -> str
    count_tokens(messages) -> int

Message format expected by callers (OpenAI-style):
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": <PIL.Image>},
                {"type": "text",  "text": "..."},
            ],
        },
        ...
    ]
"""

import logging

import config

logger = logging.getLogger(__name__)

# ─── Lazy-loaded model state ──────────────────────────────────────────────────

_qwen3_model      = None
_qwen3_processor  = None

_qwen25_model     = None
_qwen25_processor = None

_llava_ov_model     = None
_llava_ov_processor = None


# ─── Backend loaders ─────────────────────────────────────────────────────────

def _ensure_qwen3vl() -> None:
    global _qwen3_model, _qwen3_processor
    if _qwen3_model is not None:
        return
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    logger.info(f"Loading Qwen3-VL from {config.QWEN_MODEL_PATH} ...")
    _qwen3_model = Qwen3VLForConditionalGeneration.from_pretrained(
        config.QWEN_MODEL_PATH, torch_dtype="auto", device_map="auto"
    )
    _qwen3_processor = AutoProcessor.from_pretrained(config.QWEN_MODEL_PATH)
    logger.info("Qwen3-VL loaded.")


def _ensure_qwen25vl() -> None:
    global _qwen25_model, _qwen25_processor
    if _qwen25_model is not None:
        return
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    logger.info(f"Loading Qwen2.5-VL from {config.QWEN25_MODEL_PATH} ...")
    _qwen25_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.QWEN25_MODEL_PATH, torch_dtype="auto", device_map="auto"
    )
    _qwen25_processor = AutoProcessor.from_pretrained(config.QWEN25_MODEL_PATH)
    logger.info("Qwen2.5-VL loaded.")


def _ensure_llava_ov() -> None:
    global _llava_ov_model, _llava_ov_processor
    if _llava_ov_model is not None:
        return
    from transformers import AutoModelForCausalLM, AutoProcessor
    logger.info(f"Loading LLaVA-OneVision from {config.LLAVA_OV_MODEL_PATH} ...")
    _llava_ov_model = AutoModelForCausalLM.from_pretrained(
        config.LLAVA_OV_MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    _llava_ov_processor = AutoProcessor.from_pretrained(
        config.LLAVA_OV_MODEL_PATH, trust_remote_code=True
    )
    logger.info("LLaVA-OneVision loaded.")


# ─── Shared Qwen inference helpers ───────────────────────────────────────────

def _qwen_get_response(model, processor, messages, max_new_tokens, repetition_penalty) -> str:
    from qwen_vl_utils import process_vision_info
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    logger.info(f"Input tokens: {inputs.input_ids.shape[1]}")
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def _qwen_count_tokens(processor, messages) -> int:
    from qwen_vl_utils import process_vision_info
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return int(inputs.input_ids.shape[1])


# ─── Public interface ─────────────────────────────────────────────────────────

def get_response(
    messages: list,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
    repetition_penalty: float = config.REPETITION_PENALTY,
) -> str:
    """
    Run local Transformers inference with the model selected by config.MODEL_BACKEND.

    Args:
        messages:           OpenAI/Qwen chat-format message list.
                            Image items use {"type": "image", "image": PIL.Image}.
        max_new_tokens:     Maximum tokens to generate.
        repetition_penalty: Repetition penalty applied during generation.

    Returns:
        The model's text response as a plain string.
    """
    backend = config.MODEL_BACKEND

    if backend == "qwen3vl":
        _ensure_qwen3vl()
        return _qwen_get_response(
            _qwen3_model, _qwen3_processor, messages, max_new_tokens, repetition_penalty
        )

    if backend == "qwen25vl":
        _ensure_qwen25vl()
        return _qwen_get_response(
            _qwen25_model, _qwen25_processor, messages, max_new_tokens, repetition_penalty
        )

    if backend == "llava_ov":
        _ensure_llava_ov()
        return _qwen_get_response(
            _llava_ov_model, _llava_ov_processor, messages, max_new_tokens, repetition_penalty
        )

    raise ValueError(
        f"Unknown MODEL_BACKEND {backend!r}. "
        "Choose from: 'qwen3vl', 'qwen25vl', 'llava_ov'."
    )


def count_tokens(messages: list) -> int:
    """Return the approximate number of input tokens for the active model."""
    backend = config.MODEL_BACKEND

    if backend == "qwen3vl":
        _ensure_qwen3vl()
        return _qwen_count_tokens(_qwen3_processor, messages)

    if backend == "qwen25vl":
        _ensure_qwen25vl()
        return _qwen_count_tokens(_qwen25_processor, messages)

    if backend == "llava_ov":
        _ensure_llava_ov()
        return _qwen_count_tokens(_llava_ov_processor, messages)

    raise ValueError(
        f"Unknown MODEL_BACKEND {backend!r}. "
        "Choose from: 'qwen3vl', 'qwen25vl', 'llava_ov'."
    )
