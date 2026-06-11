
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

# ── 单例缓存 ──────────────────────────────────────────────────────────────────
_llm: LLM | None = None
_processor: AutoProcessor | None = None
_init_kwargs_key: tuple | None = None   # 用于检测参数变化，决定是否重新初始化


def _get_llm_and_processor(
    model_path: str = config.QWEN_MODEL_PATH,
    max_model_len: int = 81920,
    gpu_memory_utilization: float = 0.8,
    dtype: str = "bfloat16",
) -> tuple[LLM, AutoProcessor]:
    """
    懒加载并缓存 LLM 与 AutoProcessor 单例。

    关键参数说明：
      max_model_len           -- KV cache 按此长度预分配，是显存占用的决定性因素。
                                 Qwen3-VL-8B 默认 128K，会直接 OOM；
                                 评测任务上下文一般在 32K 以内，设 32768 即可。
      gpu_memory_utilization  -- vLLM 使用的显存比例（模型权重 + KV cache 合计）。
                                 默认 0.90 容易撑满，设 0.85 留出余量。
      dtype                   -- 推理精度，显式设为 bfloat16 避免意外以 float32 加载。
    """
    global _llm, _processor, _init_kwargs_key

    key = (model_path, max_model_len, gpu_memory_utilization, dtype)
    if _llm is None or _init_kwargs_key != key:
        logger.info(
            f"初始化 vLLM LLM  model={model_path}  "
            f"max_model_len={max_model_len}  "
            f"gpu_memory_utilization={gpu_memory_utilization}  "
            f"dtype={dtype}"
        )
        _processor = AutoProcessor.from_pretrained(model_path)
        _llm = LLM(
            model=model_path,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=torch.cuda.device_count(),
            seed=0,
            # vLLM v1 的多模态 LRU 处理器缓存存在 ABA 竞争：
            # is_cached 与 get_and_update_item 之间若有新项写入触发驱逐，
            # 已确认"命中"的项会被驱逐，导致 AssertionError。
            # 设为 0 禁用 LRU 缓存，每次推理均走无缓存路径，规避此 bug。
            mm_processor_cache_gb=0,
        )
        _init_kwargs_key = key
        logger.info("vLLM LLM 初始化完成。")

    return _llm, _processor


def _prepare_inputs(messages: list, processor: AutoProcessor) -> dict:
    """
    将内部消息格式转换为 vLLM generate() 所需的输入字典。

    内部消息中图像以 base64 data URL 形式存储（data:image/jpeg;base64,...），
    process_vision_info 可直接处理该格式。
    """
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # 不传 image_patch_size：动态分辨率 patch 逻辑由 vLLM 内部完成；
    # 若在此预处理，会导致 vLLM v1 多模态 cache 哈希不一致（AssertionError）。
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


def count_tokens(
    messages: list,
    model_path: str = config.QWEN_MODEL_PATH,
    max_model_len: int = 32768,
    gpu_memory_utilization: float = 0.85,
    dtype: str = "bfloat16",
) -> int:
    """
    返回 messages 经 processor 处理后的精确输入 token 数（文本 + 图像 token 全部包含）。

    使用与 get_response 相同的 processor 流程，通过 inputs.input_ids.shape[1] 得到精确值。
    processor 由 _get_llm_and_processor 懒加载并缓存，与推理共享同一实例。
    """
    _, processor = _get_llm_and_processor(
        model_path=model_path,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
    )
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


def get_response(
    messages: list,
    max_new_tokens: int = 2048,
    repetition_penalty: float = 1.0,
    model_path: str = config.QWEN_MODEL_PATH,
    max_model_len: int = 32768,
    gpu_memory_utilization: float = 0.85,
    dtype: str = "bfloat16",
) -> str:
    """
    使用本地 vLLM 推理，返回模型生成文本。

    接口与 qwenvl.vllm_api.get_response 保持一致，可直接替换。

    Args:
        messages:               内部格式消息列表（含 base64 图像）。
        max_new_tokens:         最大生成 token 数。
        repetition_penalty:     重复惩罚系数（1.0 表示不惩罚）。
        model_path:             本地模型路径，默认 config.QWEN_MODEL_PATH。
        max_model_len:          KV cache 预分配长度，直接决定显存占用，默认 32768。
        gpu_memory_utilization: vLLM 显存使用比例，默认 0.85。
        dtype:                  推理精度，默认 bfloat16。

    Returns:
        模型生成的文本字符串。
    """
    global _llm, _init_kwargs_key

    _MAX_ENGINE_RETRIES = 2
    for attempt in range(1, _MAX_ENGINE_RETRIES + 1):
        llm, processor = _get_llm_and_processor(
            model_path=model_path,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
        )

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
