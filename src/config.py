# ─── Inference engine ─────────────────────────────────────────────────────────
# Select the inference runtime.
# "transformers" – Hugging Face Transformers (single-process, lower setup cost)
# "vllm"         – vLLM engine (PagedAttention, higher throughput)
INFERENCE_ENGINE = "vllm"   # "transformers" | "vllm"

# ─── Model backend ────────────────────────────────────────────────────────────
# Select which VLM to use.
# Options: "qwen3vl" | "qwen25vl" | "llava_ov"
# Note: "llava_ov" is only supported with INFERENCE_ENGINE = "transformers".
MODEL_BACKEND = "qwen3vl"

# ─── Model paths ──────────────────────────────────────────────────────────────
QWEN_MODEL_PATH      = "/HOME/sysu_gbli2/sysu_gbli2xy_1/HDD_POOL/zdw/model/Qwen3-VL-8B-Instruct"
QWEN25_MODEL_PATH    = "/HOME/sysu_gbli2/sysu_gbli2xy_1/HDD_POOL/zdw/model/Qwen2.5-VL-3B-Instruct"
LLAVA_OV_MODEL_PATH  = "/HOME/sysu_gbli2/sysu_gbli2xy_1/HDD_POOL/zdw/model/LLaVA-OneVision-1.5-8B-Instruct"
EMBEDDING_MODEL_PATH = "/HOME/sysu_gbli2/sysu_gbli2xy_1/HDD_POOL/zdw/model/Qwen3-Embedding-0.6B"

# ─── ColPali service ──────────────────────────────────────────────────────────
# URL of the running colqwen_server.py  (see service/server.py)
COLPALI_SERVER_URL = "http://localhost:8788"
# HTTP timeout in seconds for ColPali requests (increase for large batches)
COLPALI_TIMEOUT = 240

# ─── Retrieval hyperparameters ────────────────────────────────────────────────
K1 = 5   # number of text chunks to retrieve per query
K2 = 2   # number of page images to retrieve per query

# ─── Pipeline hyperparameters ─────────────────────────────────────────────────
# MAX_TURN: total number of model calls allowed.
# On the final call the "last turn" notice is injected before the model responds.
MAX_TURN = 5

# ─── Text chunking ────────────────────────────────────────────────────────────
CHUNK_SIZE = 1024    # target chunk size in characters
CHUNK_OVERLAP = 128  # overlap between consecutive chunks

# ─── Inference ────────────────────────────────────────────────────────────────
# Switch between "local" (run model locally) and "api" (DashScope OpenAI-compatible)
INFERENCE_MODE = "local"   # "local" | "api"

MAX_NEW_TOKENS = 2048
REPETITION_PENALTY = 1.05

# ─── vLLM-specific settings (used when INFERENCE_ENGINE == "vllm") ────────────
# KV-cache pre-allocation length. Qwen3-VL default is 128K which causes OOM
# on most setups; 32768 comfortably covers typical evaluation contexts.
VLLM_MAX_MODEL_LEN = 81920
# Fraction of GPU VRAM reserved for vLLM (model weights + KV cache).
# 0.85 leaves headroom compared to the vLLM default of 0.90.
VLLM_GPU_MEMORY_UTILIZATION = 0.7
# Number of GPUs for tensor parallelism.
# None = auto-detect from CUDA_VISIBLE_DEVICES (recommended).
# Set explicitly (e.g. 2) when running with CUDA_VISIBLE_DEVICES to avoid
# torch.cuda.device_count() returning the full GPU count in vLLM worker processes.
VLLM_TENSOR_PARALLEL_SIZE = None



# ─── Page image rendering ─────────────────────────────────────────────────────
PAGE_RENDER_DPI = 200  # DPI used when rasterising PDF pages to PIL images



