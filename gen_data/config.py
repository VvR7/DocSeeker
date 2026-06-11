"""
Benchmark 数据生成配置文件
所有路径和超参数均可通过同名环境变量覆盖。
"""
import os

# ── 模型 ──────────────────────────────────────────────────────────────────────
QWEN_MODEL_PATH: str = os.environ.get(
    "QWEN_MODEL_PATH",
    "/HOME/sysu_gbli2/sysu_gbli2xy_1/HDD_POOL/zdw/model/Qwen3-VL-32B-Instruct",
)

# ── 数据目录 ──────────────────────────────────────────────────────────────────
# 论文页面图片根目录，结构: {PAPER_IMG_DIR}/{paper_name}/page{idx}.jpg
PAPER_IMG_DIR: str = os.environ.get(
    "PAPER_IMG_DIR",
    "./paper_img",
)

# 输出根目录
OUTPUT_DIR: str = os.environ.get(
    "OUTPUT_DIR",
    "./output",
)

# ── vLLM 推理超参数 ───────────────────────────────────────────────────────────
MAX_MODEL_LEN: int = int(os.environ.get("MAX_MODEL_LEN", "32768"))
GPU_MEMORY_UTILIZATION: float = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.8"))
DTYPE: str = os.environ.get("DTYPE", "bfloat16")

# 题目生成阶段的最大输出 token 数
MAX_NEW_TOKENS_GENERATE: int = int(os.environ.get("MAX_NEW_TOKENS_GENERATE", "2048"))
# Review 阶段的最大输出 token 数（只需输出答案字母）
MAX_NEW_TOKENS_REVIEW: int = int(os.environ.get("MAX_NEW_TOKENS_REVIEW", "64"))
# 参考文献检测阶段的最大输出 token 数
MAX_NEW_TOKENS_REFS: int = int(os.environ.get("MAX_NEW_TOKENS_REFS", "64"))
