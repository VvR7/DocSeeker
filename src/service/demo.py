from client import ColQwenClient
from PIL import Image
from pathlib import Path

client = ColQwenClient("http://localhost:8787")

folder=Path("/data3/zdw/Doc/2604.17087")
jpg_paths = sorted(
    folder.glob("*.jpg"),
    key=lambda p: int(p.stem.replace("page", ""))  # "page10" -> 10
)
print(jpg_paths)
# Your inputs
images = [
    Image.open(x) for x in jpg_paths
]
queries = [
    "What is figure 2 mainly about?",
]

img_embs = client.embed_images(images)   # np.ndarray (N, seq_len, dim)
qry_embs = client.embed_queries(queries) # np.ndarray (M, seq_len, dim)

scores = client.score(images, queries)   # (M, N)  直接得分矩阵
print(scores)  # 等价于原来的 processor.score_multi_vector