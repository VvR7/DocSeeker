"""
PDF processing pipeline.

Steps:
  1. Extract the full text via PyMuPDF (fitz) and split it into overlapping
     chunks, recording the 0-indexed page_id for each chunk.
  2. Encode each chunk with Qwen3-Embedding-0.6B.
  3. Render every page to a PIL image via PyMuPDF.
  4. Encode every page image with ColPali.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import PIL.Image

import config
from colpali_embedding import get_colpali_image_embedding
from embedding import EmbeddingEncoder

logger = logging.getLogger(__name__)


# ─── Text chunking ────────────────────────────────────────────────────────────

_DEFAULT_SEPARATORS: List[str] = ["\n\n", "\n", ".", "!", "?", "。", "，", " ", ""]


def _chunk_text_with_starts(
    text: str,
    chunk_size: int,
    overlap: int,
    separators: List[str] = _DEFAULT_SEPARATORS,
) -> List[Tuple[int, str]]:
    """
    Split *text* into overlapping chunks, preferring to break at natural
    boundaries defined by *separators* (tried in decreasing priority order).

    For each candidate window ``[start, start + chunk_size]``, the function
    searches backwards for the highest-priority separator that exists strictly
    inside the window.  The chunk ends just after that separator.  If no
    separator is found, the chunk is cut at the hard ``chunk_size`` boundary
    (the empty string ``""`` sentinel in *separators* triggers this fallback
    explicitly).

    Overlap is applied as a character-level back-step from the split point so
    that adjacent chunks share approximately *overlap* characters.

    Returns a list of (start_offset, chunk_text) tuples so that callers can
    map each chunk back to its source page.
    """
    if not text.strip():
        return []

    result: List[Tuple[int, str]] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end == len(text):
            chunk = text[start:end].strip()
            if chunk:
                result.append((start, chunk))
            break

        # Find the best split point near `end` using separator priority.
        # We search the window (start, end] — using pos > start to guarantee
        # at least one character of forward progress per iteration.
        split_pos = end  # default: hard cut
        for sep in separators:
            if sep == "":
                break  # hard-cut fallback already assigned above
            pos = text.rfind(sep, start + 1, end)
            if pos != -1:
                split_pos = pos + len(sep)
                break

        chunk = text[start:split_pos].strip()
        if chunk:
            result.append((start, chunk))

        next_start = split_pos - overlap
        if next_start <= start:
            next_start = split_pos  # guard: always move forward
        start = next_start

    return result


def _find_page_id(
    char_offset: int, page_offsets: List[Tuple[int, int]]
) -> int:
    """
    Return the 0-indexed page_id that owns the character at *char_offset*.

    *page_offsets* is a sorted list of (start_offset, page_id) pairs.
    The page with the largest start_offset that is still ≤ char_offset wins.
    """
    page_id = page_offsets[0][1]
    for start, pid in page_offsets:
        if start <= char_offset:
            page_id = pid
        else:
            break
    return page_id


# ─── PDF processor ────────────────────────────────────────────────────────────

class PDFProcessor:
    """
    Processes a PDF file into:
      - text_chunks: list of dicts with keys
            chunk_id  (int, 0-indexed)
            text      (str)
            page_id   (int, 0-indexed PDF page the chunk originates from)
            embedding (np.ndarray, shape (hidden_dim,))
      - pages: list of dicts with keys
            page_num   (int, 0-indexed)
            image      (PIL.Image.Image)
            embedding  (np.ndarray shape (num_patches, dim), or None if ColPali unavailable)
    """

    def __init__(self) -> None:
        self._encoder = EmbeddingEncoder()

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process the PDF at *pdf_path* and return a dict with keys
        ``text_chunks`` and ``pages``.
        """
        logger.info(f"Processing PDF: {pdf_path}")
        text_chunks = self._extract_and_embed_text(pdf_path)
        pages = self._extract_and_embed_pages(pdf_path)
        logger.info(
            f"PDF processed: {len(text_chunks)} text chunks, {len(pages)} pages."
        )
        return {"text_chunks": text_chunks, "pages": pages}

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_and_embed_text(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text, chunk it with page attribution, and embed each chunk."""
        logger.info("Extracting text from PDF via fitz ...")
        doc = fitz.open(pdf_path)

        # Collect (page_id, text) for non-empty pages
        page_info: List[Tuple[int, str]] = []
        for page_index in range(len(doc)):
            page_text = doc[page_index].get_text("text")
            if page_text.strip():
                page_info.append((page_index, page_text))
        doc.close()

        if not page_info:
            logger.warning("No text extracted from PDF.")
            return []

        # Build the full concatenated text and track each page's start offset
        page_offsets: List[Tuple[int, int]] = []  # (start_offset, page_id)
        parts: List[str] = []
        current_offset = 0
        for page_id, text in page_info:
            page_offsets.append((current_offset, page_id))
            parts.append(text)
            current_offset += len(text) + 2  # +2 for the "\n\n" separator

        md_text = "\n\n".join(parts)
        logger.info(
            f"Extracted {len(md_text)} characters from {len(page_info)} pages."
        )

        chunks_with_starts = _chunk_text_with_starts(
            md_text,
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP,
        )
        logger.info(
            f"Text split into {len(chunks_with_starts)} chunks. Embedding ..."
        )

        result: List[Dict[str, Any]] = []
        for idx, (start_offset, chunk_text) in enumerate(chunks_with_starts):
            page_id = _find_page_id(start_offset, page_offsets)
            embedding = self._encoder.encode(chunk_text)
            result.append(
                {
                    "chunk_id": idx,
                    "text": chunk_text,
                    "page_id": page_id,
                    "embedding": embedding,
                }
            )
            if (idx + 1) % 50 == 0:
                logger.info(f"  Embedded {idx + 1}/{len(chunks_with_starts)} chunks ...")

        logger.info("Text chunk embedding complete.")
        return result

    def _extract_and_embed_pages(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Render each PDF page to a PIL image and embed it with ColPali."""
        logger.info("Rendering PDF pages to images ...")
        doc = fitz.open(pdf_path)
        zoom = config.PAGE_RENDER_DPI / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        pil_images: List[PIL.Image.Image] = []
        page_nums: List[int] = []

        for page_index in range(len(doc)):
            page = doc[page_index]
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = PIL.Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            pil_images.append(img)
            page_nums.append(page_index)  # 0-indexed

        doc.close()
        logger.info(
            f"Rendered {len(pil_images)} pages. Computing ColPali embeddings ..."
        )

        colpali_embeddings: List[Optional[Any]] = [None] * len(pil_images)
        try:
            colpali_embeddings = get_colpali_image_embedding(pil_images)
            logger.info("ColPali page embeddings computed.")
        except NotImplementedError:
            logger.warning(
                "ColPali image embedding is not implemented. "
                "Page retrieval will be unavailable."
            )
        except Exception as exc:
            logger.error(
                f"Unexpected error while computing ColPali embeddings: {exc}. "
                "Page retrieval will be unavailable."
            )

        pages: List[Dict[str, Any]] = []
        for img, page_num, emb in zip(pil_images, page_nums, colpali_embeddings):
            pages.append(
                {
                    "page_num": page_num,
                    "image": img,
                    "embedding": emb,
                }
            )

        return pages



import argparse
import os


def demo_chunk_pdf(pdf_path: str, chunk_size: int, overlap: int, output_path: str):
    """
    Demo: 提取 PDF 文本并按指定参数分块，将结果写入 txt 文件。
    """
    if not os.path.exists(pdf_path):
        print(f"错误: 文件不存在: {pdf_path}")
        return

    # 提取文本
    doc = fitz.open(pdf_path)
    page_info = []
    for page_index in range(len(doc)):
        page_text = doc[page_index].get_text("text")
        if page_text.strip():
            page_info.append((page_index, page_text))
    doc.close()

    if not page_info:
        print("警告: 未从 PDF 中提取到文本")
        return

    # 构建完整文本和页码映射
    page_offsets = []
    parts = []
    current_offset = 0
    for page_id, text in page_info:
        page_offsets.append((current_offset, page_id))
        parts.append(text)
        current_offset += len(text) + 2

    full_text = "\n\n".join(parts)

    # 分块
    chunks_with_starts = _chunk_text_with_starts(full_text, chunk_size, overlap)

    # 写入结果
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"PDF 文件: {pdf_path}\n")
        f.write(f"总页数: {len(page_info)}\n")
        f.write(f"总字符数: {len(full_text)}\n")
        f.write(f"Chunk Size: {chunk_size}\n")
        f.write(f"Overlap: {overlap}\n")
        f.write(f"生成 chunks 数量: {len(chunks_with_starts)}\n")
        f.write("=" * 80 + "\n\n")

        for idx, (start_offset, chunk_text) in enumerate(chunks_with_starts):
            page_id = _find_page_id(start_offset, page_offsets)
            f.write(f"--- Chunk {idx} (Page {page_id + 1}) ---\n")
            f.write(f"起始字符偏移: {start_offset}\n")
            f.write(f"字符长度: {len(chunk_text)}\n")
            f.write(f"内容:\n{chunk_text}\n")
            f.write("\n" + "=" * 80 + "\n\n")

    print(f"分块完成！共 {len(chunks_with_starts)} 个 chunks")
    print(f"结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="PDF 文本分块 Demo")
    parser.add_argument("--pdf_path", type=str, required=True, help="PDF 文件路径")
    parser.add_argument("--chunk_size", type=int, default=512, help="每块字符数 (默认: 512)")
    parser.add_argument("--overlap", type=int, default=128, help="块间重叠字符数 (默认: 128)")
    parser.add_argument("--output", type=str, default="chunks_output.txt", help="输出文件路径 (默认: chunks_output.txt)")
    args = parser.parse_args()

    if not args.pdf_path:
        parser.error("--pdf_path 参数是必须的，请指定 PDF 文件路径。")
    demo_chunk_pdf(args.pdf_path, args.chunk_size, args.overlap, args.output)

if __name__ == "__main__":
    main()

'''
示例命令行：
python pdf_processor.py \
    --pdf_path /data3/zdw/Doc/project/2604.17087.pdf \
    --chunk_size 2048 \
    --overlap 256 \
    --output chunks_output.txt 
'''

