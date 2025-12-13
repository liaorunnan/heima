import fitz
import os
from tqdm import tqdm
import pymupdf4llm
import pathlib
from typing import Iterable


def _iter_pdf_files(path: str) -> Iterable[str]:
    """支持传入单个 PDF 或目录，返回需要处理的 PDF 列表。"""
    if os.path.isdir(path):
        return [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.lower().endswith(".pdf")
        ]
    return [path]


def pdf_to_images(pdf_path: str, image_path: str):
    """将 PDF 转为图片，支持传入目录；输出文件名为“文件名_页码.png”。"""
    pdf_files = _iter_pdf_files(pdf_path)
    os.makedirs(image_path, exist_ok=True)

    for pdf_file in pdf_files:
        pdf_doc = fitz.open(pdf_file)
        base = os.path.splitext(os.path.basename(pdf_file))[0]
        for pg in tqdm(range(pdf_doc.page_count), desc=f"{base} -> images"):
            page = pdf_doc[pg]
            rotate = 0
            zoom_x, zoom_y = 4, 4
            mat = fitz.Matrix(zoom_x, zoom_y).prerotate(rotate)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            outfile = os.path.join(image_path, f"{base}_{pg}.png")
            pix._writeIMG(format_="png", filename=outfile, jpg_quality=100)


def pdf_to_markdown(pdf_path: str, md_path: str):
    """将 PDF 转为 Markdown，支持传入目录；输出文件名为“文件名.md”。"""
    pdf_files = _iter_pdf_files(pdf_path)

    if os.path.isdir(pdf_path):
        # 输入为目录时，md_path 应为目录或不存在（会创建）。
        os.makedirs(md_path, exist_ok=True)
        for pdf_file in pdf_files:
            base = os.path.splitext(os.path.basename(pdf_file))[0]
            out_file = pathlib.Path(md_path) / f"{base}.md"
            md_text = pymupdf4llm.to_markdown(pdf_file)
            out_file.write_bytes(md_text.encode())
    else:
        # 输入为单个文件时，保留原行为：md_path 是具体文件路径
        out_file = pathlib.Path(md_path)
        os.makedirs(out_file.parent, exist_ok=True)
        md_text = pymupdf4llm.to_markdown(pdf_path)
        out_file.write_bytes(md_text.encode())


if __name__ == '__main__':
    ...
    # 示例：传入目录，输出图片文件名 = 原 PDF 文件名 + _页码
    # pdf_to_images("./writ", "./imgs/")

    # 示例：传入目录，批量生成 Markdown，文件名 = 原 PDF 文件名 + .md
    # pdf_to_markdown("./writ", "./output/md/")

    # 示例：单文件模式保持兼容
    # pdf_to_images("./writ/文书示范文本.pdf", "./imgs/")
    # pdf_to_markdown("./writ/文书示范文本.pdf", "./writ/writ.md")

    pdf_to_images("./case/", "./imgs/")