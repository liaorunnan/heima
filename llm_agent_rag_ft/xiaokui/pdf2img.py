from unittest import TestCase
import pymupdf
import os


if __name__ == '__main__':
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    doc = pymupdf.open("./data/大学英语四级词汇带音标-乱序版.pdf")
    for page in doc:
        pix = page.get_pixmap()
        filepath = os.path.join(output_dir, f"page-{page.number}.png")
        pix.save(filepath)