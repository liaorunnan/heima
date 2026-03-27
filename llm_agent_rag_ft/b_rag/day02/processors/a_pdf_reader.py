from unittest import TestCase
import pymupdf
import os
#
# class Testpymupdf(TestCase):
#     def test_pdf2text(self):
#         pass
#     def test_pdf2img(self):
#         pass

if __name__ == '__main__':
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    doc = pymupdf.open("../data/义务教育教科书·语文一年级上册.pdf")
    for page in doc:
        pix = page.get_pixmap()
        filepath = os.path.join(output_dir, f"page-{page.number}.png")
        pix.save(filepath)
