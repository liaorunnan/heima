import pymupdf
from unittest import TestCase


doc = pymupdf.open('../day01/pdfs/pdf1.pdf')
for page in doc:  # iterate through the pages
    pix = page.get_pixmap()  # render page to an image
    pix.save("page-%i.png" % page.number)  # store image as a PNG
    break

# class TestPymupdf(TestCase):
#     def test_pymupdf(self):
#         # doc = pymupdf.open('../day01/pdfs/pdf1.pdf')
#         for page in doc:
#             print(page.get_text())
#
#     def test_pdf2img(self):
#         # doc = pymupdf.open('../day01/pdfs/pdf1.pdf')  # open document
#         for page in doc:  # iterate through the pages
#             pix = page.get_pixmap()  # render page to an image
#             pix.save("page-%i.png" % page.number)  # store image as a PNG
#             break
