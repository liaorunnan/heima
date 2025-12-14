from unittest import TestCase
import pymupdf



# class TestPymupdf(TestCase):
#     def test_pdf2text(self):
#         doc = pymupdf.open("../data/0001_2512.05013.pdf")
#         for page in doc:
#             text = page.get_text()
#             print(text)

    # def test_pdf2img(self):

# import pymupdf4llm
# md_text = pymupdf4llm.to_markdown("../data/0001_2512.05013.pdf")
# print(md_text)

def pdf2text(pdf_path,save_path):
    doc = pymupdf.open(pdf_path)
    for page in doc:
        zoom_x = 3.0  # horizontal zoom
        zoom_y = 3.0  # vertical zoom
        mat = pymupdf.Matrix(zoom_x, zoom_y)  # zoom factor 2 in each dimension
        pix = page.get_pixmap(matrix=mat)
        pix.save(save_path+"page-%i.png" % page.number)

if __name__ == '__main__':
    # doc = pymupdf.open("../data/0001_2512.05013.pdf")  # open document
    # for page in doc:  # iterate through the pages
    #     pix = page.get_pixmap()  # render page to an image
    #     pix.save("page-%i.png" % page.number)  # store image as a PNG
    #     break

    ...