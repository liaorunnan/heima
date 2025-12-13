import fitz
import os
from tqdm import tqdm


def pdf2img(pdf_path, img_path):
    doc = fitz.open(pdf_path)
    for pg in tqdm(range(doc.page_count)):
        page = doc[pg]
        rotate = int(0)
        zoom_x, zoom_y = 4, 4
        mat = fitz.Matrix(zoom_x, zoom_y).prerotate(rotate)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        pix._writeIMG(format_="png", filename=os.path.join(img_path, f"{pg}.png"), jpg_quality=100)


if __name__ == '__main__':
    pdf2img("./data/义务教育教科书·语文一年级上册.pdf", "./images")
