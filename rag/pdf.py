
import  pymupdf 

class PDF:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = pymupdf.open(self.pdf_path)
    def get_text(self):
        doc = pymupdf.open(self.pdf_path)
        full_text = []
        for page in doc: 
            text = page.get_text()
            full_text.append(text)
        
        return full_text

    def get_img(self):
        doc = pymupdf.open(self.pdf_path)  
        for page in doc:  
            pix = page.get_pixmap()  
            pix.save("page-%i.png" % page.number)  
            exit()



if __name__ == '__main__':

    pdf = PDF("./data/english_wenzhang/作文模版3.pdf")
    pdf.get_text()
  