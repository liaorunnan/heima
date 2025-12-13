from playwright.sync_api import sync_playwright
from time import sleep
import requests
from base64 import b64encode,b64decode
from sqlmodel import Field,Session,SQLModel,create_engine,select
from conf import settings
from sqlalchemy.dialects.mysql import LONGTEXT  # 导入 MySQL 的 LONGTEXT
from typing import Optional


engine = create_engine(settings.database_url)

class Png(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    png_name: str
    png_link: str
    png_b64: str = Field(sa_type=LONGTEXT)  # 使用 LONGTEXT

    @classmethod
    def query(cls, png_name):
        with Session(engine) as session:
            statement = select(cls).where(Png.png_name == png_name)
            png = session.exec(statement).first()
        return png

    def insert(self):
        with Session(engine) as session:
            session.add(self)
            session.commit()
        # return self

    @classmethod
    def query_all(cls):
        with Session(engine) as session:
            statement = select(cls)
            pngs = session.exec(statement).all()
        return pngs

class Pdf(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    pdf_name: str
    pdf_link: str
    pdf_b64: str = Field(sa_type=LONGTEXT)  # 使用 LONGTEXT

    @classmethod
    def query(cls, pdf_name):
        with Session(engine) as session:
            statement = select(cls).where(Png.pdf_name == pdf_name)
            pdf = session.exec(statement).first()
        return pdf

    def insert(self):
        with Session(engine) as session:
            session.add(self)
            session.commit()
        # return self

    @classmethod
    def query_all(cls):
        with Session(engine) as session:
            statement = select(cls)
            pdfs = session.exec(statement).all()
        return pdfs

def test_get_image():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto("http://localhost:8080/explore/recommend")
        page.wait_for_selector(".item-content")
        for i in range(8):
            page.mouse.wheel(0, delta_y=500)
            sleep(3)
        images = page.locator('.fade-in').all()
        for i, image in enumerate(images):
            src = image.get_attribute('data-src')
            if src:
                Png(
                    png_name=f'image{i+1}.png',
                    png_link=src,
                    png_b64=b64encode(image.screenshot()).decode('utf-8')
                ).insert()
            # break
        browser.close()

def test_get_pdf():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()
        page.goto("https://arxiv.org/list/cs.AI/recent")
        page.wait_for_selector('#articles')
        articles = page.locator('#articles')
        dt = articles.locator('dt').all()
        for i, pdf in enumerate(dt):
            p = pdf.get_by_title('Download PDF')
            href = p.get_attribute('href')
            href = 'https://arxiv.org' + href
            res = requests.get(href)
            # with open(f'pdfs/pdf{i+1}.pdf', 'wb') as f:
            #     f.write(res.content)
            Pdf(
                pdf_name=f'pdf{i+1}.pdf',
                pdf_link=href,
                pdf_b64=b64encode(res.content).decode('utf-8')
            ).insert()
        sleep(3)
        browser.close()


if __name__ == '__main__':
    SQLModel.metadata.create_all(engine)
    # test_get_image()
    test_get_pdf()
