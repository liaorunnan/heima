# playwright install 安装浏览器内核
from sqlalchemy.dialects.mysql import insert
from sqlmodel import Field, Session, SQLModel, create_engine,select
from conf import settings
import time
from playwright.sync_api import sync_playwright
from playwright.sync_api import Page, expect


engine = create_engine(settings.url)
class images(SQLModel, table=True):

    id: int | None = Field(default=None, primary_key=True)
    image_url: str
    def insert(self):
        with Session(engine) as session:
            session.add(self)
            session.commit()
        return self

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto("http://192.168.111.26:8080/explore/recommend")
    page.wait_for_selector(".waterfall-item")
    SQLModel.metadata.create_all(engine)

    # for i in range(8):
    #     page.mouse.wheel(0, 400)
    #     time.sleep(2)
    cards = page.locator(".item-content").all()
    for i,card in enumerate(cards):
        title=card.locator(".content-title").text_content()
        author = card.locator(".clickable-name").text_content()
        # fade_in = card.locator(".content-img .fade-in")
        # image = fade_in.get_attribute("src")

        image = card.locator(".content-img .fade-in").get_attribute("src", timeout=2000)
        pic = card.locator(".content-img .fade-in")

        print(title,author,image)
        file_path = f"screenshots/{i+1}.png"
        pic.screenshot(path=file_path)
        images(image_url=image).insert()







