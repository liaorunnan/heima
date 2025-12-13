from playwright.sync_api import sync_playwright
from time import sleep
import requests

def test_has_title():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto("http://localhost:8080/explore/recommend")
        page.wait_for_selector(".item-content")
        for i in range(8):
            page.mouse.wheel(0, delta_y=500)
            sleep(3)
        cards = page.locator('.item-content').all()
        print(len(cards))
        for card in cards:
            title = card.locator('.content-title').text_content()
            print(title)
            print('------')
            author = card.locator('.clickable-name').text_content()
            print(author)
            print('======')
        sleep(3)
        browser.close()

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
                with open(f'images/image{i+1}.png', 'wb') as f:
                    f.write(image.screenshot())
            print(src)
        sleep(3)
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
            with open(f'pdfs/pdf{i+1}.pdf', 'wb') as f:
                f.write(res.content)
        sleep(3)
        browser.close()

def test_get_zhi():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto("http://localhost:9080/")

        sleep(5)
        answer_container = page.locator('.answer-container')
        question_list = answer_container.locator('.d-block').all()
        answer_list = answer_container.locator('.text-body').all()
        for i in range(len(question_list)):
            question = question_list[i].text_content()
            answer = answer_list[i].text_content()
            print({'question': question, 'answer': answer})


if __name__ == '__main__':
    # test_get_pdf()
    test_get_zhi()
