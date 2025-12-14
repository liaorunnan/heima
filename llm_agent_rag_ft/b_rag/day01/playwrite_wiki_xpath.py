# playwright install 安装浏览器内核

#全部改成Xpath选择器
import time

if __name__ == '__main__':
    from playwright.sync_api import sync_playwright
    import requests
    from bs4 import BeautifulSoup
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto("http://localhost:8080/explore/recommend")
        page.wait_for_selector(".waterfall-item")
        for i in range(8):
            page.mouse.wheel(0, 400)
            time.sleep(2)
        cards = page.locator("//[@class='waterfall-item']").all()
        for card in cards:
            title=card.locator('//[@class="content-title"]').text_content()
            author = card.locator("//[@class='clickable-name']").text_content()

            print(title,author)
        browser.close()