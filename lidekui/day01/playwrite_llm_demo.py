import openai
from conf import settings
from playwright.sync_api import sync_playwright
from time import sleep
import html2text


client = openai.Client(api_key=settings.api_key, base_url=settings.base_url)
html_con = html2text.HTML2Text()
html_con.ignore_links = True
html_con.ignore_images = True


def chat(prompt):
    response = client.chat.completions.create(
        model=settings.model_name,
        messages=[
            {"role": "system", "content": "你是一个信息抽取专家，请从提供的markdown中抽取出标题和作者，并返回一个json格式的字符串，格式为：[{'title': '', 'author': ''}]"},
            {"role": "user", "content": prompt},
        ],
        temperature=0
    )
    return response.choices[0].message.content

def crawler():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto("http://localhost:8080/explore/recommend")
        page.wait_for_selector(".item-content")
        html = page.content()
        browser.close()
        return html


if __name__ == '__main__':
    html = crawler()
    md = html_con.handle(html)
    res = chat(md)
    print(res)
