from sqlmodel import Field, Session, SQLModel, create_engine, select
from conf import settings
import time
from playwright.sync_api import sync_playwright
import os
import random

engine = create_engine(settings.url)

class Duihua(SQLModel, table=True):

    id: int | None = Field(default=None, primary_key=True)
    duanluo: str
    def insert(self):
        with Session(engine) as session:
            session.add(self)
            session.commit()
        return self

def get_random_user_agent():
    """获取随机User-Agent"""
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15'
    ]
    return random.choice(user_agents)


def conmunication():
    os.makedirs("images", exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--no-sandbox',
                '--disable-setuid-sandbox'
            ]
        )

        context = browser.new_context(
            user_agent=get_random_user_agent(),
            viewport={'width': 1920, 'height': 1080},
            locale='zh-CN',
            timezone_id='Asia/Shanghai'
        )

        page = context.new_page()

        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        """)

        url = "https://zhuanlan.zhihu.com/p/361856912"

        try:
            page.goto(url, wait_until='networkidle', timeout=60000)
            time.sleep(3)

            # 检查是否被拦截
            if "异常" in page.content() or "验证" in page.title():
                print("⚠️ 需要手动处理验证码")
                input("处理完后按回车继续...")

            page.wait_for_selector("div.RichText", timeout=30000)

            sentences = page.locator("div.RichText p").all()
            if not sentences:
                sentences = page.locator("div[class*='RichText'] p").all()

            print(f"✅ 找到 {len(sentences)} 个段落")

            saved_count = 0
            for i, sentence_locator in enumerate(sentences):
                try:
                    text = sentence_locator.text_content().strip()
                    if not text:
                        continue

                    print(f"[{i+1}] {text[:60]}...")

                    # ✅ 关键：用 text（字符串）创建模型，并立即保存
                    duihua = Duihua(duanluo=text)
                    duihua.insert()  # 调用你的 insert 方法

                    saved_count += 1
                    time.sleep(random.uniform(0.5, 1.5))

                except Exception as e:
                    print(f"❌ 保存段落 {i+1} 失败: {e}")
                    continue

            print(f"✅ 成功保存 {saved_count} 条记录到数据库")

        except Exception as e:
            print(f"❌ 整体抓取失败: {e}")
            page.screenshot(path="error.png")
        finally:
            browser.close()


if __name__ == "__main__":
    SQLModel.metadata.create_all(engine)
    conmunication()