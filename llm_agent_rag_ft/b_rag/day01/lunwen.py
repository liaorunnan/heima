import requests
from bs4 import BeautifulSoup
from unittest import TestCase
import os
import time





class app_main():

    def prepare(self,pdf_url):
        save_dir = "pdfs"
        os.makedirs(save_dir, exist_ok=True)  # 自动创建文件夹
        file_path = os.path.join(save_dir, pdf_url.split("/")[-1]+".pdf")

        # 2. 下载
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(pdf_url, headers=headers, timeout=15)
        response.raise_for_status()  # 如果状态码不是 200，抛出异常

        # 3. 保存
        with open(file_path, "wb") as f:  # 注意是 "wb"（二进制写入）
            f.write(response.content)


    def test_ar(self):
        # #创建文件夹
        # save_dir = "pdfs"
        # #
        # os.makedirs(save_dir, exist_ok=True)
        #任务头
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        print("正在获取列表...")
        #获取html
        response = requests.get("https://arxiv.org/list/cs.AI/recent", headers=headers)
        # print(response.text)
        soup = BeautifulSoup(response.text, "html.parser")
        dts = soup.select("#articles dt")  # CSS选择器  .class # id

        for i,dt in enumerate(dts):
            pdf_link = dts[i].select("a")[3].get("id")
            pdf_link = "https://arxiv.org" + pdf_link
            pdf_link = pdf_link.replace("-", "/", 1)
            pdf_link = pdf_link.replace("html", "/pdf", 1)
            print(pdf_link)
            time.sleep(1)
            self.prepare(pdf_link)


if __name__ == '__main__':
    app_main().test_ar()