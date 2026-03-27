import requests
from bs4 import BeautifulSoup
from unittest import TestCase
import os
import time




def test_ar():
    # #创建文件夹
    # save_dir = "pdfs"
    # #
    # os.makedirs(save_dir, exist_ok=True)

    # 获取html
    response = requests.get("http://192.168.111.41:9080/" )
    # print(response.text)
    soup = BeautifulSoup(response.text, "html.parser")
    titles = soup.select(".rounded-0")
    # neiron = soup.select(".text-truncate-2 mb-2")

    for i, title in enumerate(titles):
        title = titles[i].select("a")[1].get_text()

        print(title)
        time.sleep(1)

    # for i, cont in enumerate(neiron):
    #     cont = neiron[i].select("a")[1].get("href")
    #     print(cont)
    #     time.sleep(1)

if __name__ == '__main__':
    test_ar()