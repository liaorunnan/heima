import requests
from bs4 import BeautifulSoup
from unittest import TestCase

class Test(TestCase):
    def test_ar(self):
        response = requests.get("https://arxiv.org/list/cs.AI/recent")
        # print(response.text)
        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.select("#articles dd  .list-title")  # CSS选择器  .class # id
        links = soup.select("#articles dt")
        for i, article in enumerate(articles):
            title = article.text.replace("Title:", "", 1).strip()
            link = links[i].select("a")[1].get("href")
            link = "https://arxiv.org" + link
            res = requests.get(link)
            soup = BeautifulSoup(res.text, "html.parser")
            abstract = soup.select_one(".abstract").text.strip()
            abstract = abstract.replace("Abstract:", "", 1).strip()
            print(title)
            print(link)
            print(abstract)
            print("-" * 50)
            print(i,article)
    def test_xsl(self):
        response = requests.get("http://localhost:8080/explore/recommend")
        print(response.text)