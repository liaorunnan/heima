import requests
from bs4 import BeautifulSoup
from unittest import TestCase


class Test(TestCase):
    def test_arxiv(self):
        res = requests.get('https://arxiv.org/list/cs.AI/recent')
        # print(res.text)
        soup = BeautifulSoup(res.text, 'html.parser')
        articles = soup.select('#articles dd .list-title')  # css选择器
        links = soup.select('#articles dt')
        for i, article in enumerate(articles):
            title = article.text.replace('Title:', '', 1).strip()
            link = links[i].select('a')[1].get('href')
            link = 'https://arxiv.org' + link
            res1 = requests.get(link)
            soup1 = BeautifulSoup(res1.text, 'html.parser')
            abstract = soup1.select_one('.abstract').text.strip()
            abstract = abstract.replace('Abstract:', '', 1).strip()
            print(title)
            print(link)
            print(abstract)
            print('-' * 50)

    def test_xsl(self):
        res = requests.get('http://localhost:8080/explore/recommend')
        print(res)
