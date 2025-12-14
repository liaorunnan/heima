from pydantic import BaseModel


class BookItem(BaseModel):
    id: str
    word: str
    pronunciation: str
    mean:str



class FAQItem(BaseModel):
    id: str
    query: str
    answer: str
    source: str
