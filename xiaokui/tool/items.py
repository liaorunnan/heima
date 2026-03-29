from pydantic import BaseModel


class BookItem(BaseModel):
    id: str
    parent: str
    source: str = None
    score: float = 0.0

class EnglishItem(BaseModel):
    id: str
    meaning: str
    pronunciation: str
    score: float = 0.0


class FAQItem(BaseModel):
    id: str
    query: str
    answer: str
    # source: str
    score: float = 0.0
