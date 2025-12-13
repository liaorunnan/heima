from pydantic import BaseModel
from typing import Optional


class BookItem(BaseModel):
    id: Optional[str] = None
    child: Optional[str] = None
    parent: Optional[str] = None
    source: Optional[str] = None
    score: Optional[float] = None


class LawItem(BaseModel):
    id: Optional[str] = None
    law_title: str
    embedding_text: str
    score: Optional[float] = None


class FAQItem(BaseModel):
    id: Optional[str] = None
    question: str
    answer: str
    score: Optional[float] = None
