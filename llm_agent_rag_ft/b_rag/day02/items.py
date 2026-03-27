from pydantic import BaseModel


class BookItem(BaseModel):
    word: str
    pronunciation: str
    mean: str


    def __hash__(self):
        return hash(self.word)

    def __eq__(self, other):
        if not isinstance(other, BookItem):
            return False
        return self.word == other.word


class FAQItem(BaseModel):
    id: str
    query: str
    answer: str

