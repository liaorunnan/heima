
from pydantic import BaseModel
from typing import List


class YinyutlItem(BaseModel):
    id: str
    child: str
    parent: str
    source: List[str]
    score: float = 0.0

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, YinyutlItem):
            return self.id == other.id
        
        return False


class QaItem(BaseModel):
    qid: str
    query: str
    answer: str
    source: List[str]
    score: float

    # def __hash__(self):
    #     return hash(self.qid)

    # def __eq__(self, other):
    #     if isinstance(other, QaItem):
    #         return self.qid == other.qid
        
    #     return False
