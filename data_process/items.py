from pydantic import BaseModel

# 法条的召回数据类
class LawItem(BaseModel):
    id : str
    law_filename : str
    embedding_text : str
    score : float = 0.0
    def __hash__(self):
        return hash(self.id)
    def __eq__(self, other):
        if isinstance(other,LawItem):
            return self.id == other.id
        return False

# 文书的召回数据类
class WritItem(BaseModel):
    id: str
    indexbytitle: str
    context: str
    score: float = 0.0
    def __hash__(self):
        return hash(self.id)
    def __eq__(self, other):
        if isinstance(other,WritItem):
            return self.id == other.id
        return False
    
# 案例的召回数据类
class CaseItem(BaseModel):
    id: str
    title: str
    case_title: str
    section_name: str
    content: str
    enriched_content: str
    sequence_index: int
    score: float = 0.0

    def __hash__(self):
        return hash(self.id)
    def __eq__(self, other):
        if isinstance(other,CaseItem):
            return self.id == other.id
        return False
    
# FAQ的召回数据类
class FAQItem(BaseModel):
    id: str
    query: str
    answer: str
    score: float = 0.0  