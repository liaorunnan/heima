from uuid import uuid4


# file: items/laws_item.py
from typing import Optional, List
from pydantic import BaseModel


class LawsItem(BaseModel):
    """专门用来接收 ES 查询结果并返回给前端的实体类"""
    id: Optional[int] = None                    # ES 的 _id（数据库主键）
    document_id: int
    law_filename: str                           # 如：人民检察院公益诉讼办案规则_2023
    law_title: str
    part_name: Optional[str] = None             # 编
    chapter_name: Optional[str] = None          # 章
    section_name: Optional[str] = None          # 节
    article_id: str                             # 第X条（最核心！）
    content_text: Optional[str] = None                            # 正文内容
    embedding_text: Optional[str] = None
    token_count: Optional[int] = 0

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, LawsItem):
            return self.id == other.id
        return False
    # 可选：方便调试和展示
    class Config:
        from_attributes = True  # 兼容 pydantic v2（以前叫 orm_mode=True）
        # 如果你用的是 pydantic v1，改成：orm_mode = True

    def __str__(self):
        return f"[{self.law_filename}] {self.article_id}"

    def __repr__(self):
        return self.__str__()

class FAQItem(BaseModel):
    id: str
    query: str
    answer: str
    query_embedding_text: str
    score:Optional[float] = None  # 关键：加 Optional，并默认 None
