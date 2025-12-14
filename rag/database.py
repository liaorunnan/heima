# file: db/law_chunk.py
from typing import Optional, List
from sqlmodel import Field, SQLModel, Session, create_engine, select


from conf import settings

# ============ 数据库连接 ============
engine = create_engine(settings.get_url(), echo=False)  # echo=True 可看SQL


# ============ 模型定义（和你的表 100% 对齐）============
class LawChunk(SQLModel, table=True):
    __tablename__ = "law_chunks"

    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int
    law_filename: str = Field(max_length=255)
    law_title: str = Field(max_length=255)
    part_name: Optional[str] = Field(default=None, max_length=200)
    chapter_name: Optional[str] = Field(default=None, max_length=200)
    section_name: Optional[str] = Field(default=None, max_length=200)
    article_id: str = Field(max_length=200)
    content_text: str
    embedding_text: Optional[str] = None
    embedding_vector: Optional[str] = None  # JSON 字符串
    token_count: Optional[int] = Field(default=0)

    @classmethod
    def query_by_article(cls, article_id: str) -> Optional["LawChunk"]:
        """最常用：根据条号查一条"""
        with Session(engine) as session:
            statement = select(cls).where(cls.article_id == article_id)
            return session.exec(statement).first()

    @classmethod
    def query_by_filename(cls, filename: str) -> List["LawChunk"]:
        """查某部法律的所有条文"""
        with Session(engine) as session:
            statement = select(cls).where(cls.law_filename == filename)
            return session.exec(statement).all()

    @classmethod
    def query_by_chapter(cls, chapter: str) -> List["LawChunk"]:
        """查某个章节的所有条文"""
        with Session(engine) as session:
            statement = select(cls).where(cls.chapter_name == chapter)
            return session.exec(statement).all()

    @classmethod
    def search_text(cls, keyword: str) -> List["LawChunk"]:
        """全文模糊搜索（LIKE）"""
        with Session(engine) as session:
            statement = select(cls).where(cls.content_text.contains(keyword))
            return session.exec(statement).all()

    @classmethod
    def query_all(cls) -> List["LawChunk"]:
        """查全部（小心大数据量）"""
        with Session(engine) as session:
            statement = select(cls)
            return session.exec(statement).all()


# ============ 使用示例（直接运行这个文件就能测试）============
if __name__ == "__main__":
    # # 1. 查一条
    # item = LawChunk.query_by_article("第一条")
    # if item:
    #     print(f"找到：{item.law_filename} {item.article_id}")
    #     print(item.content_text[:100])
    #
    # # 2. 查一部法律的所有条文
    # all_items = LawChunk.query_by_filename("人民检察院公益诉讼办案规则_2023")
    # print(f"共找到 {len(all_items)} 条")
    #
    # # 3. 搜索关键词
    # results = LawChunk.search_text("公益诉讼")
    # print(f"搜索 '公益诉讼' 共找到 {len(results)} 条")
    ...