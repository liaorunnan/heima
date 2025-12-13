"""法律案例 Markdown 清洗、分段切分并落库的工具。

核心能力：
1. 清洗 .md 文本（去除空白、不可见字符）。
2. 按一级目录/二级目录/案件标题/# 标题/## 板块切分。
3. 采用滑动窗口（overlap=64，目标 300-400，最大 500）生成 Chunk，序号连续。
4. 构造 enriched_content（带元数据抬头），便于后续向量化。
5. 提供入库函数，演示 main 读取示例文件并打印生成的 SQLModel 对象。

处理对象：data_process/output/full_md/1.婚姻家庭与继承纠纷.md

注意：
- 代码仅做示范，真实落库前请根据实际 MySQL 配置填好 config.settings。
- 保证同一案件内 sequence_index 连续，便于窗口检索 current±1。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from sqlmodel import Field, Session, SQLModel, create_engine
from sqlalchemy import Column, Text

try:
    from config import settings
except Exception:  # pragma: no cover - 配置缺失时允许仍可跑 main 演示
    settings = None


# =====================
# SQLModel 定义
# =====================


class LegalDocument(SQLModel, table=True):
    """案例分段表，支持窗口检索。"""

    __tablename__ = "legal_documents"

    id: Optional[int] = Field(default=None, primary_key=True)
    first_level: str = Field(nullable=False, description="一级目录，如‘婚姻家庭纠纷’")
    secondarylevel_: str = Field(nullable=False, description="二级目录，如‘婚约财产纠纷’")
    title: str = Field(nullable=False, description="案件大标题")
    case_title: str = Field(nullable=False, description="案件小标题，含‘案’结尾")
    section_name: str = Field(nullable=False, description="板块名，例如【基本案情】")
    content: str = Field(sa_column=Column(Text), description="原始切分文本，不带元数据")
    enriched_content: str = Field(sa_column=Column(Text), description="带元数据的文本，供向量化")
    sequence_index: int = Field(nullable=False, description="案件内序号，从 1 开始连续")


# =====================
# 数据结构
# =====================


@dataclass
class Section:
    name: str
    text_lines: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.text_lines).strip()


@dataclass
class Case:
    first_level: str
    second_level: str
    title: str
    case_title: str
    sections: List[Section] = field(default_factory=list)


# =====================
# 文本清洗与解析
# =====================


FIRST_LEVEL_RE = re.compile(r"^[一二三四五六七八九十百]+、\s*(.+)$")
SECOND_LEVEL_RE = re.compile(r"^[（(][一二三四五六七八九十百]+[)）]\s*(.+)$")
CASE_HEADER_RE = re.compile(r"^#\s+(?!#)(.+)$")
SECTION_HEADER_RE = re.compile(r"^##\s+【(.+?)】")
MD_CONTROL_CHARS = re.compile(r"[\u200b\ufeff]")


def clean_line(line: str) -> str:
    """去除不可见字符与多余空白，保留中文与数字。"""

    line = MD_CONTROL_CHARS.sub("", line)
    return line.strip()


def normalize_text(lines: Iterable[str]) -> List[str]:
    """全文件级别清洗，去空行。"""

    normalized: List[str] = []
    for raw in lines:
        cleaned = clean_line(raw)
        if cleaned:
            normalized.append(cleaned)
    return normalized


def parse_cases(lines: List[str]) -> List[Case]:
    """依据 # / ## 标题切出案例与板块，携带目录上下文。"""

    cases: List[Case] = []
    current_case: Optional[Case] = None
    current_section: Optional[Section] = None
    pending_main_title: Optional[str] = None
    first_level = "未知一级目录"
    second_level = "未知二级目录"
    last_first_level: Optional[str] = None  # 若当前未匹配到目录，沿用上一案例
    last_second_level: Optional[str] = None

    def flush_section():
        nonlocal current_section, current_case
        if current_case and current_section and current_section.text:
            current_case.sections.append(current_section)
        current_section = None

    def flush_case():
        nonlocal current_case, current_section
        flush_section()
        if current_case:
            cases.append(current_case)
        current_case = None
        current_section = None

    for line in lines:
        # 目录定位：一级 / 二级
        m1 = FIRST_LEVEL_RE.match(line)
        if m1:
            first_level = m1.group(1).strip()
            last_first_level = first_level
            second_level = "未知二级目录"
            continue

        m2 = SECOND_LEVEL_RE.match(line)
        if m2:
            second_level = m2.group(1).strip()
            last_second_level = second_level
            continue

        # 案例标题（# 开头，非 ##）
        m_case = CASE_HEADER_RE.match(line)
        if m_case:
            text = m_case.group(1).strip()

            has_dash = "——" in text or "—" in text
            if has_dash:
                parts = re.split(r"—+", text, maxsplit=1)
                left = parts[0].strip()
                right = parts[1].strip() if len(parts) > 1 else ""
                title = pending_main_title or left or right
                case_title = right or (left if left.endswith("案") else f"{left}案")
                pending_main_title = None
                flush_case()
                eff_first = first_level if first_level != "未知一级目录" else (last_first_level or first_level)
                eff_second = second_level if second_level != "未知二级目录" else (last_second_level or second_level)
                current_case = Case(
                    first_level=eff_first,
                    second_level=eff_second,
                    title=title,
                    case_title=case_title,
                )
                last_first_level = eff_first
                last_second_level = eff_second
                current_section = None
                continue

            # 无破折号则暂存大标题，等待下一行的 case_title
            pending_main_title = text
            continue

        # 板块标题（##）
        m_sec = SECTION_HEADER_RE.match(line)
        if m_sec:
            flush_section()
            section_name = f"【{m_sec.group(1).strip()}】"
            current_section = Section(name=section_name, text_lines=[])
            continue

        # 普通正文行
        if current_case:
            if current_section is None:
                # 如果缺失板块，则归入“正文”兜底
                current_section = Section(name="【正文】", text_lines=[])
            current_section.text_lines.append(line)

    flush_case()
    return cases


# =====================
# 切分与增强
# =====================


MAX_LEN = 500
TARGET_LEN = 380
OVERLAP = 64


def chunk_text(text: str) -> List[str]:
    """按字符窗口切分，优先在句号/问号/感叹号/分号后截断。"""

    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(normalized):
        window_end = min(start + TARGET_LEN, len(normalized))
        window_slice = normalized[start:window_end]
        # 在窗口内寻找最近的句末符，避免截断句子
        cut = -1
        for sym in "。！？；;":
            pos = window_slice.rfind(sym)
            if pos > cut:
                cut = pos
        if cut != -1 and (start + cut + 1 - start) >= 150:
            end = start + cut + 1
        else:
            end = window_end
        end = min(end, start + MAX_LEN)
        chunk = normalized[start:end].strip()
        if not chunk:
            break
        chunks.append(chunk)
        # 下一段保持 overlap，确保指针单调递增并在末尾停止
        if end >= len(normalized):
            break
        next_start = max(end - OVERLAP, start + 1)
        if next_start <= start:
            next_start = start + 1
        start = next_start
    return chunks


def build_enriched(case: Case, section_name: str, chunk_text_value: str) -> str:
    header = [
        "[元数据]",
        f"分类: {case.first_level} - {case.second_level} - {case.title} - {case.case_title}",
        f"案由/板块: {section_name}",
        "----------------",
        chunk_text_value,
    ]
    enriched = "\n".join(header)
    return enriched[:MAX_LEN] if len(enriched) > MAX_LEN else enriched


def case_to_documents(case: Case) -> List[LegalDocument]:
    """把一个案例转为多个 LegalDocument，带序号。"""

    docs: List[LegalDocument] = []
    seq = 1
    for section in case.sections:
        # 兜底：若文本很短直接存一段
        chunks = chunk_text(section.text) if len(section.text) > MAX_LEN else [section.text]
        for chunk in chunks:
            chunk_trimmed = chunk[:MAX_LEN]
            enriched = build_enriched(case, section.name, chunk_trimmed)
            docs.append(
                LegalDocument(
                    first_level=case.first_level,
                    secondarylevel_=case.second_level,
                    title=case.title,
                    case_title=case.case_title,
                    section_name=section.name,
                    content=chunk_trimmed,
                    enriched_content=enriched,
                    sequence_index=seq,
                )
            )
            seq += 1
    return docs


# =====================
# 入库工具
# =====================


def build_engine_from_settings():
    if settings is None:
        raise RuntimeError("未找到 config.settings，无法创建数据库连接")
    db_url = getattr(settings, "url", None)
    if not db_url:
        db_url = (
            f"mysql+pymysql://{settings.USENAME}:{settings.PASSWORD}"
            f"@{settings.HOST}:{settings.PORT}/{settings.DATABASE}?charset=utf8mb4"
        )
    return create_engine(db_url, echo=False, pool_recycle=3600)


def insert_documents(engine, docs: List[LegalDocument]):
    """批量插入，提前建表。"""

    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all(docs)
        session.commit()


# =====================
# 主流程
# =====================


def process_markdown_file(md_path: str) -> List[LegalDocument]:
    with open(md_path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    lines = normalize_text(raw_lines)
    cases = parse_cases(lines)

    documents: List[LegalDocument] = []
    for case in cases:
        documents.extend(case_to_documents(case))
    return documents


def main():
    """读取示例文件，打印前若干条对象。"""

    md_path = os.path.join(
        os.path.dirname(__file__), "output", "full_md", "1.婚姻家庭与继承纠纷.md"
    )

    if not os.path.exists(md_path):
        print(f"未找到示例文件: {md_path}")
        return

    docs = process_markdown_file(md_path)
    print(f"共生成 {len(docs)} 条分段")
    for i, doc in enumerate(docs[:5], start=1):
        print(f"---- 示例 {i} ----")
        print(f"目录: {doc.first_level} / {doc.secondarylevel_}")
        print(f"案件: {doc.title} | {doc.case_title}")
        print(f"板块: {doc.section_name} 序号: {doc.sequence_index}")
        print(f"content[:120]: {doc.content[:120]}...")
        print(f"enriched_content[:120]: {doc.enriched_content[:120]}...")

    # 如需真实入库，取消下方注释
    engine = build_engine_from_settings()
    insert_documents(engine, docs)


if __name__ == "__main__":
    main()
