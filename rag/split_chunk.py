import glob
import os
import re
import docx
import pymysql
import json
import logging
import sys

# 尝试导入 win32com 用于处理 .doc 文件 (仅限 Windows)
try:
    import win32com.client as win32

    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False
    logging.warning("未安装 pywin32，将无法处理 .doc 文件。建议运行: pip install pywin32")

# 配置简单的日志打印，方便调试
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ==============================================================================
# 模块 0: 格式转换辅助工具 (.doc -> .docx)
# ==============================================================================

def convert_doc_to_docx(doc_path):
    """
    将 .doc 文件转换为 .docx 文件
    返回: 转换后的临时 .docx 文件路径
    注意: 需要 Windows 系统且安装了 Microsoft Word
    """
    if not WIN32_AVAILABLE:
        logging.error(f"无法处理 {doc_path}: 缺少 pywin32 库或非 Windows 环境")
        return None

    # 获取绝对路径（Word COM 接口通常需要绝对路径）
    abs_doc_path = os.path.abspath(doc_path)
    abs_docx_path = abs_doc_path + "x"  # simple append 'x' -> .docx

    # 如果转换后的文件已存在，直接返回
    if os.path.exists(abs_docx_path):
        return abs_docx_path

    word = None
    try:
        # 启动 Word 应用程序 (后台运行)
        word = win32.Dispatch("Word.Application")
        word.Visible = False

        # 打开 .doc 文件
        doc = word.Documents.Open(abs_doc_path)

        # 另存为 .docx (FileFormat=16 代表 wdFormatXMLDocument)
        doc.SaveAs2(abs_docx_path, FileFormat=16)
        doc.Close()

        logging.info(f"格式转换成功: {doc_path} -> {abs_docx_path}")
        return abs_docx_path

    except Exception as e:
        logging.error(f"转换 .doc 文件失败 {doc_path}: {e}")
        return None
    finally:
        # 尽量不要在这里 Quit Word，因为如果是批量处理，反复开关 Word 会很慢
        # 但为了代码简单独立，这里先保持 Quit。如果文件多，建议在 main 中初始化 Word。
        if word:
            try:
                word.Quit()
            except:
                pass


# ==============================================================================
# 模块 1: 法律文档切分逻辑 (核心算法)
# ==============================================================================

# --- 正则表达式定义 ---
REGEX_PART = re.compile(r"^第[零一二三四五六七八九十百]+编")
REGEX_CHAPTER = re.compile(r"^第[零一二三四五六七八九十百]+章")
REGEX_SECTION = re.compile(r"^第[零一二三四五六七八九十百]+节")
REGEX_ARTICLE = re.compile(r"^第[零一二三四五六七八九十百\d]+条")


def process_law_file(file_path):
    """
    读取 Word 文件 (.doc 或 .docx)，根据法规层级结构切分为 Chunk。
    返回一个包含字典的列表。
    """

    actual_file_to_read = file_path
    is_temp_file = False

    # --- 1. 格式检查与转换 ---
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == '.doc':
        logging.info(f"检测到旧版格式 (.doc)，尝试转换...")
        converted_path = convert_doc_to_docx(file_path)
        if converted_path and os.path.exists(converted_path):
            actual_file_to_read = converted_path
            is_temp_file = True  # 标记为临时文件，处理完后可选择删除
        else:
            logging.error(f"跳过文件 {file_path}: 格式转换失败")
            return []

    # --- 2. 使用 python-docx 读取文件对象 ---
    try:
        doc = docx.Document(actual_file_to_read)
    except Exception as e:
        logging.error(f"无法读取文件 {actual_file_to_read}: {e}")
        return []

    # --- 3. 从原始文件名提取基础信息 ---
    # 注意：我们要用原始 file_path 的文件名，而不是临时文件的
    filename = os.path.basename(file_path)

    # 使用 os.path.splitext 安全去除后缀 (.doc 或 .docx)
    # "民法典_2020.docx" -> ("民法典_2020", ".docx")
    filename_no_ext = os.path.splitext(filename)[0]
    law_title = filename_no_ext.split("_")[0]

    chunks = []

    # --- 状态机 (State Machine) 初始化 ---
    state = {
        "part": "",
        "chapter": "",
        "section": ""
    }

    current_article = {
        "id": "",
        "content": ""
    }

    # --- 4. 逐行遍历 Word 文档的段落 ---
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        clean_text = re.sub(r'\s+', '', text)

        # A. 层级识别
        if REGEX_PART.match(clean_text):
            state["part"] = text
            state["chapter"] = ""
            state["section"] = ""
            continue

        if REGEX_CHAPTER.match(clean_text):
            state["chapter"] = text
            state["section"] = ""
            continue

        if REGEX_SECTION.match(clean_text):
            state["section"] = text
            continue

        # B. 法条切分
        if REGEX_ARTICLE.match(clean_text):
            if current_article["id"]:
                save_chunk_to_list(chunks, law_title, state, current_article)

            parts = text.split(maxsplit=1)
            current_article["id"] = parts[0]
            current_article["content"] = parts[1] if len(parts) > 1 else ""

        # C. 正文累积
        else:
            if current_article["id"]:
                current_article["content"] += "\n" + text

    # 循环结束收尾
    if current_article["id"]:
        save_chunk_to_list(chunks, law_title, state, current_article)

    # --- 5. 清理临时文件 (可选) ---
    # 如果是你生成的 .docx 临时文件，建议处理完后删除，保持文件夹整洁
    if is_temp_file and os.path.exists(actual_file_to_read):
        try:
            os.remove(actual_file_to_read)
            logging.info(f"清理临时文件: {actual_file_to_read}")
        except OSError:
            pass

    return chunks


def save_chunk_to_list(chunks_list, law_title, state, article_data):
    """
    辅助函数：构造标准化的 Chunk 数据结构
    """
    path_components = [
        law_title,
        state["part"],
        state["chapter"],
        state["section"],
        article_data["id"]
    ]
    valid_paths = [p for p in path_components if p]
    path_str = " ".join(valid_paths)
    embedding_text = f"{path_str} ： {article_data['content']}"

    chunks_list.append({
        "part": state["part"],
        "chapter": state["chapter"],
        "section": state["section"],
        "article_id": article_data["id"],
        "content_text": article_data["content"],
        "embedding_text": embedding_text
    })


# ==============================================================================
# 模块 2: MySQL 数据库管理类
# ==============================================================================

class LawMySQLManager:
    def __init__(self, db_config):
        self.db_config = db_config
        self.conn = None
        self.cursor = None

    def connect(self):
        try:
            self.conn = pymysql.connect(
                **self.db_config,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            self.cursor = self.conn.cursor()
            logging.info("MySQL 数据库连接成功")
        except Exception as e:
            logging.error(f"数据库连接失败: {e}")
            raise

    def close(self):
        if self.cursor: self.cursor.close()
        if self.conn: self.conn.close()

    def insert_document(self, filename, law_title, file_path):
        sql = """
              INSERT INTO law_documents (filename, law_title, file_path, upload_date)
              VALUES (%s, %s, %s, NOW())
              """
        self.cursor.execute(sql, (filename, law_title, file_path))
        return self.cursor.lastrowid

    def insert_chunks_batch(self, chunks_data):
        sql = """
              INSERT INTO law_chunks (document_id, law_filename, law_title, \
                                      part_name, chapter_name, section_name, article_id, \
                                      content_text, embedding_text, embedding_vector) \
              VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
              """
        values = []
        for c in chunks_data:
            raw_vector = c.get('embedding', [])
            vector_json_str = json.dumps(raw_vector)
            values.append((
                c['document_id'],
                c['law_filename'],
                c['law_title'],
                c['part'],
                c['chapter'],
                c['section'],
                c['article_id'],
                c['content_text'],
                c['embedding_text'],
                vector_json_str
            ))
        try:
            self.cursor.executemany(sql, values)
            self.conn.commit()
            logging.info(f"成功批量插入 {len(values)} 条数据")
        except Exception as e:
            self.conn.rollback()
            logging.error(f"批量插入失败: {e}")
            raise


# ==============================================================================
# 模块 3: 主执行流程
# ==============================================================================

def main():
    # 1. 数据库连接配置
    DB_CONFIG = {
        "host": "8.134.221.210",
        "user": "root",
        "password": "f4765fefa271d718",
        "database": "ailaw",
        "port": 3306
    }

    # 2. 扫描文件：修改 glob 模式以匹配 .doc 和 .docx
    # 注意：glob 模式 "*.doc*" 可以匹配 .doc 和 .docx
    files = glob.glob(os.path.join("Laws", "*.doc*"))

    db_manager = LawMySQLManager(DB_CONFIG)

    try:
        db_manager.connect()

        for file_path in files:
            # 过滤掉以 ~$ 开头的临时文件 (Word 打开时产生的锁文件)
            if os.path.basename(file_path).startswith("~$"):
                continue

            if not os.path.exists(file_path):
                continue

            logging.info(f"=== 开始处理: {file_path} ===")

            # 使用 splitext 兼容两种后缀
            filename = os.path.basename(file_path)
            law_title = os.path.splitext(filename)[0].split("_")[0]

            # --- 步骤 A: 解析文档 ---
            chunks = process_law_file(file_path)
            if not chunks:
                logging.warning("未提取到有效内容或格式转换失败，跳过")
                continue

            # --- 步骤 B: 写入主表 ---
            doc_id = db_manager.insert_document(filename, law_title, file_path)
            logging.info(f"文档记录创建成功，ID: {doc_id}")

            # --- 步骤 C: 数据补充 ---
            for chunk in chunks:
                chunk['document_id'] = doc_id
                chunk['law_filename'] = filename
                chunk['law_title'] = law_title
                # 模拟向量
                chunk['embedding'] = [0.123, 0.456, 0.789]

            # --- 步骤 D: 批量入库 ---
            db_manager.insert_chunks_batch(chunks)

    except Exception as e:
        logging.error(f"程序运行发生致命错误: {e}")
    finally:
        db_manager.close()


if __name__ == "__main__":
    main()