import os
import logging

# 导入 win32com 用于处理 .doc 文件 (仅限 Windows 环境且需安装 Microsoft Word)
try:
    import win32com.client as win32
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False
    logging.warning("未安装 pywin32，将无法处理 .doc 文件。建议运行: pip install pywin32")


def convert_doc_to_docx(doc_path: str) -> str | None:
    """
    将 .doc 文件转换为 .docx 文件。
    返回转换后的 .docx 路径；失败时返回 None。
    需在 Windows 且已安装 Microsoft Word。
    """
    if not WIN32_AVAILABLE:
        logging.error(f"无法处理 {doc_path}: 缺少 pywin32 库或非 Windows 环境")
        return None

    # COM 接口常需绝对路径
    abs_doc_path = os.path.abspath(doc_path)
    abs_docx_path = abs_doc_path + "x"  # 简单追加 x 形成 .docx 后缀

    # 若已转换过直接复用
    if os.path.exists(abs_docx_path):
        return abs_docx_path

    word = None
    try:
        # 后台启动 Word，通过 COM 执行格式转换
        word = win32.Dispatch("Word.Application")
        word.Visible = False

        doc = word.Documents.Open(abs_doc_path)
        doc.SaveAs2(abs_docx_path, FileFormat=16)  # 16 = wdFormatXMLDocument
        doc.Close()

        logging.info(f"格式转换成功: {doc_path} -> {abs_docx_path}")
        return abs_docx_path
    except Exception as e:
        # 转换失败记录后返回 None
        logging.error(f"转换 .doc 文件失败 {doc_path}: {e}")
        return None
    finally:
        # 简化实现，单次调用后关闭 Word，批量场景可优化为复用
        if word:
            try:
                word.Quit()
            except Exception:
                pass
