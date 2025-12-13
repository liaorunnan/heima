import pandas as pd
import pymysql
import matplotlib.pyplot as plt

from config import settings


def _connect():
    return pymysql.connect(
        host=settings.HOST,
        user=settings.USENAME,
        password=settings.PASSWORD,
        port=int(settings.PORT),
        database=settings.DATABASE,
        charset="utf8mb4",
    )


def fetch_data(kind):
    """获取数据；kind: 'law' or 'writ'"""
    sql_map = {
        "law": "SELECT id, content_text, embedding_text FROM law_chunks;",
        "writ": "SELECT id, context, indexbytitle FROM writ;",
    }
    query = sql_map.get(kind)
    if not query:
        raise ValueError("kind 必须是 'law' 或 'writ'")
    conn = _connect()
    try:
        return pd.read_sql(query, conn)
    finally:
        conn.close()


def _print_stats(name, series):
    print(f"=== {name} 长度统计 ===")
    print(f"最大长度: {series.max()}")
    print(f"最小长度: {series.min()}")
    print(f"平均长度: {series.mean():.2f}")
    print(f"方差: {series.var():.2f}")
    # print(f"5% 分位数: {series.quantile(0.05)}")
    # print(f"25% 分位数: {series.quantile(0.25)}")
    # print(f"50% 分位数 (中位数): {series.quantile(0.5)}")
    print(f"75% 分位数: {series.quantile(0.75)}")
    print(f"95% 分位数: {series.quantile(0.95)}")


def show(kind):
    df = fetch_data(kind)
    if df.empty:
        print(f"{kind} 表为空或未读取到数据")
        return

    if kind == "law":
        df["content_length"] = df["content_text"].astype(str).str.len()  # 计算 content_text 列的字符串长度并保存
        df["embedding_length"] = df["embedding_text"].astype(str).str.len()

        _print_stats("content_text", df["content_length"])  # 打印 content_text 长度的统计信息
        print()
        _print_stats("embedding_text", df["embedding_length"])

        # 绘制直方图
        # 防止中文乱码和负号
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.hist(df['content_length'], bins=50, color='skyblue', edgecolor='black')
        ax1.set_title('content_text 长度分布')
        ax1.set_xlabel('长度')
        ax1.set_ylabel('频次')

        ax2.hist(df['embedding_length'], bins=50, color='lightcoral', edgecolor='black')
        ax2.set_title('embedding_text 长度分布')
        ax2.set_xlabel('长度')
        ax2.set_ylabel('频次')
        plt.tight_layout()
        plt.show()

    else:  # writ
        df["context_length"] = df["context"].astype(str).str.len()
        df["indexbytitle_length"] = df["indexbytitle"].astype(str).str.len()
        _print_stats("context", df["context_length"])
        print()
        _print_stats("indexbytitle", df["indexbytitle_length"])


# 删除某些字段超过95%分位数的数据
def delete(kind: str):
    df = fetch_data(kind)
    if df.empty:
        print(f"{kind} 表为空或未读取到数据")
        return

    conn = _connect()
    try:
        with conn.cursor() as cursor:
            if kind == "law":
                df["content_length"] = df["content_text"].astype(str).str.len()
                df["embedding_length"] = df["embedding_text"].astype(str).str.len()
                c_thr = df["content_length"].quantile(0.95)
                e_thr = df["embedding_length"].quantile(0.95)
                outliers = df[(df["content_length"] > c_thr) | (df["embedding_length"] > e_thr)]
                if outliers.empty:
                    print("No outliers found to delete.")
                    return
                ids = tuple(outliers["id"].tolist())
                placeholders = ",".join(["%s"] * len(ids))
                sql = f"DELETE FROM law_chunks WHERE id IN ({placeholders})"

            # writ表
            else:
                df["context_length"] = df["context"].astype(str).str.len()
                c_thr = df["context_length"].quantile(0.95)
                outliers = df[df["context_length"] > c_thr]
                if outliers.empty:
                    print("No outliers found to delete.")
                    return
                ids = tuple(outliers["id"].tolist())
                placeholders = ",".join(["%s"] * len(ids))
                sql = f"DELETE FROM writ WHERE id IN ({placeholders})"

            cursor.execute(sql, ids)
            conn.commit()
            print(f"Deleted {len(ids)} rows from {kind}.")
    except Exception as e:
        print(f"Error occurred while deleting data: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    choice = input("请选择操作: show/delete: ").strip().lower()
    target = input("请选择表: law/writ: ").strip().lower()
    if choice == "show":
        show(target)
    elif choice == "delete":
        delete(target)
    else:
        print("未知操作，仅支持 show/delete")
