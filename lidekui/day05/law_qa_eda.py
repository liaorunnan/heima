import pandas as pd
import joblib
from tqdm import tqdm

pd.set_option('display.max_colwidth', None)  # 不限制列宽
pd.set_option('display.max_rows', None)      # 显示所有行（可选）
pd.set_option('display.max_columns', None)   # 显示所有列（可选）

qas = joblib.load("./data/qas.pkl")
qa_df = pd.DataFrame(qas)
qa_df['length'] = qa_df['query'].apply(lambda x: len(x))
# print(qa_df.describe())
print(qa_df[qa_df['length'] == 512])
# print(qa_df.iloc[174, :])
