import simple_pickle as sp
from tqdm import tqdm

qas = sp.read_pickle("../../data/synonyms.pkl")
datas = [{"query":qa["query"], "pos": [p[0] for p in qa["pos"]],"neg":[]} for qa in qas]
with open("data.jsonl","w",encoding="utf-8") as f:
    for data in tqdm(datas):
        f.write(str(data).replace("'",'"')+"\n")
