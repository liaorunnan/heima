import simple_pickle as sp

from indexing_fqa import VecIndex
from items import QaItem
from embedding import get_embedding
from tqdm import tqdm


qas = sp.read_pickle("./qas_word.pkl")
ids = 1
for qa in tqdm(qas):

    try:
        embeddings = get_embedding(qa["query"]).tolist()
        VecIndex("fqa").insert(embeddings,qa["query"], qa["answer"], str(ids), [])
        ids += 1
    except Exception as e:
        print(qa["query"])
        print(e)
        continue
        
