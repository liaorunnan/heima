import simple_pickle as sp
from tqdm import tqdm

qas = sp.read_pickle("../../../xiaokui/data/qas.pkl")


def gen_by_simbert():
    from nlpcda import Simbert

    config = {
        'model_path': '/root/autodl-tmp/chinese_simbert_L-12_H-768_A-12',
        'CUDA_VISIBLE_DEVICES': '0,1',
        'max_len': 32,
        'seed': 1
    }
    simbert = Simbert(config=config)
    synonyms = []
    for qa in tqdm(qas):
        synonym = simbert.replace(sent=qa["query"], create_num=3)
        synonyms.append({"query": qa["query"], "pos": synonym})
    sp.write_pickle(synonyms, "../../../xiaokui/data/synonyms.pkl")

# def gen_by_translator():
#     from nlpcda import baidu_translate
#
#     synonyms = []
#     for qa in qas:
#         en_s = baidu_translate(content=qa["query"], appid=settings.baidu_translate_app_id, secretKey=settings.baidu_translate_secret_key, t_from='zh', t_to='en')
#         zh_s = baidu_translate(content=en_s, appid=settings.baidu_translate_app_id, secretKey=settings.baidu_translate_secret_key, t_from='en', t_to='zh')
#         synonyms.append({"query": qa["query"], "pos": zh_s})

def read_synonyms():
    path = "../../data/synonyms.pkl"
    # path = "../../../xiaokui/data/qas.pkl"
    synonyms = sp.read_pickle(path)
    # print(synonyms)
    qas = [{"query":qa["query"], "pos": [p[0] for p in qa["pos"]],"neg":[]} for qa in synonyms]
    print(qas)


if __name__ == '__main__':
    # gen_by_translator()

    # gen_by_simbert()

    read_synonyms()