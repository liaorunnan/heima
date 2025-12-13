import simple_pickle as sp
import joblib

# qas = sp.read_data("./data/qas.pkl")
qas = joblib.load("./data/qas.pkl")
print(len(qas))
print(qas[0])
