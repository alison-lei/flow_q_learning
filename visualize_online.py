import pickle

with open("online_finetune_log.pkl", "rb") as f:
    data = pickle.load(f)

print(data[0])
print(type(data))
print(len(data[0]["q_values"]))