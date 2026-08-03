import pickle

with open("robomimic_datasets/lift_300_demos_scripted_flow_2026-06-10_14-29-49-254753.pkl", "rb") as f:
    data = pickle.load(f)

print(data[0])
print(type(data))
print(len(data))

success_count = 0

for i, step in enumerate(data):
    if step["dones"]:
        success_count += 1
    if success_count == 20:
        step_end = i+1
        break

data = data[:step_end]

with open("robomimic_datasets/lift_20_demos_scripted_flow.pkl", "wb") as f:
    pickle.dump(data, f)

with open("robomimic_datasets/lift_20_demos_scripted_flow.pkl", "rb") as f:
    data = pickle.load(f)

print(data[0])
print(type(data))
print(len(data))
print(data[445])