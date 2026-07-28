import pickle
import matplotlib
import matplotlib.pyplot as plt

with open("online_finetune_log_100000.pkl", "rb") as f:
    data = pickle.load(f)

print(data[0])

epx = []
epy = []
average_40 = []
record_success = []
episode_count = []
count = 0

for i, episode in enumerate(data):
    epx.append(i+1)
    epy.append(episode["episode_length"])
    if count == 0:
        print(episode["episode_length"])
    count += 1
    if episode["success"] == True:
        average_40.append(1)
    else:
        average_40.append(0)
    if len(average_40) == 40:
        mean = sum(average_40) / len(average_40)
        record_success.append(mean)
        average_40 = []
        episode_count.append((i+1) // 40)

plt.plot(episode_count, record_success, color='blue', marker='None', linestyle='-')
plt.title("Success count over 800 episodes during online finetuning")
plt.xlabel("40 episode interval count")
plt.ylabel("Average success")
plt.savefig("online_plots/success_fine_tuning.png", dpi=140)

plt.plot(epx, epy, color='blue', marker='None', linestyle='-')
plt.title("Episode length throughout online finetuning")
plt.xlabel("episode count")
plt.ylabel("Length in steps")
plt.savefig("online_plots/episode_length_fine_tuning.png", dpi=140)
