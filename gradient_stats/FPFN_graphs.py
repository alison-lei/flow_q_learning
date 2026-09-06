import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import csv

win_sz = [10, 20, 30]
# win_sz = [5, 10, 20]
# ckpts = [40000, 80000, 120000, 160000, 200000]
ckpts = [20000, 40000, 60000, 300000]


FPwinsz30 = []
FPwinsz10 = []
FPwinsz20 = []

FNwinsz30 = []
FNwinsz10 = []
FNwinsz20 = []

for ckpt in ckpts:
    FP30 = 0
    FN30 = 0

    FP10 = 0
    FN10 = 0

    FP20 = 0
    FN20 = 0

    for window_size in win_sz:
        file_path = f"gradient_stats/nut_{ckpt}_winsz={window_size}.csv"
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for flist in reader:
                if flist[1] == "FP":
                    if window_size == 30:
                        FP30 += 1
                    elif window_size == 10:
                        FP10 += 1
                    else:
                        FP20 += 1
                elif flist[1] == "FN":
                    if window_size == 30:
                        FN30 += 1
                    elif window_size == 10:
                        FN10 += 1
                    else:
                        FN20 += 1
    FPwinsz30.append(FP30)
    FPwinsz10.append(FP10)
    FPwinsz20.append(FP20)

    FNwinsz30.append(FN30)
    FNwinsz10.append(FN10)
    FNwinsz20.append(FN20)
    
df = pd.DataFrame({
    'Checkpoint': ckpts,
    'Window 10': FPwinsz10,
    'Window 20': FPwinsz20,
    'Window 30': FPwinsz30,
})

# Set checkpoint as index
df.set_index('Checkpoint', inplace=True)

# Plot - pandas handles spacing automatically!
df.plot(kind='bar', figsize=(12, 6))

plt.xlabel('Checkpoints')
plt.ylabel('FP Count')
plt.title('False Positives by Checkpoint and Window Size')
plt.legend(title='Window Size')
plt.tight_layout()
plt.savefig("nut_FP_count_ckpts.png")

df = pd.DataFrame({
    'Checkpoint': ckpts,
    'Window 10': FNwinsz10,
    'Window 20': FNwinsz20,
    'Window 30': FNwinsz30,
})

# Set checkpoint as index
df.set_index('Checkpoint', inplace=True)

# Plot - pandas handles spacing automatically!
df.plot(kind='bar', figsize=(12, 6))

plt.xlabel('Checkpoints')
plt.ylabel('FN Count')
plt.title('False Negatives by Checkpoint and Window Size')
plt.legend(title='Window Size')
plt.tight_layout()
plt.savefig("nut_FN_count_ckpts.png")