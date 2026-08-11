import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import csv

win_sz = [5, 10, 20]
ckpts = [40000, 80000, 120000, 160000, 200000]

FPwinsz5 = []
FPwinsz10 = []
FPwinsz20 = []

FNwinsz5 = []
FNwinsz10 = []
FNwinsz20 = []

for ckpt in ckpts:
    FP5 = 0
    FN5 = 0

    FP10 = 0
    FN10 = 0

    FP20 = 0
    FN20 = 0

    for window_size in win_sz:
        file_path = f"gradient_stats/{ckpt}/winsz={window_size}.csv"
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for flist in reader:
                if flist[1] == "FP":
                    if window_size == 5:
                        FP5 += 1
                    elif window_size == 10:
                        FP10 += 1
                    else:
                        FP20 += 1
                elif flist[1] == "FN":
                    if window_size == 5:
                        FN5 += 1
                    elif window_size == 10:
                        FN10 += 1
                    else:
                        FN20 += 1
    FPwinsz5.append(FP5)
    FPwinsz10.append(FP10)
    FPwinsz20.append(FP20)

    FNwinsz5.append(FN5)
    FNwinsz10.append(FN10)
    FNwinsz20.append(FN20)
    
df = pd.DataFrame({
    'Checkpoint': ckpts,
    'Window 5': FPwinsz5,
    'Window 10': FPwinsz10,
    'Window 20': FPwinsz20
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
plt.savefig("FP_count_ckpts.png")

df = pd.DataFrame({
    'Checkpoint': ckpts,
    'Window 5': FNwinsz5,
    'Window 10': FNwinsz10,
    'Window 20': FNwinsz20
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
plt.savefig("FN_count_ckpts.png")