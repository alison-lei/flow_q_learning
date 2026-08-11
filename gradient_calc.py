"""
Compute intervention statistics (TP/FN/TN/FP, F1, accuracy) from a sliding-window
linear-regression slope of critic Q-values, evaluated one step at a time and
compared against the same decision points a training-time heuristic would hit.

Also maps every episode to its result bucket (TP/FN/TN/FP) and to its video
filename, so you can go straight from a stat to the clip that produced it.

Usage:
    python q_slope_stats.py \
        --pkl_path=/path/to/merged_eval_data.pkl \
        --video_dir=/path/to/40000 \
        --window_size=20 \
        --threshold=0.0 \
        --episodes_csv=/path/to/episode_results.csv

Expected pkl structure (list of per-episode dicts):
    merged_data = [
        {
            "episode_num": 0,
            "step_start": 1,
            "step_end": 30,
            "episode_length": 30,
            "success": True,
            "q_values": [...],   # list/array of per-step Q values, shape [T]
                                  # or [T, num_critics]
        },
        ...
    ]
"""

import csv
import glob
import os
import pickle

import numpy as np
from absl import app, flags
from scipy.stats import linregress
from sklearn.metrics import accuracy_score, f1_score

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'pkl_path', None, 'Path to the .pkl file with q_values for all episodes.'
)
flags.DEFINE_string(
    'video_dir', None,
    'Optional directory containing the per-episode video files '
    '(e.g. ep00000_success_step1-23.mp4). If set, each episode result is '
    'matched to its actual video file on disk; otherwise the expected '
    'filename is still printed even if the file cannot be located.'
)
flags.DEFINE_float(
    'threshold', 0.0, 'Slope threshold at/below which an intervention query is triggered.'
)
flags.DEFINE_string(
    'episodes_csv', None,
    'Optional path to write a per-episode CSV (episode_num, bucket, video, '
    'num_queries, first_query_step, success).'
)
flags.mark_flag_as_required('pkl_path')

winsz = [5, 10, 20]


def check_trend(data_window):
    """Return the slope of a simple linear regression over data_window."""
    x = np.arange(len(data_window))
    slope, _, _, _, _ = linregress(x, data_window)
    return slope


def process_line_fitting(data_dict, window_size):
    """Per-step slope, aligned with the training heuristic's decision points.

    At step t the training heuristic appends q_values[t] to its deque and then
    evaluates the slope; the earliest the deque can be full is t = window_size - 1
    (i.e. step 19 when window_size=20). slope[t] here is the slope of
    q_values[t - window_size + 1 : t + 1], so slope[t] matches the value the
    heuristic used to (potentially) trigger an intervention at step t.

    Returns an object-dtype array: the first `window_size - 1` entries are
    `None` (warmup, deque not yet full), the rest are floats.
    """
    critic_q_values = np.array(data_dict["q_values"]).squeeze()
    if critic_q_values.ndim == 1:
        # Single critic -> add axis so the mean below still works.
        critic_q_values = critic_q_values[:, None]
    critic_q_values = np.mean(critic_q_values, axis=1)
    trajectory_len = len(critic_q_values)

    n_pad = min(window_size - 1, trajectory_len)
    slopes = [None] * n_pad
    for t in range(window_size - 1, trajectory_len):
        data_window = critic_q_values[t - window_size + 1 : t + 1]
        slopes.append(check_trend(data_window))
    return np.array(slopes, dtype=object)


def expected_video_name(ep):
    """Reconstruct the expected video filename, e.g. ep00000_success_step1-23.mp4."""
    tag = "success" if ep.get("success") else "fail"
    return f"ep{ep['episode_num']:05d}_{tag}_step{ep['step_start']}-{ep['step_end']}.mp4"


def build_video_index(video_dir):
    """Map episode_num -> actual file path found in video_dir (best-effort)."""
    index = {}
    if not video_dir:
        return index
    for path in glob.glob(os.path.join(video_dir, "ep*.mp4")):
        base = os.path.basename(path)
        try:
            ep_num = int(base[2:7])  # 'ep00000_...' -> 00000
        except ValueError:
            continue
        index[ep_num] = path
    return index


def get_stats_from_merged(merged_data, process_data_func, window_size, threshold,
                           video_dir=None, episodes_csv=None):
    success_counter = 0
    TP = FN = TN = FP = 0
    query_true_labels = []
    query_predicted_labels = []

    video_index = build_video_index(video_dir)
    episode_rows = []  # for CSV / per-bucket printing
    buckets = {"TP": [], "FN": [], "TN": [], "FP": []}

    # Iterate episodes in order; works whether merged_data is a list or a dict.
    if isinstance(merged_data, dict):
        episodes = [merged_data[k] for k in sorted(merged_data.keys())]
    else:
        episodes = list(merged_data)
    episodes.sort(key=lambda ep: ep.get("episode_num", 0))

    for ep in episodes:
        ep_num = ep.get("episode_num")
        if "q_values" not in ep or len(ep["q_values"]) == 0:
            print(f"Skipping episode {ep_num}: no q_values")
            continue
        if "success" not in ep:
            print(f"Skipping episode {ep_num}: no 'success' (partial?)")
            continue

        success = bool(ep["success"])
        if success:
            success_counter += 1

        values = process_data_func(ep, window_size)
        warmup = window_size - 1

        first_query_step = None
        if len(values) > warmup:
            post_warmup = values[warmup:].astype(float)
            triggered = np.where(post_warmup <= threshold)[0]
            num_queries = len(triggered)
            if num_queries > 0:
                # Map back to the actual step index within the episode.
                first_query_step = int(triggered[0]) + warmup
        else:
            num_queries = 0
        query = num_queries > 0

        # These four cases are mutually exclusive and exhaustive.
        if query and not success:
            bucket = "TP"
            TP += 1
        elif not query and not success:
            bucket = "FN"
            FN += 1
        elif not query and success:
            bucket = "TN"
            TN += 1
        else:  # query and success
            bucket = "FP"
            FP += 1

        query_true_labels.append(not success)
        query_predicted_labels.append(bool(query))

        video_path = video_index.get(ep_num, expected_video_name(ep))
        buckets[bucket].append((ep_num, video_path))
        episode_rows.append({
            "episode_num": ep_num,
            "bucket": bucket,
            "video": video_path,
            "num_queries": num_queries,
            "first_query_step": first_query_step,
            "success": success,
        })

    total = TP + FN + TN + FP
    print("TOTAL:", total)
    if total > 0:
        print("SUCCESS RATE:", success_counter / total)
    print("TP:", TP, "FN:", FN, "TN:", TN, "FP:", FP)
    print("TPR (recall on failures):", TP / (TP + FN) if TP + FN > 0 else None)
    print("TNR (recall on successes):", TN / (TN + FP) if TN + FP > 0 else None)

    results = {"TP": TP, "FN": FN, "TN": TN, "FP": FP, "total": total}
    if query_true_labels:
        f1 = f1_score(query_true_labels, query_predicted_labels)
        acc = accuracy_score(query_true_labels, query_predicted_labels)
        print("F1:", f1)
        print("Acc:", acc)
        results["f1"] = f1
        results["accuracy"] = acc

    for bucket, entries in buckets.items():
        print(f"\n--- {bucket} episodes ({len(entries)}) ---")
        for ep_num, video_path in entries:
            print(f"  episode {ep_num}: {video_path}")

    if episodes_csv:
        with open(episodes_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["episode_num", "bucket", "video", "num_queries",
                               "first_query_step", "success"]
            )
            writer.writeheader()
            writer.writerows(episode_rows)
        print(f"\nWrote per-episode results to {episodes_csv}")

    results["episode_rows"] = episode_rows
    return results


def main(argv):
    with open(FLAGS.pkl_path, "rb") as f:
        merged_data = pickle.load(f)

    for win in winsz:
        print(f"Loaded {len(merged_data)} episodes from {FLAGS.pkl_path}")
        print(f"w_size={win}, threshold={FLAGS.threshold}")

        get_stats_from_merged(
            merged_data,
            process_line_fitting,
            win,
            FLAGS.threshold,
            video_dir=FLAGS.video_dir,
            episodes_csv=f"{FLAGS.episodes_csv}winsz={win}.csv",
        )


if __name__ == '__main__':
    app.run(main)