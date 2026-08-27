"""
This is if you have the videos and you recorded all the q values in a pkl file
and now you want to join them
can either get a specific episode or get all successes or all failures
"""
"""
This is if you have the videos and you recorded all the q values in a CSV
file (e.g. from eval_single_checkpoint.py) and now you want to join them.
Can get a specific episode or get all successes or all failures.

Videos were recorded at 10 fps == CONTROL_FREQ, so one frame == one step == one
Q value: the sync is 1:1. (A proportional fallback is kept only in case a given
video's frame count ever differs from its q_values length.)

Usage:
  python make_q_synced_video.py \
      --episode 3 \
      --csv 100_traj/200000/eval_log_200000.csv \
      --video_dir 100_traj/200000 \
      --out_dir q_synced_videos
"""

import csv
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image

import imageio.v2 as imageio

from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_integer("episode", None,
                     "Episode number to render (matches episode_num in the csv "
                     "and ep<N> in the video filename).")
flags.DEFINE_string("csv", "eval_log_200000.csv",
                    "Path to the per-episode CSV log.")
flags.DEFINE_string("video_dir", "online_videos", "Folder of rendered episode videos.")
flags.DEFINE_string("out_dir", "./", "Where to write the combined video.")
flags.DEFINE_float("fps", None,
                   "Override output fps. Default: source video's fps (should be 10).")


VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".gif", ".webm")


def load_episodes(csv_path):
    """Normalize the CSV into a list of episode dicts (episode_num, q_values, ...)."""
    episodes = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append({
                "episode_num": int(row["episode_num"]),
                "step_start": int(row["step_start"]),
                "step_end": int(row["step_end"]),
                "episode_length": int(row["episode_length"]),
                "success": row["success"].strip().lower() in ("true", "1"),
                "q_values": json.loads(row["q_values"]),
            })

    if not episodes:
        raise ValueError("No episode entries with 'q_values' found in csv.")
    return episodes

# make videos plots for all fails
# def find_episode(episodes, success=True):
#     matches = [e for e in episodes if e.get("success") == success]
#     if not matches:
#         raise ValueError(
#             f"All successes"
#         )
#     return matches

# to find a specific episode number
def find_episode(episodes, ep_num):
    """Return the single entry whose episode_num == ep_num."""
    matches = [e for e in episodes if e.get("episode_num") == ep_num]
    if not matches:
        available = sorted(e.get("episode_num") for e in episodes)
        raise ValueError(
            f"Episode {ep_num} not found in csv. Available episode_num values: {available}"
        )
    if len(matches) > 1:
        print(f"Warning: {len(matches)} entries have episode_num={ep_num}; using the first.")
    return matches[0]


def find_video(video_dir, ep_num):
    """Locate the video whose name encodes this episode number.

    Filenames look like ep00000_success_step1-24.mp4, so we match the integer
    immediately after a leading 'ep'. Falls back to the first integer anywhere
    in the name if no 'ep<N>' prefix is present.
    """
    files = [f for f in os.listdir(video_dir) if f.lower().endswith(VIDEO_EXTS)]
    if not files:
        raise FileNotFoundError(f"No video files found in {video_dir!r}")

    def ep_index(name):
        m = re.search(r"ep0*?(\d+)", name)          # 'ep00003_...' -> 3
        if m:
            return int(m.group(1))
        m = re.search(r"(\d+)", name)               # fallback: first int anywhere
        return int(m.group(1)) if m else None

    for f in files:
        if ep_index(f) == ep_num:
            return os.path.join(video_dir, f)

    available = sorted(x for x in (ep_index(f) for f in files) if x is not None)
    raise FileNotFoundError(
        f"No video for episode {ep_num} in {video_dir!r}. "
        f"Episode numbers found in filenames: {available}"
    )


def read_frames(video_path):
    """Return (list of (H,W,3) uint8 RGB frames, fps)."""
    reader = imageio.get_reader(video_path)
    try:
        fps = float(reader.get_meta_data().get("fps", 10) or 10)
    except Exception:
        fps = 10.0
    frames = []
    for frame in reader:
        frame = np.asarray(frame)
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        frames.append(frame[:, :, :3].astype(np.uint8))
    reader.close()
    return frames, fps


def q_index_for_frame(i, n_frames, n_q):
    """1:1 when counts match; proportional otherwise (safety net)."""
    if n_frames == n_q:
        return i
    if n_frames <= 1 or n_q <= 1:
        return min(i, n_q - 1)
    return int(round(i * (n_q - 1) / (n_frames - 1)))


def build_combined_video(frames, q_values, success, dpi=100):
    q = np.asarray(q_values, dtype=np.float32)
    n_frames, n_q = len(frames), len(q)

    vh, vw = frames[0].shape[:2]
    panel_w = vw
    target_wh = (vw + panel_w, vh)  # PIL (W, H)

    qmin, qmax = float(q.min()), float(q.max())
    pad = 0.1 * (qmax - qmin + 1e-6)

    fig, ax = plt.subplots(figsize=(panel_w / dpi, vh / dpi), dpi=dpi)
    canvas = FigureCanvas(fig)

    out = []
    for i, frame in enumerate(frames):
        qi = q_index_for_frame(i, n_frames, n_q)

        ax.clear()
        ax.plot(np.arange(qi + 1), q[: qi + 1], color="steelblue")
        ax.plot(qi, q[qi], "o", color="red")
        ax.set_xlim(0, max(n_q - 1, 1))
        ax.set_ylim(qmin - pad, qmax + pad)
        ax.set_xlabel("step"); ax.set_ylabel("Q(s,a)")
        ax.set_title(f"success={bool(success)}  step={qi}  Q={q[qi]:.2f}", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout(pad=0.5)
        canvas.draw()

        plot_img = np.asarray(canvas.buffer_rgba())[:, :, :3]
        plot_img = np.asarray(Image.fromarray(plot_img).resize((panel_w, vh)))

        combined = np.hstack([frame, plot_img])
        if (combined.shape[1], combined.shape[0]) != target_wh:
            combined = np.asarray(Image.fromarray(combined).resize(target_wh))
        out.append(combined.astype(np.uint8))

    plt.close(fig)
    return out


def main(_):
    os.makedirs(FLAGS.out_dir, exist_ok=True)

    episodes = load_episodes(FLAGS.csv)
    # if want to find a specific episode num
    ep = find_episode(episodes, FLAGS.episode)
    video_path = find_video(FLAGS.video_dir, FLAGS.episode)

    frames, src_fps = read_frames(video_path)
    if not frames:
        raise ValueError(f"Video {video_path} contained no frames.")

    q_values = ep["q_values"]
    success = ep.get("success", False)
    print(f"episode {FLAGS.episode}: video={os.path.basename(video_path)}  "
          f"frames={len(frames)}  q_len={len(q_values)}  success={success}  "
          f"steps={ep.get('step_start')}-{ep.get('step_end')}")
    if len(frames) != len(q_values):
        print(f"  note: frame count != q length -> syncing proportionally "
              f"(check recording; at 10fps==CONTROL_FREQ they should match)")

    combined = build_combined_video(frames, q_values, success)

    out_fps = FLAGS.fps if FLAGS.fps is not None else src_fps
    out_path = os.path.join(FLAGS.out_dir, f"octo_q_synced_ep{FLAGS.episode:05d}.mp4")
    writer = imageio.get_writer(out_path, fps=out_fps, macro_block_size=1)
    for f in combined:
        writer.append_data(f)
    writer.close()
    print(f"wrote {out_path}  ({len(combined)} frames @ {out_fps:.1f} fps)")


    # eps gives all the failed episodes
    # success_count = 0
    # eps = find_episode(episodes)
    # for ep in eps:
    #     success_count += 1

    #     ep_num = ep["episode_num"]
    #     video_path = find_video(FLAGS.video_dir, ep_num)

    #     frames, src_fps = read_frames(video_path)
    #     if not frames:
    #         raise ValueError(f"Video {video_path} contained no frames.")

    #     q_values = ep["q_values"]
    #     success = ep.get("success", False)
    #     print(f"episode {ep_num}: video={os.path.basename(video_path)}  "
    #         f"frames={len(frames)}  q_len={len(q_values)}  success={success}  "
    #         f"steps={ep.get('step_start')}-{ep.get('step_end')}")
    #     if len(frames) != len(q_values):
    #         print(f"  note: frame count != q length -> syncing proportionally "
    #             f"(check recording; at 10fps==CONTROL_FREQ they should match)")

    #     combined = build_combined_video(frames, q_values, success)

    #     out_fps = FLAGS.fps if FLAGS.fps is not None else src_fps
    #     out_path = os.path.join(FLAGS.out_dir, f"q_synced_ep{ep_num:05d}.mp4")
    #     writer = imageio.get_writer(out_path, fps=out_fps, macro_block_size=1)
    #     for f in combined:
    #         writer.append_data(f)
    #     writer.close()
    #     print(f"wrote {out_path}  ({len(combined)} frames @ {out_fps:.1f} fps)")

    #     if success_count == 5:
    #         break

if __name__ == "__main__":
    app.run(main)




# """
# Videos were recorded at 10 fps == CONTROL_FREQ, so one frame == one step == one
# Q value: the sync is 1:1. (A proportional fallback is kept only in case a given
# video's frame count ever differs from its q_values length.)

# Usage:
#   python make_q_synced_video.py \
#       --episode 3 \
#       --pkl 70_online_finetune_log_100000.pkl \
#       --video_dir online_videos \
#       --out_dir q_synced_videos
# """

# import os
# import pickle
# import re

# import numpy as np
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from PIL import Image

# import imageio.v2 as imageio

# from absl import app, flags

# FLAGS = flags.FLAGS

# flags.DEFINE_integer("episode", None,
#                      "Episode number to render (matches episode_num in the pkl "
#                      "and ep<N> in the video filename).")
# flags.DEFINE_string("pkl", "70_online_finetune_log_100000.pkl",
#                     "Path to the per-episode pickle log.")
# flags.DEFINE_string("video_dir", "online_videos", "Folder of rendered episode videos.")
# flags.DEFINE_string("out_dir", "./", "Where to write the combined video.")
# flags.DEFINE_float("fps", None,
#                    "Override output fps. Default: source video's fps (should be 10).")


# VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".gif", ".webm")


# def load_episodes(pkl_path):
#     """Normalize whatever container the pkl uses into a list of episode dicts."""
#     with open(pkl_path, "rb") as f:
#         data = pickle.load(f)

#     if isinstance(data, dict) and "episodes" in data:
#         data = data["episodes"]

#     if isinstance(data, dict):
#         episodes = [data] if "q_values" in data else list(data.values())
#     elif isinstance(data, (list, tuple)):
#         episodes = list(data)
#     else:
#         raise ValueError(f"Unrecognized pkl structure: {type(data)}")

#     episodes = [e for e in episodes if isinstance(e, dict) and "q_values" in e]
#     if not episodes:
#         raise ValueError("No episode entries with 'q_values' found in pkl.")
#     return episodes

# # make videos plots for all fails
# # def find_episode(episodes, success=True):
# #     matches = [e for e in episodes if e.get("success") == success]
# #     if not matches:
# #         raise ValueError(
# #             f"All successes"
# #         )
# #     return matches

# # to find a specific episode number
# def find_episode(episodes, ep_num):
#     """Return the single entry whose episode_num == ep_num."""
#     matches = [e for e in episodes if e.get("episode_num") == ep_num]
#     if not matches:
#         available = sorted(e.get("episode_num") for e in episodes)
#         raise ValueError(
#             f"Episode {ep_num} not found in pkl. Available episode_num values: {available}"
#         )
#     if len(matches) > 1:
#         print(f"Warning: {len(matches)} entries have episode_num={ep_num}; using the first.")
#     return matches[0]


# def find_video(video_dir, ep_num):
#     """Locate the video whose name encodes this episode number.

#     Filenames look like ep00000_success_step1-24.mp4, so we match the integer
#     immediately after a leading 'ep'. Falls back to the first integer anywhere
#     in the name if no 'ep<N>' prefix is present.
#     """
#     files = [f for f in os.listdir(video_dir) if f.lower().endswith(VIDEO_EXTS)]
#     if not files:
#         raise FileNotFoundError(f"No video files found in {video_dir!r}")

#     def ep_index(name):
#         m = re.search(r"ep0*?(\d+)", name)          # 'ep00003_...' -> 3
#         if m:
#             return int(m.group(1))
#         m = re.search(r"(\d+)", name)               # fallback: first int anywhere
#         return int(m.group(1)) if m else None

#     for f in files:
#         if ep_index(f) == ep_num:
#             return os.path.join(video_dir, f)

#     available = sorted(x for x in (ep_index(f) for f in files) if x is not None)
#     raise FileNotFoundError(
#         f"No video for episode {ep_num} in {video_dir!r}. "
#         f"Episode numbers found in filenames: {available}"
#     )


# def read_frames(video_path):
#     """Return (list of (H,W,3) uint8 RGB frames, fps)."""
#     reader = imageio.get_reader(video_path)
#     try:
#         fps = float(reader.get_meta_data().get("fps", 10) or 10)
#     except Exception:
#         fps = 10.0
#     frames = []
#     for frame in reader:
#         frame = np.asarray(frame)
#         if frame.ndim == 2:
#             frame = np.stack([frame] * 3, axis=-1)
#         frames.append(frame[:, :, :3].astype(np.uint8))
#     reader.close()
#     return frames, fps


# def q_index_for_frame(i, n_frames, n_q):
#     """1:1 when counts match; proportional otherwise (safety net)."""
#     if n_frames == n_q:
#         return i
#     if n_frames <= 1 or n_q <= 1:
#         return min(i, n_q - 1)
#     return int(round(i * (n_q - 1) / (n_frames - 1)))


# def build_combined_video(frames, q_values, success, dpi=100):
#     q = np.asarray(q_values, dtype=np.float32)
#     n_frames, n_q = len(frames), len(q)

#     vh, vw = frames[0].shape[:2]
#     panel_w = vw
#     target_wh = (vw + panel_w, vh)  # PIL (W, H)

#     qmin, qmax = float(q.min()), float(q.max())
#     pad = 0.1 * (qmax - qmin + 1e-6)

#     fig, ax = plt.subplots(figsize=(panel_w / dpi, vh / dpi), dpi=dpi)
#     canvas = FigureCanvas(fig)

#     out = []
#     for i, frame in enumerate(frames):
#         qi = q_index_for_frame(i, n_frames, n_q)

#         ax.clear()
#         ax.plot(np.arange(qi + 1), q[: qi + 1], color="steelblue")
#         ax.plot(qi, q[qi], "o", color="red")
#         ax.set_xlim(0, max(n_q - 1, 1))
#         ax.set_ylim(qmin - pad, qmax + pad)
#         ax.set_xlabel("step"); ax.set_ylabel("Q(s,a)")
#         ax.set_title(f"success={bool(success)}  step={qi}  Q={q[qi]:.2f}", fontsize=9)
#         ax.grid(True, alpha=0.3)
#         fig.tight_layout(pad=0.5)
#         canvas.draw()

#         plot_img = np.asarray(canvas.buffer_rgba())[:, :, :3]
#         plot_img = np.asarray(Image.fromarray(plot_img).resize((panel_w, vh)))

#         combined = np.hstack([frame, plot_img])
#         if (combined.shape[1], combined.shape[0]) != target_wh:
#             combined = np.asarray(Image.fromarray(combined).resize(target_wh))
#         out.append(combined.astype(np.uint8))

#     plt.close(fig)
#     return out


# def main(_):
#     os.makedirs(FLAGS.out_dir, exist_ok=True)

#     episodes = load_episodes(FLAGS.pkl)
#     # if want to find a specific episode num
#     ep = find_episode(episodes, FLAGS.episode)
#     video_path = find_video(FLAGS.video_dir, FLAGS.episode)

#     frames, src_fps = read_frames(video_path)
#     if not frames:
#         raise ValueError(f"Video {video_path} contained no frames.")

#     q_values = ep["q_values"]
#     success = ep.get("success", False)
#     print(f"episode {FLAGS.episode}: video={os.path.basename(video_path)}  "
#           f"frames={len(frames)}  q_len={len(q_values)}  success={success}  "
#           f"steps={ep.get('step_start')}-{ep.get('step_end')}")
#     if len(frames) != len(q_values):
#         print(f"  note: frame count != q length -> syncing proportionally "
#               f"(check recording; at 10fps==CONTROL_FREQ they should match)")

#     combined = build_combined_video(frames, q_values, success)

#     out_fps = FLAGS.fps if FLAGS.fps is not None else src_fps
#     out_path = os.path.join(FLAGS.out_dir, f"q_synced_ep{FLAGS.episode:05d}.mp4")
#     writer = imageio.get_writer(out_path, fps=out_fps, macro_block_size=1)
#     for f in combined:
#         writer.append_data(f)
#     writer.close()
#     print(f"wrote {out_path}  ({len(combined)} frames @ {out_fps:.1f} fps)")


#     # eps gives all the failed episodes
#     # success_count = 0
#     # eps = find_episode(episodes)
#     # for ep in eps:
#     #     success_count += 1

#     #     ep_num = ep["episode_num"]
#     #     video_path = find_video(FLAGS.video_dir, ep_num)

#     #     frames, src_fps = read_frames(video_path)
#     #     if not frames:
#     #         raise ValueError(f"Video {video_path} contained no frames.")

#     #     q_values = ep["q_values"]
#     #     success = ep.get("success", False)
#     #     print(f"episode {ep_num}: video={os.path.basename(video_path)}  "
#     #         f"frames={len(frames)}  q_len={len(q_values)}  success={success}  "
#     #         f"steps={ep.get('step_start')}-{ep.get('step_end')}")
#     #     if len(frames) != len(q_values):
#     #         print(f"  note: frame count != q length -> syncing proportionally "
#     #             f"(check recording; at 10fps==CONTROL_FREQ they should match)")

#     #     combined = build_combined_video(frames, q_values, success)

#     #     out_fps = FLAGS.fps if FLAGS.fps is not None else src_fps
#     #     out_path = os.path.join(FLAGS.out_dir, f"q_synced_ep{ep_num:05d}.mp4")
#     #     writer = imageio.get_writer(out_path, fps=out_fps, macro_block_size=1)
#     #     for f in combined:
#     #         writer.append_data(f)
#     #     writer.close()
#     #     print(f"wrote {out_path}  ({len(combined)} frames @ {out_fps:.1f} fps)")

#     #     if success_count == 5:
#     #         break

# if __name__ == "__main__":
#     app.run(main)