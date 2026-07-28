import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import random

import jax
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, ReplayBuffer
from utils.flax_utils import restore_agent
from utils.log_utils import get_exp_name, get_flag_dict, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'robomimic-lift', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')
flags.DEFINE_integer('num_video_episodes', 5, 'How many episodes to render.')

config_flags.DEFINE_config_file('agent', 'agents/fql.py', lock_config=False)


def render_episode(agent, eval_env, key):
    """Run one greedy episode, return (video_frames (T,C,H,W) uint8, success)."""
    obs, info = eval_env.reset()
    frames, q_values = [], []
    done = False
    while not done:
        key, ak = jax.random.split(key)
        action = np.clip(np.array(agent.sample_actions(
            observations=obs, seed=ak, temperature=0)), -1, 1)
        qs = agent.network.select('critic')(obs[None], actions=action[None])
        q_values.append(float(jnp.mean(qs)))
        frames.append(eval_env.render().copy())      # RGB frame BEFORE stepping
        obs, reward, term, trunc, info = eval_env.step(action)
        done = term or trunc

    success = int(info.get("succeed", 0))
    q_values = np.array(q_values)
    print(f"  episode len={len(frames)} success={success} "
          f"Q first={q_values[0]:.2f} last={q_values[-1]:.2f}")

    vh, vw = frames[0].shape[:2]
    panel_w = vw
    dpi = 100
    fig, ax = plt.subplots(figsize=(panel_w / dpi, vh / dpi), dpi=dpi)
    canvas = FigureCanvas(fig)

    qmin, qmax = q_values.min(), q_values.max()
    pad = 0.1 * (qmax - qmin + 1e-6)

    target_hw = (vh, vw + panel_w)   # (H, W) every combined frame must match
    video_frames = []
    for i, frame in enumerate(frames):
        ax.clear()
        ax.plot(np.arange(i + 1), q_values[:i + 1], color="steelblue")
        ax.plot(i, q_values[i], "o", color="red")
        ax.set_xlim(0, len(frames))
        ax.set_ylim(qmin - pad, qmax + pad)
        ax.set_xlabel("step"); ax.set_ylabel("Q(s,a)")
        ax.set_title(f"success={bool(success)}  step={i}  Q={q_values[i]:.2f}", fontsize=9)
        ax.grid(True, alpha=0.3)
        canvas.draw()

        plot_img = np.asarray(canvas.buffer_rgba())[:, :, :3]        # RGB
        # resize plot panel to (vh, panel_w) using PIL to avoid an OpenCV dep
        from PIL import Image
        plot_img = np.asarray(Image.fromarray(plot_img).resize((panel_w, vh)))

        combined = np.hstack([frame, plot_img])                     # RGB, (vh, vw+panel_w, 3)
        if combined.shape[:2] != target_hw:
            combined = np.asarray(Image.fromarray(combined).resize((vw + panel_w, vh)))
        video_frames.append(combined.astype(np.uint8).transpose(2, 0, 1))  # (C,H,W)

    plt.close(fig)
    return np.stack(video_frames), success   # (T, C, H, W)


def main(_):
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project='fql', group=FLAGS.run_group, name=exp_name)
    flag_dict = get_flag_dict()

    config = FLAGS.agent
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name, frame_stack=FLAGS.frame_stack)

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    train_dataset = Dataset.create(**train_dataset)

    example_batch = train_dataset.sample(1)
    agent = agents[config['agent_name']].create(
        FLAGS.seed, example_batch['observations'], example_batch['actions'], config)

    assert FLAGS.restore_path is not None and FLAGS.restore_epoch is not None, \
        "Pass --restore_path (run dir) and --restore_epoch (e.g. 200000)."
    agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # sanity: critic should sit near your q_mean (~ -10), not ~0
    b = train_dataset.sample(256)
    mean_q = float(jnp.mean(agent.network.select('critic')(b['observations'], actions=b['actions'])))
    print(f"mean Q after restore: {mean_q:.3f}  (expect ~-10; near 0 => restore failed)")

    key = jax.random.PRNGKey(0)
    for ep in range(FLAGS.num_video_episodes):
        print(f"episode {ep}:")
        key, ek = jax.random.split(key)
        video_np, success = render_episode(agent, eval_env, ek)
        wandb.log(
            {
                f"q_synced_video_ep{ep}": wandb.Video(video_np, fps=10, format="mp4"),
                f"episode_success_ep{ep}": success,
            },
            step=FLAGS.restore_epoch,
        )
        print(f"  logged episode {ep} to wandb")


if __name__ == '__main__':
    app.run(main)