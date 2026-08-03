"""
Sweep over checkpoints, run evaluation episodes at each, and record:
  - one mp4 per (rendered) episode, named like  ep00000_success_step1-24.mp4
  - a pickle per checkpoint whose entries look like
      {'episode_num': 0, 'step_start': 1, 'step_end': 24, 'episode_length': 24,
       'success': True, 'q_values': [ ... one float per control step ... ]}
  - the success rate at each checkpoint, logged to wandb so it plots as a curve
    across checkpoints (x = offline step / epoch, y = success rate).

Checkpoints swept: ckpt_start, ckpt_start+interval, ... , ckpt_end inclusive
(default 40000, 80000, 120000, 160000, 200000).

Videos and pickle for epoch E go under:  <video_root>/<E>/
  <video_root>/40000/ep00000_success_step1-24.mp4
  <video_root>/40000/q_log_40000.pkl
(A per-epoch subfolder keeps the ep00000_... naming free of collisions across
checkpoints.)

Run headless on the second GPU, e.g.:
  CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl python eval_checkpoints.py \
      --restore_path=exp/fql/Debug/sd000_20260730_000059 \
      --env_name=robomimic-lift
"""

import os
import pickle

import numpy as np
import jax
import jax.numpy as jnp
import imageio
import wandb

from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, ReplayBuffer
from utils.flax_utils import restore_agent
from utils.log_utils import setup_wandb, get_exp_name, get_flag_dict

FLAGS = flags.FLAGS

flags.DEFINE_string("restore_path", None, "Run dir containing the saved checkpoints.")
flags.DEFINE_string("env_name", "robomimic-lift", "Environment (dataset) name.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("frame_stack", None, "Number of frames to stack.")

flags.DEFINE_integer("ckpt_start", 40000, "First checkpoint to evaluate.")
flags.DEFINE_integer("ckpt_end", 200000, "Last checkpoint to evaluate (inclusive).")
flags.DEFINE_integer("ckpt_interval", 40000, "Spacing between checkpoints.")

flags.DEFINE_integer("eval_episodes", 100,
                     "Episodes per checkpoint used for the success rate and the pickle.")
flags.DEFINE_integer("video_episodes", 100,
                     "Of the eval episodes, how many (the first N) to also render to mp4. "
                     "Set lower (e.g. 5) to save time/disk while still scoring 100.")
flags.DEFINE_string("video_root", "100_traj", "Root folder for per-checkpoint videos + pickle.")
flags.DEFINE_integer("video_fps", 10, "FPS for saved videos (match CONTROL_FREQ).")

flags.DEFINE_string("run_group", "ckpt_sweep", "wandb run group.")
flags.DEFINE_string("wandb_project", "fql", "wandb project.")

config_flags.DEFINE_config_file("agent", "agents/fql.py", lock_config=False)

flags.mark_flag_as_required("restore_path")


def q_of(agent, obs, action):
    """Mean-over-ensemble Q(s, a) for the plain FQL critic."""
    qs = agent.network.select("critic")(obs[None], actions=action[None])
    return float(jnp.mean(qs))


def run_episode(agent, eval_env, key, record_video):
    """One greedy (temperature=0) rollout.

    Returns (q_values list, success int 0/1, frames list or [], new key).
    Q and the render frame are taken for the CURRENT obs+action, before stepping,
    so len(frames) == len(q_values): a clean 1:1 step alignment for later syncing.
    """
    obs, info = eval_env.reset()
    q_values, frames = [], []
    success = 0
    done = False
    while not done:
        key, ak = jax.random.split(key)
        action = np.clip(
            np.array(agent.sample_actions(observations=obs, seed=ak, temperature=0)),
            -1, 1,
        )
        q_values.append(q_of(agent, obs, action))
        if record_video:
            frames.append(np.asarray(eval_env.render()).copy())
        obs, reward, term, trunc, info = eval_env.step(action)
        success = max(success, int(info.get("succeed", 0)))
        done = term or trunc

    return q_values, success, frames, key


def evaluate_checkpoint(agent, eval_env, epoch, key):
    """Run eval_episodes rollouts, write videos + pickle, return (success_rate, key)."""
    ckpt_dir = os.path.join(FLAGS.video_root, str(epoch))
    os.makedirs(ckpt_dir, exist_ok=True)

    records = []
    successes = 0
    global_step = 1  # cumulative step counter across episodes (for the step1-24 naming)

    for ep in range(FLAGS.eval_episodes):
        record_video = ep < FLAGS.video_episodes
        q_values, success, frames, key = run_episode(agent, eval_env, key, record_video)

        length = len(q_values)
        step_start = global_step
        step_end = global_step + length - 1
        global_step = step_end + 1

        records.append({
            "episode_num": ep,
            "step_start": step_start,
            "step_end": step_end,
            "episode_length": length,
            "success": bool(success),
            "q_values": q_values,
        })
        successes += success

        if record_video and frames:
            succ_str = "success" if success else "fail"
            fname = f"ep{ep:05d}_{succ_str}_step{step_start}-{step_end}.mp4"
            out_path = os.path.join(ckpt_dir, fname)
            writer = imageio.get_writer(out_path, fps=FLAGS.video_fps, macro_block_size=1)
            for f in frames:
                writer.append_data(f.astype(np.uint8))
            writer.close()

        if (ep + 1) % 10 == 0 or ep == FLAGS.eval_episodes - 1:
            print(f"    [epoch {epoch}] {ep + 1}/{FLAGS.eval_episodes} eps, "
                  f"running success rate = {successes / (ep + 1):.3f}")

    # Write the per-checkpoint pickle (list of episode records).
    pkl_path = os.path.join(ckpt_dir, f"q_log_{epoch}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(records, f)
    print(f"    wrote {pkl_path}  ({len(records)} episodes)")

    return successes / FLAGS.eval_episodes, key


def main(_):
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project=FLAGS.wandb_project, group=FLAGS.run_group, name=exp_name)
    # Make success rate plot against epoch on the wandb x-axis.
    wandb.define_metric("eval/epoch")
    wandb.define_metric("eval/success_rate", step_metric="eval/epoch")

    config = FLAGS.agent
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name, frame_stack=FLAGS.frame_stack)

    np.random.seed(FLAGS.seed)
    train_dataset = Dataset.create(**train_dataset)

    # Build the agent skeleton once; restore different params per checkpoint.
    example_batch = train_dataset.sample(1)
    agent = agents[config["agent_name"]].create(
        FLAGS.seed, example_batch["observations"], example_batch["actions"], config)

    os.makedirs(FLAGS.video_root, exist_ok=True)
    key = jax.random.PRNGKey(FLAGS.seed)

    epochs = list(range(FLAGS.ckpt_start, FLAGS.ckpt_end + 1, FLAGS.ckpt_interval))
    print(f"Evaluating checkpoints: {epochs}")

    for epoch in epochs:
        try:
            agent = restore_agent(agent, FLAGS.restore_path, epoch)
        except Exception as e:
            print(f"[epoch {epoch}] could not restore (skipping): {e}")
            continue

        # Sanity: critic should sit near your training q_mean, not ~0.
        b = train_dataset.sample(256)
        mean_q = float(jnp.mean(agent.network.select("critic")(
            b["observations"], actions=b["actions"])))
        print(f"[epoch {epoch}] restored. mean Q on data = {mean_q:.3f}")

        success_rate, key = evaluate_checkpoint(agent, eval_env, epoch, key)
        print(f"[epoch {epoch}] SUCCESS RATE over {FLAGS.eval_episodes} eps = {success_rate:.3f}")

        wandb.log({"eval/success_rate": success_rate, "eval/epoch": epoch})

    print("Done.")


if __name__ == "__main__":
    app.run(main)