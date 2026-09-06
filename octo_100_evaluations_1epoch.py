"""
Evaluate ONE checkpoint (image+proprio, Octo-embedding FQL agent), run
eval_episodes trajectories, record an mp4 for each, and write a CSV log with
per-episode step range, success, and Q-value stats.

Videos + CSV for the given epoch go under:  <video_root>/<epoch>/
  <video_root>/200000/ep00000_success_step1-26.mp4
  <video_root>/200000/eval_log_200000.csv

CSV columns: episode_num, step_start, step_end, episode_length, success,
             mean_q_value, final_q_value, q_values (JSON-encoded list)

Run headless on the second GPU, e.g.:
  CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl python eval_single_checkpoint.py \
      --restore_path=exp/image_proprio/Debug/sd000_20260821_131218 \
      --restore_epoch=200000 \
      --env_name=robomimic-lift \
      --eval_seed=345
"""

import os
import csv
import json

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

from octo.model.octo_model import OctoModel

FLAGS = flags.FLAGS

flags.DEFINE_string("restore_path", None, "Run dir containing the saved checkpoint.")
flags.DEFINE_integer("restore_epoch", None, "Checkpoint step to evaluate.")
flags.DEFINE_string("env_name", "robomimic-lift", "Environment (dataset) name.")
flags.DEFINE_integer("seed", 0, "Random seed (agent skeleton / dataset sampling).")
flags.DEFINE_integer("eval_seed", None, "Seed for the eval env (read directly by RobosuiteBaseEnv).")
flags.DEFINE_integer("frame_stack", None, "Number of frames to stack.")

flags.DEFINE_integer("eval_episodes", 100, "Number of episodes to evaluate.")
flags.DEFINE_integer("video_episodes", 100,
                     "Of the eval episodes, how many (the first N) to also render to mp4.")
flags.DEFINE_string("video_root", "60000_nut_videos", "Root folder for videos + CSV.")
flags.DEFINE_integer("video_fps", 10, "FPS for saved videos (match CONTROL_FREQ).")

flags.DEFINE_string("run_group", "single_ckpt_eval", "wandb run group.")
flags.DEFINE_string("wandb_project", "nut", "wandb project.")

config_flags.DEFINE_config_file("agent", "agents/fql.py", lock_config=False)

flags.mark_flag_as_required("restore_path")
flags.mark_flag_as_required("restore_epoch")

# TASK_DESC = "pick up the red block"   # <- confirmed exact string from data collection
TASK_DESC = "pick up the brown square nut and insert it onto the square peg" # nut assembly square
OCTO_PATH = "hf://rail-berkeley/octo-small"


def make_add_embeddings(octo_model, tasks):
    def add_embeddings(observation):
        image_primary = observation["side_policy_256"][np.newaxis, ...]
        image_wrist = observation["wrist_1"][np.newaxis, ...]
        timestep_pad_mask = np.array([[True, True]])
        octo_obs = {
            "image_primary": image_primary,
            "image_wrist": image_wrist,
            "timestep_pad_mask": timestep_pad_mask,
        }
        transformer_outputs = octo_model.sample_transformer(octo_obs, tasks, train=False)
        action_embeddings = transformer_outputs[:, -1, :]  # (1, embed_dim)
        observation = dict(observation)
        observation["embeddings"] = action_embeddings[0]
        return observation
    return add_embeddings


def q_of(agent, batched_obs, action):
    """Mean-over-ensemble Q(s, a) for the image+proprio FQL critic."""
    qs = agent.network.select("critic")(batched_obs, actions=action[None])
    return float(jnp.mean(qs))


def run_episode(agent, eval_env, add_embeddings, key, record_video):
    """One greedy (temperature=0) rollout.

    Returns (q_values list, success int 0/1, frames list or [], new key).
    Q and the render frame are taken for the CURRENT obs+action, before
    stepping, so len(frames) == len(q_values): a clean 1:1 step alignment.
    """
    obs, info = eval_env.reset()
    obs = add_embeddings(obs)
    q_values, frames = [], []
    success = 0
    done = False
    while not done:
        key, ak = jax.random.split(key)
        batched_obs = jax.tree_util.tree_map(lambda x: x[None], obs)
        action = np.clip(
            np.array(agent.sample_actions(observations=batched_obs, seed=ak, temperature=0))[0],
            -1, 1,
        )
        q_values.append(q_of(agent, batched_obs, action))
        if record_video:
            frames.append(np.asarray(eval_env.render()).copy())

        next_obs, reward, term, trunc, info = eval_env.step(action)
        next_obs = add_embeddings(next_obs)
        success = max(success, int(info.get("succeed", 0)))
        done = term or trunc
        obs = next_obs

    return q_values, success, frames, key


def evaluate_checkpoint(agent, eval_env, add_embeddings, epoch, key):
    """Run eval_episodes rollouts, write videos + CSV, return (success_rate, key)."""
    ckpt_dir = os.path.join(FLAGS.video_root, str(epoch))
    os.makedirs(ckpt_dir, exist_ok=True)

    csv_path = os.path.join(ckpt_dir, f"eval_log_{epoch}.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "episode_num", "step_start", "step_end", "episode_length",
        "success", "mean_q_value", "final_q_value", "q_values",
    ])

    successes = 0
    episode_lengths = []
    global_step = 1  # cumulative step counter across episodes (for the step1-24 naming)

    for ep in range(FLAGS.eval_episodes):
        record_video = ep < FLAGS.video_episodes
        q_values, success, frames, key = run_episode(agent, eval_env, add_embeddings, key, record_video)

        length = len(q_values)
        step_start = global_step
        step_end = global_step + length - 1
        global_step = step_end + 1

        mean_q = float(np.mean(q_values)) if q_values else float("nan")
        final_q = float(q_values[-1]) if q_values else float("nan")

        writer.writerow([
            ep, step_start, step_end, length, bool(success),
            mean_q, final_q, json.dumps(q_values),
        ])

        successes += success
        episode_lengths.append(length)

        if record_video and frames:
            succ_str = "success" if success else "fail"
            fname = f"ep{ep:05d}_{succ_str}_step{step_start}-{step_end}.mp4"
            out_path = os.path.join(ckpt_dir, fname)
            vwriter = imageio.get_writer(out_path, fps=FLAGS.video_fps, macro_block_size=1)
            for f in frames:
                vwriter.append_data(f.astype(np.uint8))
            vwriter.close()

        if (ep + 1) % 10 == 0 or ep == FLAGS.eval_episodes - 1:
            print(f"    [epoch {epoch}] {ep + 1}/{FLAGS.eval_episodes} eps, "
                  f"running success rate = {successes / (ep + 1):.3f}")

    csv_file.close()
    print(f"    wrote {csv_path}  ({FLAGS.eval_episodes} episodes)")

    success_rate = successes / FLAGS.eval_episodes
    mean_length = float(np.mean(episode_lengths))
    return success_rate, mean_length, key


def main(_):
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project=FLAGS.wandb_project, group=FLAGS.run_group, name=exp_name)

    config = FLAGS.agent
    config["encoder"] = "image_proprio"
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name, frame_stack=FLAGS.frame_stack)
    if env is not None:
        env.close()

    octo_model = OctoModel.load_pretrained(OCTO_PATH)
    tasks = octo_model.create_tasks(texts=[TASK_DESC])
    add_embeddings = make_add_embeddings(octo_model, tasks)

    np.random.seed(FLAGS.seed)
    train_dataset = Dataset.create(**train_dataset)

    # Build the agent skeleton, then restore the single checkpoint.
    example_batch = train_dataset.sample(1)
    agent = agents[config["agent_name"]].create(
        FLAGS.seed, example_batch["observations"], example_batch["actions"], config)
    agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Sanity: critic should sit near your training q_mean, not ~0.
    b = train_dataset.sample(256)
    mean_q = float(jnp.mean(agent.network.select("critic")(
        b["observations"], actions=b["actions"])))
    print(f"[epoch {FLAGS.restore_epoch}] restored. mean Q on data = {mean_q:.3f}")

    os.makedirs(FLAGS.video_root, exist_ok=True)
    key = jax.random.PRNGKey(FLAGS.seed)

    success_rate, mean_length, key = evaluate_checkpoint(
        agent, eval_env, add_embeddings, FLAGS.restore_epoch, key)
    print(f"[epoch {FLAGS.restore_epoch}] SUCCESS RATE over {FLAGS.eval_episodes} eps "
          f"= {success_rate:.3f}, mean episode length = {mean_length:.1f}")

    wandb.log({
        "eval/success_rate": success_rate,
        "eval/mean_episode_length": mean_length,
        "eval/epoch": FLAGS.restore_epoch,
    })

    eval_env.close()
    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    app.run(main)