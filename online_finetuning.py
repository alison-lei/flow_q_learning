"""
Restore a qc/FQL offline checkpoint and run
online fine-tuning, while:

  - saving a video of EVERY episode locally
  - uploading one episode's video to wandb every --wandb_video_interval steps
  - recording per-episode Q values + success into a pickle file

Place in the qc repo root (next to main.py) and run with the SAME agent/env
flags used for the original offline training, plus --restore_path/--restore_epoch:

  MUJOCO_GL=egl python online_finetune_qvideo.py \
      --restore_path=exp/qc/reproduce/lift-mh-low_dim/sd000_20260721_195741 \
      --restore_epoch=1000000 \
      --env_name=lift-mh-low_dim --sparse=False --horizon_length=1 \
      --agent.alpha=10000 \
      --online_steps=100000

Requires moviepy + imageio-ffmpeg for wandb video encoding:
  pip install moviepy imageio-ffmpeg "numpy<2"
"""

import os
import pickle
import time

import numpy as np
import jax
import wandb
import imageio
import tqdm

from absl import app, flags
from ml_collections import config_flags

from envs.env_utils import make_env_and_datasets
from envs.robomimic_utils import is_robomimic_env
from utils.datasets import Dataset, ReplayBuffer
from utils.flax_utils import restore_agent, save_agent
from agents import agents

try:
    from log_utils import setup_wandb, get_exp_name
except ImportError:
    setup_wandb = None

FLAGS = flags.FLAGS

flags.DEFINE_string("restore_path", None, "Offline checkpoint dir to warm-start from.")
flags.DEFINE_integer("restore_epoch", None, "Checkpoint step to restore.")
flags.DEFINE_string("env_name", "lift-mh-low_dim", "Environment (dataset) name.")
flags.DEFINE_bool("sparse", False, "Sparse reward (must match the checkpoint's training).")
flags.DEFINE_integer("horizon_length", 1, "Action chunk length (must match the checkpoint).")
flags.DEFINE_integer("seed", 0, "Random seed.")

flags.DEFINE_integer("online_steps", 100000, "Number of online fine-tuning steps.")
flags.DEFINE_integer("buffer_size", 2000000, "Replay buffer size.")
flags.DEFINE_integer("utd_ratio", 1, "Update-to-data ratio.")
flags.DEFINE_integer("start_training", 0, "Env steps to collect before training starts "
                                           "(0 = train from the first online step).")
flags.DEFINE_float("discount", 0.99, "Discount factor (must match the checkpoint).")
flags.DEFINE_integer("log_interval", 1000, "Console/wandb scalar logging interval.")
flags.DEFINE_integer("save_interval", -1, "Agent checkpoint save interval (-1 = never).")

flags.DEFINE_string("video_dir", "online_videos", "Local dir for per-episode videos.")
flags.DEFINE_integer("wandb_video_interval", 10000,
                      "Upload one episode's video to wandb every N online steps.")
flags.DEFINE_integer("video_fps", 20, "FPS for saved/uploaded videos.")

flags.DEFINE_string("pickle_path", "online_finetune_log.pkl",
                     "Where to write per-episode Q values + success.")
flags.DEFINE_integer("pickle_save_interval", 10000,
                      "Re-write the pickle every N steps (crash safety), plus once at the end.")

flags.DEFINE_string("run_group", "online_finetune", "wandb run group.")
flags.DEFINE_string("wandb_project", "qc", "wandb project.")

config_flags.DEFINE_config_file("agent", "agents/acfql.py", lock_config=False)


def make_q_fn(agent):
    candidates = [
        lambda o, a: agent.network.select("critic")(o, actions=a),
        lambda o, a: agent.network.select("critic")(o, a),
    ]

    def f(obs, actions):
        last_err = None
        for fn in candidates:
            try:
                return np.asarray(jax.device_get(fn(obs, actions)))
            except Exception as e:  # noqa: BLE001
                last_err = e
        raise RuntimeError(
            "Could not call the critic. Check the signature with:\n"
            "  grep -n \"select('critic')\" agents/*.py\n"
            f"Last error: {last_err}"
        )

    return f


def policy_actions(agent, obs, rng):
    try:
        a = agent.sample_actions(observations=obs, rng=rng)
    except TypeError:
        a = agent.sample_actions(observations=obs, seed=rng)
    return np.asarray(jax.device_get(a))


# ---------------------------------------------------------------------------

def save_video_local(frames, path, fps):
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    try:
        imageio.mimsave(path, list(frames), fps=fps, macro_block_size=1)
    except Exception:
        # no ffmpeg -> fall back to gif so the episode isn't lost
        imageio.mimsave(os.path.splitext(path)[0] + ".gif", list(frames), fps=fps)


def dump_pickle(records, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(records, f)
    os.replace(tmp, path)  # atomic-ish, avoids a truncated file on crash mid-write


def main(_):
    assert FLAGS.restore_path is not None, "--restore_path is required"
    h = FLAGS.horizon_length
    os.makedirs(FLAGS.video_dir, exist_ok=True)

    config = FLAGS.agent
    config["horizon_length"] = h

    if setup_wandb is not None:
        setup_wandb(project=FLAGS.wandb_project, group=FLAGS.run_group,
                    name=get_exp_name(FLAGS.seed))
    else:
        wandb.init(project=FLAGS.wandb_project, group=FLAGS.run_group)

    save_dir = os.path.join("exp", FLAGS.wandb_project, FLAGS.run_group,
                             FLAGS.env_name, get_exp_name(FLAGS.seed)
                             if setup_wandb is not None else f"sd{FLAGS.seed:03d}")
    os.makedirs(save_dir, exist_ok=True)

    # --- env + offline dataset (dataset seeds the replay buffer) -----------
    env, _, train_raw, _ = make_env_and_datasets(FLAGS.env_name)
    ds = Dataset.create(**train_raw)
    if is_robomimic_env(FLAGS.env_name):
        d = {k: v for k, v in ds.items()}
        d["rewards"] = ds["rewards"] - 1.0
        ds = Dataset.create(**d)
    if FLAGS.sparse:
        d = {k: v for k, v in ds.items()}
        d["rewards"] = (ds["rewards"] != 0.0) * -1.0
        ds = Dataset.create(**d)

    action_dim = np.asarray(ds["actions"]).shape[-1]

    # --- restore the offline checkpoint -------------------------------------
    example = ds.sample(())
    agent = agents[config["agent_name"]].create(
        FLAGS.seed, example["observations"], example["actions"], config
    )
    agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)
    q_fn = make_q_fn(agent)
    print(f"restored checkpoint: {FLAGS.restore_path} @ step {FLAGS.restore_epoch}")

    replay_buffer = ReplayBuffer.create_from_initial_dataset(
        dict(ds), size=max(FLAGS.buffer_size, ds.size + 1)
    )

    # --- bookkeeping ---------------------------------------------------------
    rng = jax.random.PRNGKey(FLAGS.seed)
    episode_records = []          # -> pickled at the end (and periodically)
    next_video_upload_at = FLAGS.wandb_video_interval
    last_log_time = time.time()

    ob, _ = env.reset()
    action_queue = []
    ep_frames, ep_q, ep_start_step = [], [], 1
    update_info = {}
    q_val = None

    for i in tqdm.tqdm(range(1, FLAGS.online_steps + 1)):
        rng, key = jax.random.split(rng)

        if len(action_queue) == 0:
            chunk = policy_actions(agent, ob, key).reshape(-1)
            q_val = float(q_fn(np.asarray(ob)[None], chunk[None]).min(axis=0)[0])
            for a in chunk.reshape(-1, action_dim):
                action_queue.append(a)
        # q_val is held constant across the chunk it belongs to
        ep_frames.append(np.asarray(env.render()).astype(np.uint8))
        ep_q.append(q_val)

        action = action_queue.pop(0)
        next_ob, raw_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        int_reward = raw_reward
        if is_robomimic_env(FLAGS.env_name):
            int_reward = int_reward - 1.0
        if FLAGS.sparse:
            int_reward = (int_reward != 0.0) * -1.0

        replay_buffer.add_transition(dict(
            observations=ob,
            actions=action,
            rewards=int_reward,
            terminals=float(done),
            masks=1.0 - terminated,
            next_observations=next_ob,
        ))
        ob = next_ob

        # --- episode boundary: save video, log record, maybe upload --------
        if done:
            success = bool(info.get("success", raw_reward > 0))
            ep_idx = len(episode_records)
            tag = "success" if success else "failure"
            fname = f"ep{ep_idx:05d}_{tag}_step{ep_start_step}-{i}.mp4"
            local_path = os.path.join(FLAGS.video_dir, fname)
            save_video_local(ep_frames, local_path, FLAGS.video_fps)

            episode_records.append({
                "episode": ep_idx,
                "step_start": ep_start_step,
                "step_end": i,
                "length": len(ep_frames),
                "success": success,
                "q_values": list(ep_q),
                "local_video": local_path,
            })

            if i >= next_video_upload_at:
                frames_arr = np.stack(ep_frames).transpose(0, 3, 1, 2)  # T,C,H,W
                wandb.log({
                    f"online/video_{tag}": wandb.Video(frames_arr, fps=FLAGS.video_fps, format="mp4"),
                }, step=i)
                next_video_upload_at += FLAGS.wandb_video_interval

            ob, _ = env.reset()
            action_queue = []
            ep_frames, ep_q, ep_start_step = [], [], i + 1

        # --- training update -----------------------------------------------
        if i >= FLAGS.start_training:
            batch = replay_buffer.sample_sequence(
                config["batch_size"] * FLAGS.utd_ratio,
                sequence_length=h, discount=FLAGS.discount,
            )
            batch = jax.tree.map(
                lambda x: x.reshape((FLAGS.utd_ratio, config["batch_size"]) + x.shape[1:]),
                batch,
            )
            agent, update_info["online_agent"] = agent.batch_update(batch)

        # --- scalar logging --------------------------------------------------
        if i % FLAGS.log_interval == 0:
            recent = episode_records[-20:]
            if recent:
                sr = np.mean([r["success"] for r in recent])
                wandb.log({"online/recent_success_rate": sr,
                           "online/episodes_completed": len(episode_records),
                           "online/steps_per_sec": FLAGS.log_interval / (time.time() - last_log_time)},
                          step=i)
            for key, ui in update_info.items():
                wandb.log({f"{key}/{k}": v for k, v in ui.items()}, step=i)
            update_info = {}
            last_log_time = time.time()

        if i % FLAGS.pickle_save_interval == 0:
            dump_pickle(episode_records, FLAGS.pickle_path)

        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, save_dir, i)

    # final flush
    dump_pickle(episode_records, FLAGS.pickle_path)
    if FLAGS.save_interval > 0:
        save_agent(agent, save_dir, FLAGS.online_steps)

    n_success = sum(r["success"] for r in episode_records)
    print(f"done: {len(episode_records)} episodes, "
          f"{n_success}/{len(episode_records)} successful, "
          f"pickle -> {FLAGS.pickle_path}, videos -> {FLAGS.video_dir}/")
    wandb.finish()


if __name__ == "__main__":
    app.run(main)