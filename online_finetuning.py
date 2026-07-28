import os
import pickle
import time

import numpy as np
import jax
import jax.numpy as jnp
import wandb
import imageio
import tqdm

from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, ReplayBuffer
from utils.evaluation import flatten
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import setup_wandb, get_exp_name, get_flag_dict

FLAGS = flags.FLAGS

flags.DEFINE_string("restore_path", None, "Offline checkpoint dir to warm-start from.")
flags.DEFINE_integer("restore_epoch", None, "Checkpoint step to restore.")
flags.DEFINE_string("env_name", "robomimic-lift", "Environment (dataset) name.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("frame_stack", None, "Number of frames to stack.")
flags.DEFINE_integer("balanced_sampling", 0, "Whether to use balanced sampling for online fine-tuning.")
flags.DEFINE_float("p_aug", None, "Probability of applying image augmentation.")

flags.DEFINE_integer("online_steps", 100000, "Number of online fine-tuning steps.")
flags.DEFINE_integer("buffer_size", 2000000, "Replay buffer size.")
flags.DEFINE_integer("log_interval", 1000, "Console/wandb scalar logging interval.")
flags.DEFINE_integer("save_interval", -1, "Agent checkpoint save interval (-1 = never).")

flags.DEFINE_string("video_dir", "online_videos", "Local dir for per-episode videos.")
flags.DEFINE_integer("wandb_video_interval", 10000,
                      "Upload one episode's video to wandb every N online steps.")
flags.DEFINE_integer("video_fps", 10, "FPS for saved/uploaded videos (matches CONTROL_FREQ).")

flags.DEFINE_string("pickle_path", "online_finetune_log.pkl",
                     "Where to write the per-episode record list.")
flags.DEFINE_integer("pickle_save_interval", 10000,
                      "Re-write the pickle every N steps (crash safety), plus once at the end.")

flags.DEFINE_string("run_group", "online_finetune", "wandb run group.")
flags.DEFINE_string("wandb_project", "fql", "wandb project.")

config_flags.DEFINE_config_file("agent", "agents/fql.py", lock_config=False)


def q_of(agent, obs, action):
    """Q(s, a) for the plain FQL critic ensemble, meaned over ensemble members."""
    qs = agent.network.select('critic')(obs[None], actions=action[None])
    return float(jnp.mean(qs))


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
    os.replace(tmp, path)  # avoids a truncated file if interrupted mid-write


def main(_):
    assert FLAGS.restore_path is not None, "--restore_path is required"
    os.makedirs(FLAGS.video_dir, exist_ok=True)

    config = FLAGS.agent
    setup_wandb(project=FLAGS.wandb_project, group=FLAGS.run_group, name=get_exp_name(FLAGS.seed))
    save_dir = os.path.join("exp", FLAGS.wandb_project, FLAGS.run_group, get_exp_name(FLAGS.seed))
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'flags.json'), 'w') as f:
        pickle.dump(get_flag_dict(), open(os.path.join(save_dir, 'flags.pkl'), 'wb'))

    # env: used for ONLINE rollouts. Same object collects data and renders
    # frames for the video, matching the rest of this codebase's online loop.
    _, eval_env, train_raw, val_raw = make_env_and_datasets(FLAGS.env_name, frame_stack=FLAGS.frame_stack)

    np.random.seed(FLAGS.seed)
    ds = Dataset.create(**train_raw)

    if FLAGS.balanced_sampling:
        example_transition = {k: v[0] for k, v in ds.items()}
        replay_buffer = ReplayBuffer.create(example_transition, size=FLAGS.buffer_size)
        train_dataset = ds
    else:
        # Offline data pre-populates the buffer; online transitions get
        # appended on top, so fine-tuning keeps access to the original demos.
        train_dataset = ReplayBuffer.create_from_initial_dataset(
            dict(ds), size=max(FLAGS.buffer_size, ds.size + 1)
        )
        replay_buffer = train_dataset

    for dataset in [train_dataset, replay_buffer]:
        dataset.p_aug = FLAGS.p_aug
        dataset.frame_stack = FLAGS.frame_stack

    # --- restore the offline checkpoint -------------------------------------
    example_batch = train_dataset.sample(1)
    agent = agents[config['agent_name']].create(
        FLAGS.seed, example_batch['observations'], example_batch['actions'], config,
    )
    agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Sanity check: a correctly-loaded critic sits near the offline q_mean
    # (roughly -10 on Lift). Near 0 means the restore silently failed.
    b = train_dataset.sample(256)
    mean_q = float(jnp.mean(agent.network.select('critic')(b['observations'], actions=b['actions'])))
    print(f"mean Q after restore: {mean_q:.3f} (expect ~-10; near 0 => restore failed)")

    # --- bookkeeping ---------------------------------------------------------
    rng = jax.random.PRNGKey(FLAGS.seed)
    episode_records = []          # -> pickled once at the end (and periodically)
    next_video_upload_at = FLAGS.wandb_video_interval
    last_log_time = time.time()

    ob, _ = eval_env.reset()
    ep_frames, ep_q, ep_start_step = [], [], 1
    update_info = {}

    for i in tqdm.tqdm(range(1, FLAGS.online_steps + 1)):
        rng, key = jax.random.split(rng)

        action = np.array(agent.sample_actions(observations=ob, temperature=1, seed=key))
        action = np.clip(action, -1, 1)
        q_val = q_of(agent, ob, action)

        ep_frames.append(np.asarray(eval_env.render()).astype(np.uint8))
        ep_q.append(q_val)

        next_ob, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated

        if 'antmaze' in FLAGS.env_name and (
            'diverse' in FLAGS.env_name or 'play' in FLAGS.env_name or 'umaze' in FLAGS.env_name
        ):
            reward = reward - 1.0  # D4RL antmaze convention, matches main.py

        replay_buffer.add_transition(dict(
            observations=ob,
            actions=action,
            rewards=reward,
            terminals=float(done),
            masks=1.0 - terminated,
            next_observations=next_ob,
        ))
        ob = next_ob

        # --- episode boundary: save video, log record, maybe upload --------
        if done:
            success = bool(info.get("succeed", 0))
            ep_idx = len(episode_records)
            tag = "success" if success else "fail"
            fname = f"ep{ep_idx:05d}_{tag}_step{ep_start_step}-{i}.mp4"
            local_path = os.path.join(FLAGS.video_dir, fname)
            save_video_local(ep_frames, local_path, FLAGS.video_fps)

            episode_records.append({
                "episode_num": ep_idx,
                "step_start": ep_start_step,
                "step_end": i,
                "episode_length": len(ep_frames),
                "success": success,
                "q_values": list(ep_q),
            })

            if i >= next_video_upload_at:
                frames_arr = np.stack(ep_frames).transpose(0, 3, 1, 2)  # T,C,H,W
                wandb.log({f"online/video_{tag}": wandb.Video(frames_arr, fps=FLAGS.video_fps, format="mp4")},
                          step=i)
                next_video_upload_at += FLAGS.wandb_video_interval

            ob, _ = eval_env.reset()
            ep_frames, ep_q, ep_start_step = [], [], i + 1
            expl_metrics = {f'exploration/{k}': np.mean(v) for k, v in flatten(info).items()}

        # --- training update -----------------------------------------------
        batch = replay_buffer.sample(config['batch_size'])
        agent, update_info = agent.update(batch)

        # --- scalar logging --------------------------------------------------
        if i % FLAGS.log_interval == 0:
            recent = episode_records[-20:]
            log = {f'training/{k}': v for k, v in update_info.items()}
            log['time/steps_per_sec'] = FLAGS.log_interval / (time.time() - last_log_time)
            if recent:
                log['online/recent_success_rate'] = np.mean([r["success"] for r in recent])
                log['online/episodes_completed'] = len(episode_records)
            wandb.log(log, step=i)
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