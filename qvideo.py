import numpy as np
import jax
import wandb
import matplotlib

matplotlib.use("Agg")  # headless server
import matplotlib.pyplot as plt

from absl import app, flags
from ml_collections import config_flags

from envs.env_utils import make_env_and_datasets
from envs.robomimic_utils import is_robomimic_env
from utils.datasets import Dataset
from utils.flax_utils import restore_agent
from agents import agents

try:
    from log_utils import setup_wandb, get_exp_name
except ImportError:  # fall back to plain wandb.init
    setup_wandb = None

FLAGS = flags.FLAGS

flags.DEFINE_string("restore_path", None, "Checkpoint dir (the run's save_dir).")
flags.DEFINE_integer("restore_epoch", None, "Checkpoint step to restore.")
flags.DEFINE_string("env_name", "lift-mh-low_dim", "Environment (dataset) name.")
flags.DEFINE_bool("sparse", False, "Sparse reward (must match training).")
flags.DEFINE_integer("horizon_length", 1, "Action chunk length (must match training).")
flags.DEFINE_integer("num_episodes", 5, "How many rollouts to record.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("fps", 20, "Playback frame rate of the logged video.")
flags.DEFINE_integer("frame_skip", 1, "Record every Nth env step (1 = every step).")
flags.DEFINE_string("run_group", "qvideo", "wandb run group.")
flags.DEFINE_string("wandb_project", "qc", "wandb project.")

config_flags.DEFINE_config_file("agent", "agents/acfql.py", lock_config=False)


# ---------------------------------------------------------------------------
# Agent API shims (same as plot_q_episodes.py)
# ---------------------------------------------------------------------------

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
# Rollout: collect frames + a Q value for every recorded timestep
# ---------------------------------------------------------------------------

def rollout(agent, q_fn, env, action_dim, h, rng):
    ob, _ = env.reset()
    frames, q_per_step = [], []
    queue, chunk_q = [], None
    t, success_t = 0, None

    while True:
        # New chunk whenever the queue empties. Q is defined on (state, chunk),
        # so it is held constant while that chunk executes.
        if len(queue) == 0:
            rng, key = jax.random.split(rng)
            chunk = policy_actions(agent, ob, key).reshape(-1)
            chunk_q = float(q_fn(np.asarray(ob)[None], chunk[None]).min(axis=0)[0])
            for a in chunk.reshape(-1, action_dim):
                queue.append(a)

        if t % FLAGS.frame_skip == 0:
            frames.append(np.asarray(env.render()).astype(np.uint8))
            q_per_step.append(chunk_q)

        ob, reward, terminated, truncated, _ = env.step(queue.pop(0))
        t += 1
        if success_t is None and reward > 0:
            success_t = len(frames) - 1  # index into the recorded frames
        if terminated or truncated:
            break

    return frames, np.array(q_per_step), success_t, t, rng


# ---------------------------------------------------------------------------
# Compose [robot | Q plot] frames
# ---------------------------------------------------------------------------

def compose(frames, qs, success_idx, ep_i, config):
    h_px = frames[0].shape[0]
    dpi = 100
    fig, ax = plt.subplots(figsize=(4.4, h_px / dpi), dpi=dpi)

    # Static background: the whole Q curve, greyed out.
    ax.plot(np.arange(len(qs)), qs, color="0.8", lw=1.4)
    if success_idx is not None:
        ax.axvline(success_idx, color="tab:green", ls=":", lw=1.2)
    (trace,) = ax.plot([], [], color="tab:blue", lw=1.8)
    (dot,) = ax.plot([], [], "o", color="tab:red", ms=7, zorder=5)
    title = ax.set_title("", fontsize=9)

    pad = max(1e-6, (qs.max() - qs.min()) * 0.1)
    ax.set_xlim(0, max(1, len(qs) - 1))
    ax.set_ylim(qs.min() - pad, qs.max() + pad)
    ax.set_xlabel("timestep", fontsize=8)
    ax.set_ylabel("Q(s, a)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = []
    for i, frame in enumerate(frames):
        trace.set_data(np.arange(i + 1), qs[: i + 1])
        dot.set_data([i], [qs[i]])
        title.set_text(f"episode {ep_i}   t={i}   Q={qs[i]:.2f}")
        fig.canvas.draw()
        plot = np.asarray(fig.canvas.buffer_rgba())[..., :3]

        # Match heights, then place side by side.
        if plot.shape[0] != frame.shape[0]:
            k = min(plot.shape[0], frame.shape[0])
            plot, frame = plot[:k], frame[:k]
        combined = np.concatenate([frame, plot], axis=1)

        # h264 needs even dimensions.
        if combined.shape[0] % 2:
            combined = combined[:-1]
        if combined.shape[1] % 2:
            combined = combined[:, :-1]
        out.append(combined)

    plt.close(fig)
    # wandb.Video wants (time, channel, height, width)
    return np.stack(out).transpose(0, 3, 1, 2)


def main(_):
    assert FLAGS.restore_path is not None, "--restore_path is required"
    h = FLAGS.horizon_length
    rng = jax.random.PRNGKey(FLAGS.seed)

    config = FLAGS.agent
    config["horizon_length"] = h

    if setup_wandb is not None:
        setup_wandb(project=FLAGS.wandb_project, group=FLAGS.run_group,
                    name=get_exp_name(FLAGS.seed))
    else:
        wandb.init(project=FLAGS.wandb_project, group=FLAGS.run_group)

    # Env + dataset (dataset only needed to build the agent shapes / reward scale).
    _, eval_env, train_raw, _ = make_env_and_datasets(FLAGS.env_name)
    ds = Dataset.create(**train_raw)
    if is_robomimic_env(FLAGS.env_name):
        d = {k: v for k, v in ds.items()}
        d["rewards"] = ds["rewards"] - 1.0
        ds = Dataset.create(**d)
    action_dim = np.asarray(ds["actions"]).shape[-1]

    example = ds.sample(1)
    agent = agents[config["agent_name"]].create(
        FLAGS.seed, example["observations"], example["actions"], config
    )
    agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)
    q_fn = make_q_fn(agent)

    log, n_success = {}, 0
    for ep_i in range(FLAGS.num_episodes):
        frames, qs, succ_idx, n_steps, rng = rollout(
            agent, q_fn, eval_env, action_dim, h, rng)
        ok = succ_idx is not None
        n_success += int(ok)
        print(f"episode {ep_i}: {'success' if ok else 'failure'}  "
              f"{n_steps} steps  Q range [{qs.min():.2f}, {qs.max():.2f}]",
              flush=True)

        video = compose(frames, qs, succ_idx, ep_i, config)
        tag = "success" if ok else "failure"
        log[f"qvideo/ep{ep_i}_{tag}"] = wandb.Video(video, fps=FLAGS.fps, format="mp4")
        del frames, video

    log["qvideo/success_rate"] = n_success / FLAGS.num_episodes
    wandb.log(log, step=FLAGS.restore_epoch or 0)
    print(f"logged {FLAGS.num_episodes} videos to wandb "
          f"({n_success}/{FLAGS.num_episodes} successful)")
    wandb.finish()


if __name__ == "__main__":
    app.run(main)