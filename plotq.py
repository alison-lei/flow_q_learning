import os

import numpy as np
import jax
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

FLAGS = flags.FLAGS

flags.DEFINE_string("restore_path", None, "Checkpoint dir (the run's save_dir).")
flags.DEFINE_integer("restore_epoch", None, "Checkpoint step to restore.")
flags.DEFINE_string("env_name", "lift-mh-low_dim", "Environment (dataset) name.")
flags.DEFINE_bool("sparse", False, "Sparse reward (must match training).")
flags.DEFINE_integer("horizon_length", 1, "Action chunk length (must match training).")
flags.DEFINE_integer("num_episodes", 5, "How many episodes to plot in each figure.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("out_path", "q_episodes.png", "Output image (dataset episodes).")
flags.DEFINE_string("out_path_env", "q_episodes_env.png", "Output image (env rollouts).")

config_flags.DEFINE_config_file("agent", "agents/acfql.py", lock_config=False)


def make_q_fn(agent):
    """Return f(obs, actions) -> (num_qs, batch) as numpy."""
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
    """qc's main.py uses sample_actions(observations=..., rng=...)."""
    try:
        a = agent.sample_actions(observations=obs, rng=rng)
    except TypeError:
        a = agent.sample_actions(observations=obs, seed=rng)
    return np.asarray(jax.device_get(a))


def style_axis(ax, title):
    ax.set_xlabel("timestep within episode")
    ax.set_ylabel("Q(s_t, a_t) values")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)


# ---------------------------------------------------------------------------
# Figure 1: Q along recorded dataset demonstrations
# ---------------------------------------------------------------------------

def plot_dataset_episodes(q_fn, obs_all, act_all, rew_all, term_all, h, colors, config):
    ends = np.flatnonzero(term_all > 0.5)
    episodes, start = [], 0
    for e in ends:
        episodes.append((start, int(e)))
        start = int(e) + 1
    episodes = [b for b in episodes if b[1] - b[0] >= h]

    # Evenly spaced episodes so we get a spread, not 5 near-identical ones.
    pick = np.linspace(0, len(episodes) - 1, FLAGS.num_episodes).astype(int)

    fig, ax = plt.subplots(figsize=(10, 6))
    for n, ep_i in enumerate(pick):
        s, e = episodes[ep_i]
        idx = np.arange(s, e - h + 1)
        obs = obs_all[idx]
        acts = np.stack([act_all[i : i + h].reshape(-1) for i in idx], axis=0)

        q = q_fn(obs, acts).min(axis=0)
        t = np.arange(len(q))
        ax.plot(t, q, lw=1.6, color=colors[n], label=f"episode {ep_i} (len {len(q)})")

        succ = np.flatnonzero(rew_all[idx] >= 0.0)  # reward back to 0 = success
        if len(succ):
            ax.scatter([succ[0]], [q[succ[0]]], color=colors[n],
                       marker="*", s=160, edgecolors="k", zorder=5)

    style_axis(ax, f"Q over DATASET episodes -- {FLAGS.env_name} @ step "
                   f"{FLAGS.restore_epoch}  (alpha={config.get('alpha', 'n/a')}, "
                   f"h={h})")
    fig.tight_layout()
    fig.savefig(FLAGS.out_path, dpi=140)
    print(f"wrote {FLAGS.out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Q along the agent's own evaluation rollouts
# ---------------------------------------------------------------------------

def plot_rollout_episodes(agent, q_fn, eval_env, action_dim, h, colors, config, rng):
    fig, ax = plt.subplots(figsize=(10, 6))
    n_success = 0

    for n in range(FLAGS.num_episodes):
        ob, _ = eval_env.reset()
        queue, q_series, t_series = [], [], []
        t, success_t = 0, None

        while True:
            # A new chunk is sampled whenever the queue empties; Q is defined on
            # (state, full chunk), so record it at each chunk boundary.
            if len(queue) == 0:
                rng, key = jax.random.split(rng)
                chunk = policy_actions(agent, ob, key).reshape(-1)
                q = q_fn(np.asarray(ob)[None], chunk[None]).min(axis=0)[0]
                q_series.append(float(q))
                t_series.append(t)
                for a in chunk.reshape(-1, action_dim):
                    queue.append(a)

            action = queue.pop(0)
            ob, reward, terminated, truncated, info = eval_env.step(action)
            t += 1
            if success_t is None and reward > 0:
                success_t = t
            if terminated or truncated:
                break

        ok = success_t is not None
        n_success += int(ok)
        ax.plot(t_series, q_series, lw=1.6, color=colors[n],
                label=f"rollout {n} ({'success' if ok else 'failure'}, {t} steps)")
        if ok:
            j = int(np.searchsorted(t_series, success_t) - 1)
            j = max(0, min(j, len(t_series) - 1))
            ax.scatter([t_series[j]], [q_series[j]], color=colors[n],
                       marker="*", s=160, edgecolors="k", zorder=5)

    style_axis(ax, f"Q over EVALUATION rollouts -- {FLAGS.env_name} @ step "
                   f"{FLAGS.restore_epoch}  (alpha={config.get('alpha', 'n/a')}, "
                   f"h={h})\n{n_success}/{FLAGS.num_episodes} succeeded")
    fig.tight_layout()
    fig.savefig(FLAGS.out_path_env, dpi=140)
    print(f"wrote {FLAGS.out_path_env}  ({n_success}/{FLAGS.num_episodes} successful)")


def main(_):
    assert FLAGS.restore_path is not None, "--restore_path is required"
    h = FLAGS.horizon_length
    rng = jax.random.PRNGKey(FLAGS.seed)

    config = FLAGS.agent
    config["horizon_length"] = h

    # Load env + dataset, applying the same reward shaping as main.py so the Q
    # scale matches what the critic was trained on.
    _, eval_env, train_raw, _ = make_env_and_datasets(FLAGS.env_name)
    ds = Dataset.create(**train_raw)
    if is_robomimic_env(FLAGS.env_name):
        d = {k: v for k, v in ds.items()}
        d["rewards"] = ds["rewards"] - 1.0
        ds = Dataset.create(**d)
    if FLAGS.sparse:
        d = {k: v for k, v in ds.items()}
        d["rewards"] = (ds["rewards"] != 0.0) * -1.0
        ds = Dataset.create(**d)

    obs_all = np.asarray(ds["observations"])
    act_all = np.asarray(ds["actions"])
    rew_all = np.asarray(ds["rewards"])
    term_all = np.asarray(ds["terminals"])
    action_dim = act_all.shape[-1]

    # Restore the agent.
    example = ds.sample(1)
    agent = agents[config["agent_name"]].create(
        FLAGS.seed, example["observations"], example["actions"], config
    )
    agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)
    q_fn = make_q_fn(agent)

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    plot_dataset_episodes(q_fn, obs_all, act_all, rew_all, term_all, h, colors, config)
    plot_rollout_episodes(agent, q_fn, eval_env, action_dim, h, colors, config, rng)


if __name__ == "__main__":
    app.run(main)