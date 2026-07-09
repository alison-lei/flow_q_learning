from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    agent,
    env,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    # list containing dictionaries, each dictionary is an episode and contains lists of observations (ex) at each timestep
    trajs = []
    stats = defaultdict(list)

    renders = []
    # iterate through each episode
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        # first observation when environment resets
        observation, info = env.reset()
        done = False
        step = 0
        render = []
        episode_q_values = []

        while not done:
            # distillation actor given observation produces action
            action = actor_fn(observations=observation, temperature=eval_temperature)
            action = np.array(action)
            action = np.clip(action, -1, 1)
            
            qs = agent.network.select('critic')(observation[None], actions=action[None]) # current estimate of Q in this state, critic for one-step specifically
            q = jnp.mean(qs, axis=0)
            # episode_q_values store mean q value per observation-action value pair
            episode_q_values.append(q)

            # environment takes that step and reports the outcome like next observations and whether was success
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            # saves the frames to create videos
            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
            
            stats["trajectory_length"].append(step)
            stats["mean_q_value"].append(np.mean(episode_q_values)) # gets average q in each episode
            stats["final_q_value"].append(episode_q_values[-1]) # gets final q in each episode
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        try:
            stats[k] = np.mean(v)
        except (TypeError, ValueError):
            print("TypeError or ValueError")
            print(f"non-numeric stat key={k} sample={v}")
            stats[k] = np.mean([float(x) for x in np.asarray(v) != ''])
            continue


    return stats, trajs, renders
