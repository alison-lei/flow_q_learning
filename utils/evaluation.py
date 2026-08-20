from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
# from moviepy import ImageSequenceClip

def run_single_episode_video(agent, env, octo_model, tasks, out_path="eval_rollout.mp4", fps=10, temperature=0):
    """Run one episode, save an mp4 with the rendered frame next to a
    live Q-value plot. Returns (success, episode_length, q_values)."""

    def add_embeddings(observation):
        image_primary = observation["side_policy_256"][np.newaxis, ...]
        image_wrist = observation["wrist_1"][np.newaxis, ...]
        timestep_pad_mask = np.array([[True, True]])
        octo_obs = {
            "image_primary": image_primary,
            "image_wrist": image_wrist,
            "timestep_pad_mask": timestep_pad_mask
        }
    
        transformer_outputs = octo_model.sample_transformer(octo_obs, tasks, train=False)
        action_embeddings = transformer_outputs[:, -1, :]                                  # (1, embed_dim)

        observation = dict(observation)
        observation["embeddings"] = action_embeddings[0]
        return observation

    rng = jax.random.PRNGKey(np.random.randint(0, 2**32))
    observation, info = env.reset()
    observation = add_embeddings(observation)
    done = False
    frames, q_values = [], []

    while not done:
        rng, key = jax.random.split(rng)
        batched_obs = jax.tree_util.tree_map(lambda x: x[None], observation)
        action = agent.sample_actions(observations=batched_obs, seed=key, temperature=temperature)
        action = np.clip(np.array(action)[0], -1, 1)

        qs = agent.network.select('critic')(batched_obs, actions=action[None])
        q_values.append(float(jnp.mean(qs)))

        frames.append(env.render().copy())

        next_observation, reward, terminated, truncated, info = env.step(action)
        next_observation = add_embeddings(next_observation)
        done = terminated or truncated
        observation = next_observation

    success = bool(info.get("succeed", False))

    # --- compose frame + running Q-plot into one video, fixed axes so it doesn't jitter ---
    composite_frames = []
    q_min, q_max = min(q_values), max(q_values)
    pad = 0.1 * (q_max - q_min + 1e-6)
    for t in range(len(frames)):
        fig, (ax_img, ax_q) = plt.subplots(1, 2, figsize=(10, 4))
        ax_img.imshow(frames[t]); ax_img.axis("off"); ax_img.set_title(f"step {t}")

        ax_q.plot(q_values[: t + 1], color="tab:blue")
        ax_q.set_xlim(0, len(q_values))
        ax_q.set_ylim(q_min - pad, q_max + pad)
        ax_q.set_xlabel("step"); ax_q.set_ylabel("Q value"); ax_q.set_title("critic Q estimate")

        fig.tight_layout()
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
        composite_frames.append(img)
        plt.close(fig)

    ImageSequenceClip(composite_frames, fps=fps).write_videofile(out_path, codec="libx264", audio=False, logger=None)
    print(f"episode length={len(frames)} success={success} saved to {out_path}")
    return success, len(frames), q_values


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


# proprioception
# def evaluate(
#     agent,
#     env,
#     config=None,
#     num_eval_episodes=50,
#     num_video_episodes=0,
#     video_frame_skip=3,
#     eval_temperature=0,
# ):
#     """Evaluate the agent in the environment.

#     Args:
#         agent: Agent.
#         env: Environment.
#         config: Configuration dictionary.
#         num_eval_episodes: Number of episodes to evaluate the agent.
#         num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
#         video_frame_skip: Number of frames to skip between renders.
#         eval_temperature: Action sampling temperature.

#     Returns:
#         A tuple containing the statistics, trajectories, and rendered videos.
#     """
#     actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
#     # list containing dictionaries, each dictionary is an episode and contains lists of observations (ex) at each timestep
#     trajs = []
#     stats = defaultdict(list)
#     renders = []

#     # iterate through each episode
#     for i in trange(num_eval_episodes + num_video_episodes):
#         traj = defaultdict(list)
#         should_render = i >= num_eval_episodes

#         # first observation when environment resets
#         observation, info = env.reset()
#         done = False
#         step = 0
#         render = []
#         episode_q_values = []

#         while not done:
#             # distillation actor given observation produces action
#             action = actor_fn(observations=observation, temperature=eval_temperature)
#             action = np.array(action)
#             action = np.clip(action, -1, 1)
            
#             qs = agent.network.select('critic')(observation[None], actions=action[None]) # current estimate of Q in this state, critic for one-step specifically
#             q = jnp.mean(qs, axis=0)
#             # episode_q_values store mean q value per observation-action value pair
#             episode_q_values.append(q)

#             # environment takes that step and reports the outcome like next observations and whether was success
#             next_observation, reward, terminated, truncated, info = env.step(action)
#             done = terminated or truncated
#             step += 1

#             # saves the frames to create videos
#             if should_render and (step % video_frame_skip == 0 or done):
#                 frame = env.render().copy()
#                 render.append(frame)

#             transition = dict(
#                 observation=observation,
#                 next_observation=next_observation,
#                 action=action,
#                 reward=reward,
#                 done=done,
#                 info=info,
#             )
#             add_to(traj, transition)
#             observation = next_observation
#         if i < num_eval_episodes:
#             print({k: type(v).__name__ for k, v in flatten(info).items()})
#             add_to(stats, flatten(info))
#             trajs.append(traj)
            
#             stats["trajectory_length"].append(step)
#             stats["mean_q_value"].append(np.mean(episode_q_values)) # gets average q in each episode
#             stats["final_q_value"].append(episode_q_values[-1]) # gets final q in each episode
#         else:
#             renders.append(np.array(render))

#     for k, v in stats.items():
#         try:
#             stats[k] = np.mean(v)
#         except (TypeError, ValueError):
#             print("TypeError or ValueError")
#             print(f"non-numeric stat key={k} sample={v}")
#             stats[k] = np.mean([float(x) for x in np.asarray(v) != ''])
#             continue


#     return stats, trajs, renders



# images + proprio
def evaluate(agent, env, octo_model, tasks, config=None, num_eval_episodes=50,
             num_video_episodes=5, video_frame_skip=1, eval_temperature=0):

    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)
    renders = []

    def add_embeddings(observation):
        image_primary = observation["side_policy_256"][np.newaxis, ...]
        image_wrist = observation["wrist_1"][np.newaxis, ...]
        timestep_pad_mask = np.array([[True, True]])
        octo_obs = {
            "image_primary": image_primary,
            "image_wrist": image_wrist,
            "timestep_pad_mask": timestep_pad_mask
        }
    
        transformer_outputs = octo_model.sample_transformer(octo_obs, tasks, train=False)
        action_embeddings = transformer_outputs[:, -1, :]                                  # (1, embed_dim)
    
        observation = dict(observation)
        observation["embeddings"] = action_embeddings[0]
        return observation

    # for each rollout
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset()
        observation = add_embeddings(observation)          # <- run through Octo before the loop starts
        done = False
        step = 0
        render = []
        episode_q_values = []
        
        # while rollout is not done
        while not done:
            batched_obs = jax.tree_util.tree_map(lambda x: x[None], observation)
            action = actor_fn(observations=batched_obs, temperature=eval_temperature)
            action = np.array(action)[0]
            action = np.clip(action, -1, 1)

            qs = agent.network.select('critic')(batched_obs, actions=action[None])
            q = jnp.mean(qs, axis=0)
            episode_q_values.append(q)

            next_observation, reward, terminated, truncated, info = env.step(action)
            next_observation = add_embeddings(next_observation)   # <- and again every step
            done = terminated or truncated
            step += 1
            
            # saves the frames to create videos
            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                render.append(frame)

            obs_to_store = {k: v for k, v in observation.items() if k not in ("side_policy_256", "wrist_1")}
            next_obs_to_store = {k: v for k, v in next_observation.items() if k not in ("side_policy_256", "wrist_1")}

            transition = dict(
                observation=obs_to_store,
                next_observation=next_obs_to_store,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)


            observation = next_observation
        
        if i < num_eval_episodes:
            print({k: type(v).__name__ for k, v in flatten(info).items()})
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

