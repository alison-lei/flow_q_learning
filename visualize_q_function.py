# visualize the Q-functions on a graph, just as an image
# i think this is just to make the q function graphs
# python visualize_q_function.py --restore_path=exp/image_proprio/Debug/sd000_20260825_025957 --restore_epoch=200000

import os
import platform

import json
import random
import time

import jax
import numpy as np, jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, ReplayBuffer
from utils.evaluation import evaluate, flatten
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('eval_seed', 345, 'evaluation seed')
flags.DEFINE_string('env_name', 'robomimic-lift', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('offline_steps', 200000, 'Number of offline steps.')
flags.DEFINE_integer('online_steps', 0, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 500, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 10000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 20000, 'Saving interval.')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 5, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 1, 'Frame skip for videos.')
    
flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')
flags.DEFINE_integer('balanced_sampling', 0, 'Whether to use balanced sampling for online fine-tuning.')

config_flags.DEFINE_config_file('agent', 'agents/fql.py', lock_config=False)


def main(_):
    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project='fql', group=FLAGS.run_group, name=exp_name)
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)
    
    # Make environment and datasets.
    config = FLAGS.agent
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, frame_stack=FLAGS.frame_stack)

    # create popup window for live viewer
    # vis_env = robomimic_utils.make_eval_env(env_name="robomimic_lift", render_mode="human")
    
    # get rid of this assertion error to let lift task run
    # if FLAGS.video_episodes > 0:
    #     assert 'singletask' in FLAGS.env_name, 'Rendering is currently only supported for OGBench environments.'
    if FLAGS.online_steps > 0:
        assert 'visual' not in FLAGS.env_name, 'Online fine-tuning is currently not supported for visual environments.'
    
    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    # Set up datasets.
    train_dataset = Dataset.create(**train_dataset)
    
    if FLAGS.balanced_sampling:
        # Create a separate replay buffer so that we can sample from both the training dataset and the replay buffer.
        example_transition = {k: v[0] for k, v in train_dataset.items()}
        replay_buffer = ReplayBuffer.create(example_transition, size=FLAGS.buffer_size)
    else:
        # Use the training dataset as the replay buffer.
        train_dataset = ReplayBuffer.create_from_initial_dataset(
            dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
        )
        replay_buffer = train_dataset
    
    # Set p_aug and frame_stack.
    for dataset in [train_dataset, val_dataset, replay_buffer]:
        if dataset is not None:
            dataset.p_aug = FLAGS.p_aug
            dataset.frame_stack = FLAGS.frame_stack
            if config['agent_name'] == 'rebrac':
                dataset.return_next_actions = True
    
    # Create agent.
    example_batch = train_dataset.sample(1)
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # dont comment this out or then it starts a completely fresh agent with random weights
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)
    

    # this bottom part plots Q(s,a) on the dataset's own state-action pairs,
    # one line per training trajectory, in distinct colors, all in one figure.
    # pull the full dataset arrays (ReplayBuffer/Dataset stores them as fields)
    obs_all_dict = train_dataset['observations']   # dict: state, side_policy_256, wrist_1, embeddings
    act_all = np.asarray(train_dataset['actions'])
    term_all = np.asarray(train_dataset['terminals'])

    end_idxs = np.where(term_all > 0.5)[0]
    starts = np.concatenate([[0], end_idxs[:-1] + 1])
    bounds = list(zip(starts, end_idxs + 1))

    n_traj = len(bounds)
    cmap = plt.get_cmap('tab20' if n_traj <= 20 else 'turbo')
    colors = [cmap(i / max(n_traj - 1, 1)) if n_traj > 20 else cmap(i) for i in range(n_traj)]

    plt.figure(figsize=(11, 7))
    for ei, (s, e) in enumerate(bounds):
        obs = {k: np.asarray(v)[s:e] for k, v in obs_all_dict.items()}   # <- slice each field
        act = act_all[s:e]
        qs = agent.network.select('critic')(obs, actions=act)
        q = np.asarray(jnp.mean(qs, axis=0))
        plt.plot(q, marker='.', markersize=3, linewidth=1.3,
                color=colors[ei], label=f"traj {ei} (len {e-s})")
        print(f"traj {ei}: len={e-s}  Q first={q[0]:.2f}  last={q[-1]:.2f}")

    plt.xlabel("step"); plt.ylabel("Q(s, a)")
    plt.title(f"Per-step Q on all {n_traj} training trajectories (epoch {FLAGS.restore_epoch})")
    # 20 lines is too many for an inline legend -- put it outside the axes,
    # two columns, small font, so it doesn't overlap the curves.
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=1, fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig("20_demos_octo.png", dpi=150, bbox_inches='tight')
    print("saved 20_demos_octo.png")
    wandb.log({"q_train_traj": wandb.Image("20_demos_octo.png")}, step=FLAGS.restore_epoch)
    return

    # record graphs of q values per episode
    if FLAGS.restore_path is not None:
        b = train_dataset.sample(256)
        print("mean Q after restore:",
            float(jnp.mean(agent.network.select('critic')(b['observations'], actions=b['actions']))))

        key = jax.random.PRNGKey(0)
        episodes = []   # (q_trace, success)

        for ep in range(5):
            obs, info = eval_env.reset()
            done = False
            q_trace = []
            success = 0
            while not done:
                key, ak = jax.random.split(key)
                action = np.clip(np.array(agent.sample_actions(
                    observations=obs, seed=ak, temperature=0)), -1, 1)
                qs = agent.network.select('critic')(obs[None], actions=action[None])
                q_trace.append(float(jnp.mean(qs)))
                obs, reward, term, trunc, info = eval_env.step(action)
                done = term or trunc
                success = max(success, int(info.get("succeed", 0)))
            episodes.append((q_trace, success))
            print(f"ep {ep}: len={len(q_trace)} success={success} "
                  f"Q first={q_trace[0]:.2f} last={q_trace[-1]:.2f}")

        # plot: successes in one color, failures in another
        plt.figure(figsize=(8, 5))
        for q_trace, success in episodes:
            plt.plot(q_trace, color=("green" if success else "red"),
                     alpha=0.7, label=("success" if success else "fail"))
        # de-dup legend
        h, l = plt.gca().get_legend_handles_labels()
        plt.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys())
        plt.xlabel("step"); plt.ylabel("Q(s, a)")
        plt.title(f"Per-step Q values (epoch {FLAGS.restore_epoch})")
        plt.savefig("20_demos_octo.png", dpi=120)
        print("20_demos_octo.png")

        # optional: log to wandb too
        wandb.log({"q_traces": wandb.Image("20_demos_octo.png")}, step=FLAGS.restore_epoch)
        return


    # Evaluate on 20 episodes and record 5 videos
    # python main.py --restore_path=exp/fql/Debug/sd000_20260710_025149 --restore_epoch=100000 --eval_episodes=20 --video_episodes=5
    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)
    
    # --- eval-only: run evaluation + video, log to wandb, then exit ---
    if FLAGS.restore_path is not None:
        eval_info, trajs, cur_renders = evaluate(
            agent=agent,
            env=eval_env,
            config=config,
            num_eval_episodes=FLAGS.eval_episodes,
            num_video_episodes=FLAGS.video_episodes,
            video_frame_skip=FLAGS.video_frame_skip,
        )
        eval_metrics = {f'evaluation/{k}': v for k, v in eval_info.items()}
        if FLAGS.video_episodes > 0:
            eval_metrics['video'] = get_wandb_video(renders=cur_renders)
        wandb.log(eval_metrics, step=FLAGS.restore_epoch)
        print({k: v for k, v in eval_metrics.items() if k != 'video'})
        return
    # --- end eval-only ---

if __name__ == '__main__':
    app.run(main) # absl, parses flags, loads config file, and defines all flags, then runs main(args)



# # i think this is just to make the q function graphs state

# import os
# import platform

# import json
# import random
# import time

# import jax
# import numpy as np, jax.numpy as jnp
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# import tqdm
# import wandb
# from absl import app, flags
# from ml_collections import config_flags

# from agents import agents
# from envs.env_utils import make_env_and_datasets
# from utils.datasets import Dataset, ReplayBuffer
# from utils.evaluation import evaluate, flatten
# from utils.flax_utils import restore_agent, save_agent
# from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

# FLAGS = flags.FLAGS

# flags.DEFINE_string('run_group', 'Debug', 'Run group.')
# flags.DEFINE_integer('seed', 0, 'Random seed.')
# flags.DEFINE_string('env_name', 'robomimic-lift', 'Environment (dataset) name.')
# flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
# flags.DEFINE_string('restore_path', None, 'Restore path.')
# flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

# flags.DEFINE_integer('offline_steps', 200000, 'Number of offline steps.')
# flags.DEFINE_integer('online_steps', 0, 'Number of online steps.')
# flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
# flags.DEFINE_integer('log_interval', 500, 'Logging interval.')
# flags.DEFINE_integer('eval_interval', 10000, 'Evaluation interval.')
# flags.DEFINE_integer('save_interval', 20000, 'Saving interval.')

# flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
# flags.DEFINE_integer('video_episodes', 5, 'Number of video episodes for each task.')
# flags.DEFINE_integer('video_frame_skip', 1, 'Frame skip for videos.')
    
# flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
# flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')
# flags.DEFINE_integer('balanced_sampling', 0, 'Whether to use balanced sampling for online fine-tuning.')

# config_flags.DEFINE_config_file('agent', 'agents/fql.py', lock_config=False)


# def main(_):
#     # Set up logger.
#     exp_name = get_exp_name(FLAGS.seed)
#     setup_wandb(project='fql', group=FLAGS.run_group, name=exp_name)
#     FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
#     os.makedirs(FLAGS.save_dir, exist_ok=True)
#     flag_dict = get_flag_dict()
#     with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
#         json.dump(flag_dict, f)
    
#     # Make environment and datasets.
#     config = FLAGS.agent
#     env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, frame_stack=FLAGS.frame_stack)

#     # create popup window for live viewer
#     # vis_env = robomimic_utils.make_eval_env(env_name="robomimic_lift", render_mode="human")
    
#     # get rid of this assertion error to let lift task run
#     # if FLAGS.video_episodes > 0:
#     #     assert 'singletask' in FLAGS.env_name, 'Rendering is currently only supported for OGBench environments.'
#     if FLAGS.online_steps > 0:
#         assert 'visual' not in FLAGS.env_name, 'Online fine-tuning is currently not supported for visual environments.'
    
#     # Initialize agent.
#     random.seed(FLAGS.seed)
#     np.random.seed(FLAGS.seed)
#     # Set up datasets.
#     train_dataset = Dataset.create(**train_dataset)
    
#     if FLAGS.balanced_sampling:
#         # Create a separate replay buffer so that we can sample from both the training dataset and the replay buffer.
#         example_transition = {k: v[0] for k, v in train_dataset.items()}
#         replay_buffer = ReplayBuffer.create(example_transition, size=FLAGS.buffer_size)
#     else:
#         # Use the training dataset as the replay buffer.
#         train_dataset = ReplayBuffer.create_from_initial_dataset(
#             dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
#         )
#         replay_buffer = train_dataset
    
#     # Set p_aug and frame_stack.
#     for dataset in [train_dataset, val_dataset, replay_buffer]:
#         if dataset is not None:
#             dataset.p_aug = FLAGS.p_aug
#             dataset.frame_stack = FLAGS.frame_stack
#             if config['agent_name'] == 'rebrac':
#                 dataset.return_next_actions = True
    
#     # Create agent.
#     example_batch = train_dataset.sample(1)
#     agent_class = agents[config['agent_name']]
#     agent = agent_class.create(
#         FLAGS.seed,
#         example_batch['observations'],
#         example_batch['actions'],
#         config,
#     )

#     # dont comment this out or then it starts a completely fresh agent with random weights
#     if FLAGS.restore_path is not None:
#         agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)
    

#     # this bottom part is for just visualizing the training data as videos
#     # pull the full dataset arrays (ReplayBuffer/Dataset stores them as fields)
#     obs_all  = np.asarray(train_dataset['observations'])
#     act_all  = np.asarray(train_dataset['actions'])
#     term_all = np.asarray(train_dataset['terminals'])
#     # find episode boundaries: a trajectory ends where terminals == 1
#     end_idxs = np.where(term_all > 0.5)[0]
#     starts = np.concatenate([[0], end_idxs[:-1] + 1])
#     bounds = list(zip(starts, end_idxs + 1))[:5]   # first 5 trajectories
#     plt.figure(figsize=(9, 5))
#     for ei, (s, e) in enumerate(bounds):
#         obs = obs_all[s:e]
#         act = act_all[s:e]
#         # Q(s,a) for the dataset's own state-action pairs, meaned over the ensemble
#         qs = agent.network.select('critic')(obs, actions=act)   # (ensemble, T)
#         q = np.asarray(jnp.mean(qs, axis=0))                    # (T,)
#         plt.plot(q, marker='.', label=f"traj {ei} (len {e-s})")
#         print(f"traj {ei}: len={e-s}  Q first={q[0]:.2f}  last={q[-1]:.2f}")
#     plt.xlabel("step"); plt.ylabel("Q(s, a)")
#     plt.title(f"Per-step Q on first 5 training trajectories (epoch {FLAGS.restore_epoch})")
#     plt.legend()
#     plt.savefig("20_ep_q_train_traj_70.png", dpi=120)
#     print("20_ep_saved q_train_traj_70.png")
#     wandb.log({"q_train_traj": wandb.Image("20_ep_q_train_traj_70.png")}, step=FLAGS.restore_epoch)
#     return

#     # record graphs of q values per episode
#     if FLAGS.restore_path is not None:
#         b = train_dataset.sample(256)
#         print("mean Q after restore:",
#             float(jnp.mean(agent.network.select('critic')(b['observations'], actions=b['actions']))))

#         key = jax.random.PRNGKey(0)
#         episodes = []   # (q_trace, success)

#         for ep in range(5):
#             obs, info = eval_env.reset()
#             done = False
#             q_trace = []
#             success = 0
#             while not done:
#                 key, ak = jax.random.split(key)
#                 action = np.clip(np.array(agent.sample_actions(
#                     observations=obs, seed=ak, temperature=0)), -1, 1)
#                 qs = agent.network.select('critic')(obs[None], actions=action[None])
#                 q_trace.append(float(jnp.mean(qs)))
#                 obs, reward, term, trunc, info = eval_env.step(action)
#                 done = term or trunc
#                 success = max(success, int(info.get("succeed", 0)))
#             episodes.append((q_trace, success))
#             print(f"ep {ep}: len={len(q_trace)} success={success} "
#                   f"Q first={q_trace[0]:.2f} last={q_trace[-1]:.2f}")

#         # plot: successes in one color, failures in another
#         plt.figure(figsize=(8, 5))
#         for q_trace, success in episodes:
#             plt.plot(q_trace, color=("green" if success else "red"),
#                      alpha=0.7, label=("success" if success else "fail"))
#         # de-dup legend
#         h, l = plt.gca().get_legend_handles_labels()
#         plt.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys())
#         plt.xlabel("step"); plt.ylabel("Q(s, a)")
#         plt.title(f"Per-step Q values (epoch {FLAGS.restore_epoch})")
#         plt.savefig("20_ep_q_traces_70.png", dpi=120)
#         print("20_ep_saved q_traces_70.png")

#         # optional: log to wandb too
#         wandb.log({"q_traces": wandb.Image("20_ep_q_traces_70.png")}, step=FLAGS.restore_epoch)
#         return


#     # Evaluate on 20 episodes and record 5 videos
#     # python main.py --restore_path=exp/fql/Debug/sd000_20260710_025149 --restore_epoch=100000 --eval_episodes=20 --video_episodes=5
#     # Restore agent.
#     if FLAGS.restore_path is not None:
#         agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)
    
#     # --- eval-only: run evaluation + video, log to wandb, then exit ---
#     if FLAGS.restore_path is not None:
#         eval_info, trajs, cur_renders = evaluate(
#             agent=agent,
#             env=eval_env,
#             config=config,
#             num_eval_episodes=FLAGS.eval_episodes,
#             num_video_episodes=FLAGS.video_episodes,
#             video_frame_skip=FLAGS.video_frame_skip,
#         )
#         eval_metrics = {f'evaluation/{k}': v for k, v in eval_info.items()}
#         if FLAGS.video_episodes > 0:
#             eval_metrics['video'] = get_wandb_video(renders=cur_renders)
#         wandb.log(eval_metrics, step=FLAGS.restore_epoch)
#         print({k: v for k, v in eval_metrics.items() if k != 'video'})
#         return
#     # --- end eval-only ---

# if __name__ == '__main__':
#     app.run(main) # absl, parses flags, loads config file, and defines all flags, then runs main(args)
