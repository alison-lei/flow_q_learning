import os
import platform

import json
import random
import time

import jax
import numpy as np
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
flags.DEFINE_string('env_name', 'robomimic-lift', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('offline_steps', 100000, 'Number of offline steps.')
flags.DEFINE_integer('online_steps', 0, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 20000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 20000, 'Saving interval.')

flags.DEFINE_integer('eval_episodes', 100, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')
flags.DEFINE_integer('balanced_sampling', 0, 'Whether to use balanced sampling for online fine-tuning.')

config_flags.DEFINE_config_file('agent', 'agents/fql.py', lock_config=False)


def temp_main(_):
    # Set up logger.
    # exp_name = get_exp_name(FLAGS.seed)
    # setup_wandb(project='fql', group=FLAGS.run_group, name=exp_name)

    # FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    # os.makedirs(FLAGS.save_dir, exist_ok=True)
    # flag_dict = get_flag_dict()
    # with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
    #     json.dump(flag_dict, f)

    # Make environment and datasets.
    config = FLAGS.agent
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, frame_stack=FLAGS.frame_stack)
    if FLAGS.video_episodes > 0:
        assert 'singletask' in FLAGS.env_name, 'Rendering is currently only supported for OGBench environments.'
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
    print(agent.config["encoder"])


def print_states():
    from envs import robomimic_utils
    train_dataset = robomimic_utils.lift_dataset("lift_5_demos_scripted_flow_2026-06-05_01-10-42-709394.pkl")
    # train_dataset = robomimic_utils.lift_dataset("lift_300_demos_scripted_flow_2026-06-10_14-29-49-254753.pkl")

    print(type(train_dataset))
    # for key, value in train_dataset:
    #     if key == "dones":
    print(1 if (train_dataset['terminals'].any()) else 0)
    done_array = train_dataset["terminals"]
    traj_steps = []
    counter = 0
    for i in range(len(done_array)):
        counter += 1
        if done_array[i] == 1:
            traj_steps.append(counter)
            counter = 0

    sum = 0.0
    for e in traj_steps:
        sum += e
    average = sum / len(traj_steps)

    print(average)

    # print(train_dataset)

    # data is a list of dictionaries

    # print(type(data))
    # print(len(data))
    # print(data[0].keys())
    # print(data[0])
    # print(data[0]["observations"].keys())

    # for i in range (len(data)):
    #     # if (data[i]['dones'] == True):
    #     print(data[i]['dones'])


if __name__ == '__main__':
    app.run(temp_main)
