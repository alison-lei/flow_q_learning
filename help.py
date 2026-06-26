import sys
sys.path.insert(0, '/home/alisonlei/flow_q_learning/fql')

import numpy as np
from utils.datasets import Dataset
from envs.robomimic_utils import lift_dataset
from envs.env_utils import make_env_and_datasets


raw = lift_dataset('robomimic_datasets/lift_300_demos_scripted_flow_2026-06-10_14-29-49-254753.pkl')
print("raw observations shape:", raw['observations'].shape)
print("raw actions shape:", raw['actions'].shape)

train_dataset = Dataset.create(**raw)
example_batch = train_dataset.sample(1)
print("example_batch observations shape:", example_batch['observations'].shape)
print("example_batch actions shape:", example_batch['actions'].shape)

_, eval_env, _, _ = make_env_and_datasets("robomimic-lift")
obs, _ = eval_env.reset()
print("eval env obs shape:", obs.shape)
