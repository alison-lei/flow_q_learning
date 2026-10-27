# TEMP FILE TO TEST add_embeddings

import robosuite
import numpy as np
import pickle
import itertools
from utils.datasets import Dataset
from octo.model.octo_model import OctoModel
from scipy.spatial.transform import Rotation as R
from envs.robomimic_utils import make_eval_env


with open('robomimic_datasets/final_20_demos_scripted_2026-08-20_12-58-12-186961.pkl', 'rb') as f:
    data = pickle.load(f)

print(data[0]) # see that left and right are gone from infos

episode_lengths = []
counter = 0
# octo_model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small")
# tasks = octo_model.create_tasks(texts=["pick up the red block"])

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


# training_transition = data[0]
# stored = training_transition["embeddings"]
# reconstructed = add_embeddings(training_transition["observations"])["embeddings"]
# print(np.allclose(stored, reconstructed, atol=1))

# info = training_transition["infos"]["original_state_obs"]
# state = training_transition["next_observations"]["state"][-1]   # last window frame, (19,)

# for name, val in sorted(info.items()):
#     val = np.atleast_1d(val)
#     print(f"{name}: dim={val.shape}, value={val[:3]}...")

# print("state:", state)

print(len(data))
first_transition = data[0]   # first step of a trajectory
print(first_transition)

for i in range(446):
    counter += 1
    if data[i]["infos"]["succeed"] == 1:
        episode_lengths.append(counter)
        counter = 0

print(episode_lengths)

with open('robomimic_datasets/lift_20_demos_scripted_2026-05-11_15-23-51-466619_CORL.pkl', 'rb') as f:
    data_old = pickle.load(f)

with open('robomimic_datasets/final_20_demos_scripted_2026-08-20_12-58-12-186961.pkl', 'rb') as f:
    data_new = pickle.load(f)

def summarize(data, name):
    print(f"--- {name} ---")
    print("num transitions:", len(data))
    print("trajectory count (dones=True):", sum(t["dones"] for t in data))
    print("reward mean/std:", np.mean([t["rewards"] for t in data]), np.std([t["rewards"] for t in data]))
    gripper = np.array([t["actions"][-1] for t in data])
    print("gripper-closing fraction (>0):", (gripper > 0).mean())
    emb = np.stack([t["embeddings"].squeeze(0) for t in data])
    print("embedding mean/std:", emb.mean(), emb.std())
    img = np.stack([t["observations"]["side_policy_256"][-1] for t in data])
    print("image brightness mean:", img.mean())

summarize(data_new, "your data")
summarize(data_old, "original data")