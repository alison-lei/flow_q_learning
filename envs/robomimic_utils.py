import pickle
import numpy as np
from lift_env import PandaLiftGymEnv

def lift_dataset(path: str) -> dict:
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    # data is a list of dictionaries

    # for i in range (len(data)):
    #     if (data[i]['rewards'] == 1):
    #         print(data[i]['rewards'])

    observations = []
    actions = []
    next_observations = []
    rewards = []
    masks = []
    terminals = []

    for t in data:
        obs = t["observations"]["state"]
        next_obs = t["next_observations"]["state"]

        observations.append(obs.squeeze(0))
        next_observations.append(next_obs.squeeze(0))

        actions.append(t["actions"])
        rewards.append(t["rewards"])
        masks.append(t["masks"])
        terminals.append(t["dones"])

    dataset = {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "next_observations": np.asarray(next_observations, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "masks": np.asarray(masks, dtype=np.float32), # masks = 1 terminals, stops bootstrapping (guessing the final reward based on what its guess of how much the next state is worth)
        "terminals": np.asarray(terminals, dtype=np.float32),
    }

    return dataset

def make_eval_env(env_name):
    env = PandaLiftGymEnv(privileged_obs = True)
    env

