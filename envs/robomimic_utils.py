import pickle
import numpy as np
import gymnasium as gym
from envs.active_rl_for_assembly.lift_env_cfg import LiftEnvConfig
from envs.active_rl_for_assembly.lift_env import PandaLiftGymEnv
from scipy.spatial.transform import Rotation as R

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
        "masks": np.asarray(masks, dtype=np.float32), # masks = 1 - terminals, stops bootstrapping (guessing the final reward based on what its guess of how much the next state is worth)
        "terminals": np.asarray(terminals, dtype=np.float32),
    }

    return dataset

# purpose of this class is only to flatten dict output
# Use gym.Wrapper instead of gym.ObservationWrapper
# to avoid recursion bug in gymnasium 0.29.1
class FlattenObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)  # calls initialization method of parent class
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(13,), dtype=np.float32
        )

    def _flatten_obs(self, obs):
        gripper = np.array([obs["state"]["gripper_pose"]])
        privileged = obs["state"]["privileged"]

        tcp_pose_raw = obs["state"]["tcp_pose"]
        pos = tcp_pose_raw[:3]
        euler = R.from_quat(tcp_pose_raw[3:]).as_euler('xyz')
        tcp_pose = np.concatenate([pos, euler])

        return np.concatenate([gripper, privileged, tcp_pose]).astype(np.float32)

    # Calls the env's reset and step which is the PandaLiftGymEnv
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flatten_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._flatten_obs(obs), reward, terminated, truncated, info

    def render(self):
        return self.env.render()


def make_eval_env(env_name, render_mode="rgb_array"):
    if env_name != "robomimic-lift":
        return
    env = PandaLiftGymEnv(
        config=LiftEnvConfig,
        render_mode=render_mode,
        privileged_obs=True,
    )
    return FlattenObsWrapper(env)



# # ObservationWrapper is a special gymnasium class that automatically
# # intercepts every reset() and step() call and passes the observation
# # through your observation() method before returning it
# class FlattenObsWrapper(gym.ObservationWrapper):
#     def __init__(self, env):
#         self.observation_space = gym.spaces.Box(
#             -np.inf, np.inf, shape=(13,), dtype=np.float32
#         )

#     # This is the method gym.ObservationWrapper calls automatically
#     # every time an observation comes out of PandaLiftGymEnv.
#     # You override it to define how to convert the dict into a flat array.
#     def observation(self, obs):
#         return np.concatenate([
#             np.array([obs["state"]["gripper_pose"]]),
#             obs["state"]["privileged"],
#             obs["state"]["tcp_vel"]
#         ]).astype(np.float32)


# def make_eval_env(env_name, render_mode="rgb_array"):
#     if env_name != "robomimic-lift":
#         return
#     env = PandaLiftGymEnv(config=LiftEnvConfig, render_mode=render_mode, privileged_obs=True)
    
#     return FlattenObsWrapper(env)

