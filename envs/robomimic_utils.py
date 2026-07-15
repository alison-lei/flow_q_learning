import pickle
import numpy as np
import gymnasium as gym
from envs.active_rl_for_assembly.lift_env_cfg import LiftEnvConfig
from envs.active_rl_for_assembly.lift_env import PandaLiftGymEnv
from scipy.spatial.transform import Rotation as R
# from envs.active_rl_for_assembly.wrappers import Quat2EulerWrapper
from envs.active_rl_for_assembly import scalar_first_modified_wrappers
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
# class FlattenObsWrapper(gym.Wrapper):
#     def __init__(self, env):
#         super().__init__(env)  # calls initialization method of parent class
#         self.observation_space = gym.spaces.Box(
#             -np.inf, np.inf, shape=(13,), dtype=np.float32
#         )

#     def _flatten_obs(self, obs):
#         gripper = np.array([obs["state"]["gripper_pose"]])
#         print("gripper value: ", gripper)
#         privileged = obs["state"]["privileged"]

#         tcp_pose_raw = obs["state"]["tcp_pose"]
#         pos = tcp_pose_raw[:3]
#         euler = R.from_quat(tcp_pose_raw[3:]).as_euler('xyz')
#         tcp_pose = np.concatenate([pos, euler])

#         return np.concatenate([gripper, privileged, tcp_pose]).astype(np.float32)

#     # Calls the env's reset and step which is the PandaLiftGymEnv
#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         return self._flatten_obs(obs), info

#     def step(self, action):
#         obs, reward, terminated, truncated, info = self.env.step(action)
#         return self._flatten_obs(obs), reward, terminated, truncated, info

#     def render(self):
#         return self.env.render() # self.env is PandaLiftGymEnv

PROPRIO_KEYS = ["gripper_pose", "privileged", "tcp_pose"]

class StateArrayWrapper(gym.Wrapper):
    """FQL consumes a flat (13,) array; SERLObsWrapper emits {'state': array}."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space["state"]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs["state"], info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs["state"], reward, terminated, truncated, info

    def render(self):
        return self.env.render()    


class RelativeFrameEval(scalar_first_modified_wrappers.RelativeFrame):
    """Eval-only relative-frame wrapper for FQL rollouts.

    Reuses RelativeFrameRG.transform_observation / transform_action and its
    reset-pose (T_r_o_inv) + adjoint bookkeeping VERBATIM, so eval obs and
    actions live in the exact frame the dataset was collected in. Only the
    interface changes: FQL emits a plain (7,) array, and there is no
    spacemouse / keyboard / expert_acts path at eval. reset() is inherited
    unchanged (it already sets adjoint + T_r_o_inv from the reset pose).
    """

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        # policy delta is body/EE-frame -> base frame, same adjoint the
        # collection pipeline applied before PandaLiftGymEnv.step
        base_action = self.transform_action(action)

        obs, reward, done, truncated, info = self.env.step(base_action)

        # refresh adjoint from the NEW base-frame tcp_pose, then map obs into
        # the reset-relative frame -- identical order to RelativeFrameRG.step
        self.adjoint_matrix = scalar_first_modified_wrappers.construct_adjoint_matrix(obs["state"]["tcp_pose"])
        transformed_obs = self.transform_observation(obs)
        return transformed_obs, reward, done, truncated, info


# def make_eval_env(env_name, render_mode="rgb_array"):
#     if env_name != "robomimic-lift":
#         return
#     env = PandaLiftGymEnv(
#         config=LiftEnvConfig,
#         render_mode=render_mode,
#         privileged_obs=True,
#     )
#     return FlattenObsWrapper(env)


def make_eval_env(env_name, render_mode="rgb_array"):
    if env_name != "robomimic-lift":
        return
    env = PandaLiftGymEnv(config=LiftEnvConfig, render_mode=render_mode, privileged_obs=True)
    env = RelativeFrameEval(env, include_relative_pose=True)  # obs base->relative, action EE->base
    env = scalar_first_modified_wrappers.Quat2EulerWrapper(env)                              # tcp_pose quat->euler (6D)
    env = scalar_first_modified_wrappers.SERLObsWrapper(env, proprio_keys=PROPRIO_KEYS)      # sorted flatten -> {"state": (13,)}
    env = StateArrayWrapper(env)                              # -> (13,) array for FQL
    return env

