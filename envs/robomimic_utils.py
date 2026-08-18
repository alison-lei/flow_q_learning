import pickle
import numpy as np
from collections import deque
import gymnasium as gym
from envs.active_rl_for_assembly.lift_env_cfg import LiftEnvConfig
from envs.active_rl_for_assembly.lift_env import PandaLiftGymEnv
from scipy.spatial.transform import Rotation as R
# from envs.active_rl_for_assembly.wrappers import Quat2EulerWrapper
from envs.active_rl_for_assembly import scalar_first_modified_wrappers

# just proprioception
# def lift_dataset(path: str) -> dict:
#     with open(path, "rb") as f:
#         data = pickle.load(f)
    
#     # data is a list of dictionaries

#     # for i in range (len(data)):
#     #     if (data[i]['rewards'] == 1):
#     #         print(data[i]['rewards'])

#     observations = []
#     actions = []
#     next_observations = []
#     rewards = []
#     masks = []
#     terminals = []

#     for t in data:
#         obs = t["observations"]["state"]
#         next_obs = t["next_observations"]["state"]

#         observations.append(obs.squeeze(0))
#         next_observations.append(next_obs.squeeze(0))

#         actions.append(t["actions"])
#         rewards.append(t["rewards"])
#         masks.append(t["masks"])
#         terminals.append(t["dones"])

#     dataset = {
#         "observations": np.asarray(observations, dtype=np.float32),
#         "actions": np.asarray(actions, dtype=np.float32),
#         "next_observations": np.asarray(next_observations, dtype=np.float32),
#         "rewards": np.asarray(rewards, dtype=np.float32),
#         "masks": np.asarray(masks, dtype=np.float32), # masks = 1 - terminals, stops bootstrapping (guessing the final reward based on what its guess of how much the next state is worth)
#         "terminals": np.asarray(terminals, dtype=np.float32),
#     }

#     return dataset


# image embeddings and proprioception
def lift_dataset(path: str) -> dict:
    with open(path, "rb") as f:
        data = pickle.load(f)

    obs_state, obs_side, obs_wrist, obs_emb = [], [], [], []
    next_state, next_side, next_wrist, next_emb = [], [], [], []
    actions, rewards, masks, terminals = [], [], [], []

    for t in data:
        obs_state.append(t["observations"]["state"])
        obs_side.append(t["observations"]["side_policy_256"])
        obs_wrist.append(t["observations"]["wrist_1"])
        obs_emb.append(t["embeddings"].squeeze(0))

        next_state.append(t["next_observations"]["state"])
        next_side.append(t["next_observations"]["side_policy_256"])
        next_wrist.append(t["next_observations"]["wrist_1"])
        next_emb.append(t["next_embeddings"].squeeze(0))

        actions.append(t["actions"])
        rewards.append(t["rewards"])
        masks.append(t["masks"])
        terminals.append(t["dones"])

    dataset = {
        "observations": {
            "state": np.asarray(obs_state, dtype=np.float32),        # (N, 2, 19)
            "side_policy_256": np.asarray(obs_side, dtype=np.uint8), # (N, 2, H, W, 3)
            "wrist_1": np.asarray(obs_wrist, dtype=np.uint8),
            "embeddings": np.asarray(obs_emb, dtype=np.float32),
        },
        "next_observations": {
            "state": np.asarray(next_state, dtype=np.float32),
            "side_policy_256": np.asarray(next_side, dtype=np.uint8),
            "wrist_1": np.asarray(next_wrist, dtype=np.uint8),
            "embeddings": np.asarray(next_emb, dtype=np.float32),
        },
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "masks": np.asarray(masks, dtype=np.float32),
        "terminals": np.asarray(terminals, dtype=np.float32),
    }
    return dataset

# for proprio
# PROPRIO_KEYS = ["gripper_pose", "privileged", "tcp_pose"]

# for images + proprio
PROPRIO_KEYS = ["tcp_pose", "tcp_vel", "gripper_pose", "wrist_force", "wrist_torque"]

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



class HistoryWrapper(gym.Wrapper):
    """Maintains a rolling window of the last `horizon` observations,
    stacking every key in the obs dict along a new leading axis.
    Matches data-collection behavior: at reset, the first observation
    fills the whole window (duplicated), not zero-padded.
    """
    def __init__(self, env, horizon=2):
        super().__init__(env)
        self.horizon = horizon
        self.history = deque(maxlen=horizon)

    def _stack(self):
        keys = self.history[0].keys()
        return {k: np.stack([o[k] for o in self.history], axis=0) for k in keys}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.history.clear()
        for _ in range(self.horizon):
            self.history.append(obs)          # duplicate first obs to fill window
        return self._stack(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.history.append(obs)               # oldest auto-dropped (maxlen)
        return self._stack(), reward, terminated, truncated, info


class RelativeFrameEval(scalar_first_modified_wrappers.RelativeFrame):
    """Eval-only relative-frame wrapper for FQL rollouts.

    Reuses RelativeFrame.transform_observation / transform_action and its
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
        

def make_eval_env(env_name, render_mode="rgb_array"):
    if env_name != "robomimic-lift":
        return
    env = PandaLiftGymEnv(config=LiftEnvConfig, render_mode=render_mode, privileged_obs=True)
    env = RelativeFrameEval(env, include_relative_pose=True)  # obs base->relative, action EE->base
    env = scalar_first_modified_wrappers.Quat2EulerWrapper(env)                              # tcp_pose quat->euler (6D)
    env = scalar_first_modified_wrappers.SERLObsWrapper(env, proprio_keys=PROPRIO_KEYS)      # sorted flatten -> {"state": (13,)}
    # this is just for proprio
    # env = StateArrayWrapper(env)                              # -> (13,) array for FQL
    # image + proprio
    env = HistoryWrapper(env, horizon=2)
    return env

