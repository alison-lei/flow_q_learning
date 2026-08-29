import gymnasium as gym
from gymnasium.spaces import flatten_space, flatten
from gymnasium import Env, spaces
from scipy.spatial.transform import Rotation as R
import numpy as np
import copy

from gymnasium.spaces import Box
# from franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert

from tensorflow_probability.substrates.jax.experimental.mcmc.sample_sequential_monte_carlo import PRINT_DEBUG

SCALAR_FIRST = False #if True quat = [w, x, y, z]
# if False quat = [x, y, z, w]

# based on "external/hil-serl/examples/experiments/usb_pickup_insertion/wrapper.py"
# obs["state"][0,0] because keys are sorted in lexicographical order (gripper_pose comes first)
class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = self.env.config.GRIPPER_PENALTY
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, 0]
        return obs, info

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]

        if (action[-1] < -0.5 and self.last_gripper_pos > 0.9) or (
            action[-1] > 0.5 and self.last_gripper_pos < 0.9
        ):
            info["grasp_penalty"] = self.penalty
        else:
            info["grasp_penalty"] = 0.0

        self.last_gripper_pos = observation["state"][0, 0]
        return observation, reward, terminated, truncated, info


# based on "external/hil-serl/serl_robot_infra/franka_env/envs/wrappers.py"
class Quat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        return observation

# used for nut_assembly_square
class RelativeFrame(gym.Wrapper):
    """
    Local copy of franka_env.envs.relative_env.RelativeFrame (the non-RG path) kept
    in-repo so the reference_pose option can be maintained here instead of editing
    the conrft submodule.

    Transforms the observation and action to be expressed in the end-effector frame.
    Optionally re-expresses tcp_pose relative to either the per-reset pose
    (include_relative_pose=True) or a fixed base-frame pose (reference_pose).
    """

    def __init__(self, env: Env, include_relative_pose=True, reference_pose=None):
        super().__init__(env)
        self.adjoint_matrix = np.zeros((6, 6))

        self.include_relative_pose = include_relative_pose
        # Optional FIXED base-frame reference pose (x, y, z, qx, qy, qz, qw).
        # When set, tcp_pose is expressed relative to this constant pose instead
        # of the per-reset pose, and it is NOT recomputed on reset. Takes
        # precedence over include_relative_pose's per-reset capture below.
        self.reference_pose = (
            None if reference_pose is None else np.asarray(reference_pose, dtype=float)
        )
        # tcp_pose is re-expressed in a relative frame if either mode is active.
        self.use_relative_pose = self.include_relative_pose or (self.reference_pose is not None)
        if self.use_relative_pose:
            # Homogeneous transformation matrix from the reference frame to base frame
            self.T_r_o_inv = np.zeros((4, 4))
            if self.reference_pose is not None:
                # Built once from the fixed reference pose; never updated at reset.
                self.T_r_o_inv = np.linalg.inv(
                    construct_homogeneous_matrix(self.reference_pose)
                )

    def step(self, action: np.ndarray):
        # action is assumed to be (x, y, z, rx, ry, rz, gripper)
        # Transform action from end-effector frame to base frame
        transformed_action = self.transform_action(action)
        obs, reward, done, truncated, info = self.env.step(transformed_action)
        info['original_state_obs'] = copy.deepcopy(obs['state'])

        # this is to convert the spacemouse intervention action
        if "intervene_action" in info:
            info["intervene_action"] = self.transform_action_inv(info["intervene_action"])

        # Update adjoint matrix
        self.adjoint_matrix = construct_adjoint_matrix(obs["state"]["tcp_pose"])

        # Transform observation to spatial frame
        transformed_obs = self.transform_observation(obs)
        return transformed_obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['original_state_obs'] = copy.deepcopy(obs['state'])

        # Update adjoint matrix
        self.adjoint_matrix = construct_adjoint_matrix(obs["state"]["tcp_pose"])
        if self.include_relative_pose and self.reference_pose is None:
            # Update transformation matrix from the reset pose's relative frame to base frame
            self.T_r_o_inv = np.linalg.inv(
                construct_homogeneous_matrix(obs["state"]["tcp_pose"])
            )

        # Transform observation to spatial frame
        return self.transform_observation(obs), info

    def transform_observation(self, obs):
        """
        Transform observations from spatial(base) frame into body(end-effector) frame
        using the adjoint matrix
        """
        adjoint_inv = np.linalg.inv(self.adjoint_matrix)
        obs["state"]["tcp_vel"] = adjoint_inv @ obs["state"]["tcp_vel"]

        if self.use_relative_pose:
            T_b_o = construct_homogeneous_matrix(obs["state"]["tcp_pose"])
            T_b_r = self.T_r_o_inv @ T_b_o

            # Reconstruct transformed tcp_pose vector
            p_b_r = T_b_r[:3, 3]
            theta_b_r = R.from_matrix(T_b_r[:3, :3]).as_quat()
            obs["state"]["tcp_pose"] = np.concatenate((p_b_r, theta_b_r))

        return obs

    def transform_action(self, action: np.ndarray):
        """
        Transform action from body(end-effector) frame into into spatial(base) frame
        using the adjoint matrix.
        """
        action = np.array(action)  # in case action is a jax read-only array
        action[:6] = self.adjoint_matrix @ action[:6]
        return action

    def transform_action_inv(self, action: np.ndarray):
        """
        Transform action from spatial(base) frame into body(end-effector) frame
        using the adjoint matrix.
        """
        action = np.array(action)
        action[:6] = np.linalg.inv(self.adjoint_matrix) @ action[:6]
        return action



# # used for lift
# # based on "external/hil-serl/serl_robot_infra/franka_env/envs/relative_env.py"
# class RelativeFrame(gym.Wrapper):
#     """
#     This wrapper transforms the observation and action to be expressed in the end-effector frame.
#     Optionally, it can transform the tcp_pose into a relative frame defined as the reset pose.

#     This wrapper is expected to be used on top of the base Franka environment, which has the following
#     observation space:
#     {
#         "state": spaces.Dict(
#             {
#                 "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,)), # xyz + quat
#                 ......
#             }
#         ),
#         ......
#     }, and at least 6 DoF action space with (x, y, z, rx, ry, rz, ...)
#     """

#     def __init__(self, env: Env, include_relative_pose=True):
#         super().__init__(env)
#         self.adjoint_matrix = np.zeros((6, 6))

#         self.include_relative_pose = include_relative_pose
#         if self.include_relative_pose:
#             # Homogeneous transformation matrix from reset pose's relative frame to base frame
#             self.T_r_o_inv = np.zeros((4, 4))

#     def step(self, action: np.ndarray):
#         # action is assumed to be (x, y, z, rx, ry, rz, gripper)
#         # Transform action from end-effector frame to base frame
#         transformed_action = self.transform_action(action)
#         obs, reward, done, truncated, info = self.env.step(transformed_action)
#         info['original_state_obs'] = copy.deepcopy(obs['state'])

#         # this is to convert the spacemouse intervention action

#         if "intervene_action" in info:
#             info["intervene_action"] = self.transform_action_inv(info["intervene_action"])

#         # Update adjoint matrix
#         self.adjoint_matrix = construct_adjoint_matrix(obs["state"]["tcp_pose"])

#         # Transform observation to spatial frame
#         transformed_obs = self.transform_observation(obs)
#         return transformed_obs, reward, done, truncated, info

#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         info['original_state_obs'] = copy.deepcopy(obs['state'])

#         # Update adjoint matrix
#         self.adjoint_matrix = construct_adjoint_matrix(obs["state"]["tcp_pose"])
#         if self.include_relative_pose:
#             # Update transformation matrix from the reset pose's relative frame to base frame
#             self.T_r_o_inv = np.linalg.inv(
#                 construct_homogeneous_matrix(obs["state"]["tcp_pose"])
#             )

#         # Transform observation to spatial frame
#         return self.transform_observation(obs), info

#     def transform_observation(self, obs):
#         """
#         Transform observations from spatial(base) frame into body(end-effector) frame
#         using the adjoint matrix
#         """
#         adjoint_inv = np.linalg.inv(self.adjoint_matrix)
#         obs["state"]["tcp_vel"] = adjoint_inv @ obs["state"]["tcp_vel"]

#         if self.include_relative_pose:
#             T_b_o = construct_homogeneous_matrix(obs["state"]["tcp_pose"])
#             T_b_r = self.T_r_o_inv @ T_b_o

#             # Reconstruct transformed tcp_pose vector
#             p_b_r = T_b_r[:3, 3]
#             theta_b_r = R.from_matrix(T_b_r[:3, :3]).as_quat()
#             obs["state"]["tcp_pose"] = np.concatenate((p_b_r, theta_b_r))

#         return obs

#     def transform_action(self, action: np.ndarray):
#         """
#         Transform action from body(end-effector) frame into into spatial(base) frame
#         using the adjoint matrix.
#         """
#         action = np.array(action)  # in case action is a jax read-only array
#         action[:6] = self.adjoint_matrix @ action[:6]
#         return action

#     def transform_action_inv(self, action: np.ndarray):
#         """
#         Transform action from spatial(base) frame into body(end-effector) frame
#         using the adjoint matrix.
#         """
#         action = np.array(action)
#         action[:6] = np.linalg.inv(self.adjoint_matrix) @ action[:6]
#         return action


# based on "/external/hil-serl/serl_launcher/serl_launcher/wrappers/serl_obs_wrappers.py"
class SERLObsWrapper(gym.ObservationWrapper):
    """
    This observation wrapper treat the observation space as a dictionary
    of a flattened state space and the images.
    """

    def __init__(self, env, proprio_keys=None):
        super().__init__(env)
        self.proprio_keys = proprio_keys
        if self.proprio_keys is None:
            self.proprio_keys = list(self.env.observation_space["state"].keys())
        self.proprio_space = gym.spaces.Dict(
            {key: self.env.observation_space["state"][key] for key in self.proprio_keys}
        )

        if "images" in self.env.observation_space.keys():
            self.observation_space = gym.spaces.Dict(
                {
                    "state": flatten_space(self.proprio_space),
                    **(self.env.observation_space["images"]),
                }
            )
        else:
            self.observation_space = gym.spaces.Dict(
                {
                    "state": flatten_space(self.proprio_space)
                }
            )

    def observation(self, obs):
        if "images" in obs.keys():
            obs = {
                "state": flatten(
                    self.proprio_space,
                    {key: obs["state"][key] for key in self.proprio_keys},
                ),
                **(obs["images"]),
            }

        else:
            obs = {
                "state": flatten(
                    self.proprio_space,
                    {key: obs["state"][key] for key in self.proprio_keys},
                )
            }
        return obs

    def reset(self, **kwargs):
        obs, info =  self.env.reset(**kwargs)
        return self.observation(obs), info



# use action dict instead of just action??? -- for RG
#------------------------------------------------------------------------------------------------
# based on "external/hil-serl/serl_robot_infra/franka_env/envs/wrappers.py"
class GripperCloseEnvRG(gym.ActionWrapper):
    """
    Use this wrapper to task that requires the gripper to be closed
    """

    def __init__(self, env):
        super().__init__(env)
        ub = self.env.action_space
        assert ub.shape == (7,)
        self.action_space = Box(ub.low[:6], ub.high[:6])

    def action(self, action: np.ndarray) -> np.ndarray:
        new_action = np.zeros((7,), dtype=np.float32)
        new_action[:6] = action.copy()
        return new_action

    def step(self, action_dict):
        new_action_dict = action_dict.copy() #TODO: deep?
        new_action_dict["action"] = self.action(new_action_dict["action"])
        obs, rew, done, truncated, info = self.env.step(new_action_dict)
        if "intervene_action" in info:
            info["intervene_action"] = info["intervene_action"][:6]
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

import time
# based on "external/hil-serl/serl_robot_infra/franka_env/envs/wrappers.py"
class SpacemouseInterventionRG(gym.ActionWrapper):
    def __init__(self, env, action_indices=None):
        super().__init__(env)

        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

        self.expert = SpaceMouseExpert()
        self.left, self.right = False, False
        self.action_indices = action_indices

    def action(self, action_dict: dict) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        if not action_dict["expert_acts"]:
            return action_dict, False


        while True:
            expert_a, buttons = self.expert.get_action()
            self.left, self.right = tuple(buttons[:2])
            intervened = False

            if np.linalg.norm(expert_a) > 0.001:
                intervened = True

            if self.gripper_enabled:
                if self.left:  # close gripper
                    gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                    intervened = True
                elif self.right:  # open gripper
                    gripper_action = np.random.uniform(0.9, 1, size=(1,))
                    intervened = True
                else:
                    gripper_action = np.zeros((1,))
                expert_a = np.concatenate((expert_a, gripper_action), axis=0)

            if self.action_indices is not None:
                filtered_expert_a = np.zeros_like(expert_a)
                filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
                expert_a = filtered_expert_a

            if intervened:
                action_dict["action"] = expert_a
                return action_dict, True
            #time.sleep(0.01) #TODO
            time.sleep(0.001)
            # print("wait for an expert action")

    def step(self, action_dict):
        new_action_dict = action_dict.copy()
        new_action_dict, replaced = self.action(new_action_dict)

        obs, rew, done, truncated, info = self.env.step(new_action_dict)
        if replaced:
            info["intervene_action"] = new_action_dict["action"]
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info

class RLExpertInterventionRG(gym.ActionWrapper): #TODO: test it
    def __init__(self, env):
        super().__init__(env)

        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

    def action(self, action_dict: dict):
        if action_dict["expert_acts"]:
            return action_dict, True
        return action_dict, False

    def step(self, action_dict):
        action_dict, replaced = self.action(action_dict)

        obs, rew, done, truncated, info = self.env.step(action_dict)
        if replaced:
            info["intervene_action"] = action_dict["action"]
        return obs, rew, done, truncated, info

# based on "external/hil-serl/serl_robot_infra/franka_env/envs/relative_env.py"
class RelativeFrameRG(gym.Wrapper):
    """
    This wrapper transforms the observation and action to be expressed in the end-effector frame.
    Optionally, it can transform the tcp_pose into a relative frame defined as the reset pose.

    This wrapper is expected to be used on top of the base Franka environment, which has the following
    observation space:
    {
        "state": spaces.Dict(
            {
                "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,)), # xyz + quat
                ......
            }
        ),
        ......
    }, and at least 6 DoF action space with (x, y, z, rx, ry, rz, ...)
    """

    def __init__(self, env: Env, include_relative_pose=True):
        super().__init__(env)
        self.adjoint_matrix = np.zeros((6, 6))

        self.include_relative_pose = include_relative_pose
        if self.include_relative_pose:
            # Homogeneous transformation matrix from reset pose's relative frame to base frame
            self.T_r_o_inv = np.zeros((4, 4))

    def step(self, action_dict: dict):
        # action is assumed to be (x, y, z, rx, ry, rz, gripper)
        # Transform action from end-effector frame to base frame
        transformed_action_dict = action_dict.copy()
        transformed_action_dict["action"] = self.transform_action(transformed_action_dict["action"])
        obs, reward, done, truncated, info = self.env.step(transformed_action_dict)
        info['original_state_obs'] = copy.deepcopy(obs['state'])

        # this is to convert the spacemouse intervention action
        if "intervene_action" in info:
            info["intervene_action"] = self.transform_action_inv(info["intervene_action"])

        # Update adjoint matrix
        self.adjoint_matrix = construct_adjoint_matrix(obs["state"]["tcp_pose"])

        # Transform observation to spatial frame
        transformed_obs = self.transform_observation(obs)
        return transformed_obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['original_state_obs'] = copy.deepcopy(obs['state'])

        # Update adjoint matrix
        self.adjoint_matrix = construct_adjoint_matrix(obs["state"]["tcp_pose"])
        if self.include_relative_pose:
            # Update transformation matrix from the reset pose's relative frame to base frame
            self.T_r_o_inv = np.linalg.inv(
                construct_homogeneous_matrix(obs["state"]["tcp_pose"])
            )

        # Transform observation to spatial frame
        return self.transform_observation(obs), info

    def transform_observation(self, obs):
        """
        Transform observations from spatial(base) frame into body(end-effector) frame
        using the adjoint matrix
        """
        adjoint_inv = np.linalg.inv(self.adjoint_matrix)
        obs["state"]["tcp_vel"] = adjoint_inv @ obs["state"]["tcp_vel"]

        if self.include_relative_pose:
            T_b_o = construct_homogeneous_matrix(obs["state"]["tcp_pose"])
            T_b_r = self.T_r_o_inv @ T_b_o

            # Reconstruct transformed tcp_pose vector
            p_b_r = T_b_r[:3, 3]
            theta_b_r = R.from_matrix(T_b_r[:3, :3]).as_quat()
            obs["state"]["tcp_pose"] = np.concatenate((p_b_r, theta_b_r))

        return obs

    def transform_action(self, action: np.ndarray):
        """
        Transform action from body(end-effector) frame into into spatial(base) frame
        using the adjoint matrix.
        """
        action = np.array(action)  # in case action is a jax read-only array
        action[:6] = self.adjoint_matrix @ action[:6]
        return action

    def transform_action_inv(self, action: np.ndarray):
        """
        Transform action from spatial(base) frame into body(end-effector) frame
        using the adjoint matrix.
        """
        action = np.array(action)
        action[:6] = np.linalg.inv(self.adjoint_matrix) @ action[:6]
        return action

# based on "external/serl/serl_robot_infra/franka_env/utils/transformations.py"
def construct_adjoint_matrix(tcp_pose):
    """
    Construct the adjoint matrix for a spatial velocity vector
    :args: tcp_pose: (x, y, z, qx, qy, qz, qw)

    if SCALAR_FIRST=True:
    :args: tcp_pose: (x, y, z, qw, qx, qy, qz)
    """
    rotation = R.from_quat(tcp_pose[3:]).as_matrix()
    translation = np.array(tcp_pose[:3])
    skew_matrix = np.array(
        [
            [0, -translation[2], translation[1]],
            [translation[2], 0, -translation[0]],
            [-translation[1], translation[0], 0],
        ]
    )
    adjoint_matrix = np.zeros((6, 6))
    adjoint_matrix[:3, :3] = rotation
    adjoint_matrix[3:, 3:] = rotation
    adjoint_matrix[3:, :3] = skew_matrix @ rotation
    return adjoint_matrix


def construct_homogeneous_matrix(tcp_pose):
    """
    Construct the homogeneous transformation matrix from given pose.
    args: tcp_pose: (x, y, z, qx, qy, qz, qw)

    if SCALAR_FIRST=True:
    :args: tcp_pose: (x, y, z, qw, qx, qy, qz)
    """
    rotation = R.from_quat(tcp_pose[3:]).as_matrix()
    translation = np.array(tcp_pose[:3])
    T = np.zeros((4, 4))
    T[:3, :3] = rotation
    T[:3, 3] = translation
    T[3, 3] = 1
    return T
