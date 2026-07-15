import gymnasium as gym
from gymnasium.spaces import flatten_space, flatten
from gymnasium import Env, spaces
from scipy.spatial.transform import Rotation as R
import numpy as np
import copy
import time

from gymnasium.spaces import Box
# from franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert
from franka_env.utils.transformations import construct_adjoint_matrix, construct_homogeneous_matrix


# to add ee/base frame support, we need to modify the SpaceMouseExpert class
import multiprocessing
import numpy as np
from franka_env.spacemouse import pyspacemouse
from typing import Tuple

# from serl_launcher.wrappers.chunking import ChunkingWrapper
from franka_env.envs.wrappers import Quat2EulerWrapper

class SpaceMouseExpert:
    """
    This class provides an interface to the SpaceMouse.
    It continuously reads the SpaceMouse state and provides
    a "get_action" method to get the latest action and button state.
    """

    def __init__(self, frame="base"):
        pyspacemouse.open()
        self.frame = frame

        # Manager to handle shared state between processes
        self.manager = multiprocessing.Manager()
        self.latest_data = self.manager.dict()
        self.latest_data["action"] = [0.0] * 6  # Using lists for compatibility
        self.latest_data["buttons"] = [0, 0]

        # Start a process to continuously read the SpaceMouse state
        self.process = multiprocessing.Process(target=self._read_spacemouse)
        self.process.daemon = True
        self.process.start()

    def _read_spacemouse(self):
        while True:
            state = pyspacemouse.read_all()
            action = [0.0] * 6
            buttons = [0, 0]
            # print_green(f"TEST STATE LENGTH: {len(state)}")
            if len(state) == 2:
                action = [
                    -state[0].y, state[0].x, state[0].z,
                    -state[0].roll, -state[0].pitch, -state[0].yaw,
                    -state[1].y, state[1].x, state[1].z,
                    -state[1].roll, -state[1].pitch, -state[1].yaw
                ]
                buttons = state[0].buttons + state[1].buttons
            elif len(state) == 1:
                if self.frame == "ee":
                    # pass #TODO: implement ee frame support
                    action = [
                        state[0].x, -state[0].y, -state[0].z,
                        -state[0].pitch, -state[0].roll, state[0].yaw #TODO: roll and pitch
                    ]
                    buttons = state[0].buttons
                elif self.frame == "base":
                    action = [
                        -state[0].y, state[0].x, state[0].z,
                        -state[0].roll, -state[0].pitch, -state[0].yaw
                    ]
                    buttons = state[0].buttons
                    # print(buttons)

            # Update the shared state
            self.latest_data["action"] = action
            self.latest_data["buttons"] = buttons

    def get_action(self) -> Tuple[np.ndarray, list]:
        """Returns the latest action and button state of the SpaceMouse."""
        action = self.latest_data["action"]
        buttons = self.latest_data["buttons"]
        return np.array(action), buttons
    
    def close(self):
        # pyspacemouse.close()
        self.process.terminate()



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

class KeyboardExpert:
    """
    Keyboard teleop source based on robosuite viewer callbacks (OpenCV waitKey path).
    """

    def __init__(self, env, linear_scale=1.0, angular_scale=1.0):
        self.linear_scale = linear_scale
        self.angular_scale = angular_scale
        self._rs_env = self._find_robosuite_env(env)
        self._rs_viewer_id = None
        self._last_key_event = {}
        self._gripper_closed = False
        self._last_gripper_toggle_ts = 0.0
        self._register_viewer_callbacks_if_needed()
        print("Keyboard teleop active (viewer callback).")
        print("  Move: arrows + '.' / ';'  OR  W/S, A/D, Q/E")
        print("  Rotate: o/p, y/h, e/r      OR  I/K, J/L, U/O")
        print("  Gripper: space (toggle) OR Z(close)/X(open)")

    @staticmethod
    def _find_robosuite_env(env):
        cur = env
        for _ in range(32):
            if hasattr(cur, "_env"):
                return getattr(cur, "_env")
            if hasattr(cur, "env"):
                cur = cur.env
            else:
                break
        return None

    def _register_viewer_callbacks_if_needed(self):
        if self._rs_env is None:
            return
        viewer = getattr(self._rs_env, "viewer", None)
        if viewer is None:
            return
        if id(viewer) != self._rs_viewer_id:
            viewer.add_keypress_callback(self._on_keypress)
            self._rs_viewer_id = id(viewer)

    def _on_keypress(self, key):
        self._last_key_event[int(key)] = time.time()

    def _active(self, keycodes, now, hold_s=0.25):
        for code in keycodes:
            ts = self._last_key_event.get(int(code), -1e9)
            if (now - ts) <= hold_s:
                return True
        return False

    def get_action(self):
        self._register_viewer_callbacks_if_needed()
        now = time.time()
        action = np.zeros(6, dtype=np.float32)

        # Translation x/y/z
        if self._active([82, 2490368], now) or self._active([ord("w"), ord("W")], now):   # up / w
            action[0] -= self.linear_scale
        if self._active([84, 2621440], now) or self._active([ord("s"), ord("S")], now):   # down / s
            action[0] += self.linear_scale
        if self._active([81, 2424832], now) or self._active([ord("a"), ord("A")], now):   # left / a
            action[1] -= self.linear_scale
        if self._active([83, 2555904], now) or self._active([ord("d"), ord("D")], now):   # right / d
            action[1] += self.linear_scale
        if self._active([ord("."), ord(";")], now):
            # In robosuite keyboard, '.' is z-, ';' is z+
            if self._active([ord(".")], now):
                action[2] -= self.linear_scale
            if self._active([ord(";")], now):
                action[2] += self.linear_scale
        if self._active([ord("q"), ord("Q")], now):
            action[2] += self.linear_scale
        if self._active([ord("e"), ord("E")], now):
            action[2] -= self.linear_scale

        # Rotation rx/ry/rz (support both robosuite and fallback scheme)
        if self._active([ord("i"), ord("I")], now):
            action[3] += self.angular_scale
        if self._active([ord("k"), ord("K")], now):
            action[3] -= self.angular_scale
        if self._active([ord("j"), ord("J")], now):
            action[4] += self.angular_scale
        if self._active([ord("l"), ord("L")], now):
            action[4] -= self.angular_scale
        if self._active([ord("u"), ord("U")], now):
            action[5] += self.angular_scale
        if self._active([ord("o"), ord("O")], now):
            action[5] -= self.angular_scale

        if self._active([ord("e"), ord("E")], now):
            action[3] += self.angular_scale
        if self._active([ord("r"), ord("R")], now):
            action[3] -= self.angular_scale
        if self._active([ord("y"), ord("Y")], now):
            action[4] += self.angular_scale
        if self._active([ord("h"), ord("H")], now):
            action[4] -= self.angular_scale
        if self._active([ord("p"), ord("P")], now):
            action[5] += self.angular_scale

        # Gripper control
        left_close = self._active([ord("z"), ord("Z")], now)
        right_open = self._active([ord("x"), ord("X")], now)
        if self._active([32], now, hold_s=0.08):  # space
            if now - self._last_gripper_toggle_ts > 0.25:
                self._gripper_closed = not self._gripper_closed
                self._last_gripper_toggle_ts = now
        if self._gripper_closed:
            left_close = True
            right_open = False
        action = np.clip(action, -1.0, 1.0)
        buttons = np.array([left_close, right_open], dtype=bool)
        return action, buttons


#TODO: refactor to use with spacemouse class
class KeyboardInterventionRG(gym.ActionWrapper):
    def __init__(self, env, action_indices=None):
        super().__init__(env)

        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

        self.expert = KeyboardExpert(env)
        self.left, self.right = False, False
        self.action_indices = action_indices

    def action(self, action_dict: dict) -> np.ndarray:
        if not action_dict["expert_acts"]:
            return action_dict, False

        while True:
            expert_a, buttons = self.expert.get_action()
            self.left, self.right = tuple(buttons[:2])
            intervened = False

            if np.linalg.norm(expert_a) > 0.001:
                intervened = True

            if self.gripper_enabled:
                if self.left:
                    gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                    intervened = True
                elif self.right:
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
            time.sleep(0.001)
            # return action_dict, False

    def step(self, action_dict):
        new_action_dict = action_dict.copy()
        new_action_dict, replaced = self.action(new_action_dict)

        obs, rew, done, truncated, info = self.env.step(new_action_dict)
        if replaced:
            info["intervene_action"] = new_action_dict["action"]
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info


class KeyboardInterventionHGFinishAfterStart(KeyboardInterventionRG):
    def action(self, action_dict: dict) -> np.ndarray:
        while True:
            expert_a, buttons = self.expert.get_action()
            self.left, self.right = tuple(buttons[:2])
            intervened = False

            if np.linalg.norm(expert_a) > 0.001:
                intervened = True

            if self.gripper_enabled:
                if self.left:
                    gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                    intervened = True
                elif self.right:
                    gripper_action = np.random.uniform(0.9, 1, size=(1,))
                    intervened = True
                else:
                    gripper_action = np.zeros((1,))
                expert_a = np.concatenate((expert_a, gripper_action), axis=0)

            if self.action_indices is not None:
                filtered_expert_a = np.zeros_like(expert_a)
                filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
                expert_a = filtered_expert_a

            if not action_dict["expert_acts"] and not intervened:
                return action_dict, False

            if intervened:
                action_dict["action"] = expert_a
                return action_dict, True

            time.sleep(0.001)


class KeyboardToggleHG(gym.Wrapper):
    """Thin wrapper that surfaces SPACE-press events as info signals.

    On detection, sets ``info["human_toggle_signal"] = True`` for that
    step. The action is NOT modified. The caller is responsible for
    using the signal to drive whatever gating logic it wants.

    NOTE: the main train flow (HumanGatedHeuristicWithButtonPresses)
    polls the keyboard inside the heuristic's evaluate_state(), so this
    wrapper is not needed there. It is kept for stand-alone test scripts
    (e.g. test_human_gated_toggle.py) that drive the heuristic
    externally via start_/end_human_intervention().

    Each physical keypress is consumed once: only the most recent press
    triggers a signal, and only on the first step that observes it.
    Default toggle key is SPACE (keycode 32); override with ``toggle_keys``.
    """

    def __init__(self, env, toggle_keys=None):
        super().__init__(env)
        self.expert = KeyboardExpert(env)
        self.toggle_keys = toggle_keys if toggle_keys is not None else [32]
        self._last_consumed_ts = 0.0

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        # Robosuite's viewer is created lazily; make sure the keypress
        # callback is wired up before we read _last_key_event.
        self.expert._register_viewer_callbacks_if_needed()
        latest_ts = -1.0
        for code in self.toggle_keys:
            ts = self.expert._last_key_event.get(int(code), -1e9)
            if ts > latest_ts:
                latest_ts = ts
        if latest_ts > self._last_consumed_ts:
            info["human_toggle_signal"] = True
            self._last_consumed_ts = latest_ts
        return obs, rew, done, truncated, info


# based on "external/hil-serl/serl_robot_infra/franka_env/envs/wrappers.py"
class SpacemouseInterventionRG(gym.ActionWrapper):
    def __init__(self, env, action_indices=None, frame="base"): #frame: "base" or "ee"
        super().__init__(env)

        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

        self.expert = SpaceMouseExpert(frame=frame)
        self.left, self.right = False, False
        self.action_indices = action_indices

        self.frame = frame

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
                action_dict["action"] = expert_a #TODO: transform action to base frame if frame="ee"
                return action_dict, True
            #time.sleep(0.01) #TODO
            time.sleep(0.001)
            # print("wait for an expert action")

    def step(self, action_dict):
        new_action_dict = action_dict.copy()
        if self.frame == "base": #read expert action in base frame (standard HIL-SERL implementation)
            new_action_dict, replaced = self.action(new_action_dict) 
        elif self.frame == "ee": #read expert action in ee frame (modified HIL-SERL implementation); action reading happens in RelativeFrameRG wrapper
            replaced = new_action_dict["expert_acts"]

        obs, rew, done, truncated, info = self.env.step(new_action_dict)
        # Actor attribution (TEMP DEBUG): SPACEMOUSE = the human space-mouse
        # overrode this step; EXPERT = the action passed into step() ran
        # unchanged. 'executed' is what actually ran; 'expert' is what the
        # scripted expert computed -- so you can tell whether the [wall_clear] /
        # [offset_ee] offset logged this step actually drove the robot or was
        # overridden by the human. cyan = spacemouse, yellow = expert.
        actor = "SPACEMOUSE" if replaced else "EXPERT"
        color = "\033[96m" if replaced else "\033[93m"
        try:
            exec_a = np.round(np.asarray(new_action_dict.get("action"), dtype=float), 4)
            in_a = np.round(np.asarray(action_dict.get("action"), dtype=float), 4)
            print(f"{color}[ACTOR={actor}] executed={exec_a} expert={in_a}\033[0m")
        except (TypeError, ValueError):
            print(f"{color}[ACTOR={actor}]\033[0m")
        if replaced:
            info["intervene_action"] = new_action_dict["action"]
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info

class SpacemouseInterventionHGFinishAfterStart(SpacemouseInterventionRG):
  def action(self, action_dict: dict) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
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
            
            # print_green(f"TEST SpacemouseInterventionHGFinishAfterStart INTERVENED: {intervened}")
            
            # if expert haven't acted yet and expert didn't intervene this time
            # we execute robot policy action
            if not action_dict["expert_acts"] and not intervened:
                return action_dict, False

            if intervened:
                action_dict["action"] = expert_a
                return action_dict, True
            #time.sleep(0.01) #TODO
            time.sleep(0.001)
            # print("wait for an expert action")


class SpacemouseInterventionHG(SpacemouseInterventionRG):
  def action(self, action_dict: dict) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
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
        
        # print_green(f"TEST SpacemouseInterventionHGFinishAfterStart INTERVENED: {intervened}")
        
        # if expert haven't acted yet and expert didn't intervene this time
        # we execute robot policy action
        if intervened:
            print("SPACEMOUSE ACTS")
        else:
            print("EXPERT ACTS")
        if not action_dict["expert_acts"] and not intervened:
            return action_dict, False

        if intervened:
            action_dict["action"] = expert_a
            return action_dict, True
        
        return action_dict, False

class SpacemouseInterventionFake(gym.ActionWrapper):
    def __init__(self, env, action_indices=None, frame="base"): #frame: "base" or "ee"
        super().__init__(env)

        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

        self.expert = None
        self.left, self.right = False, False
        self.action_indices = action_indices

        self.frame = frame
    
    def action(self, action_dict: dict) -> np.ndarray:
        return action_dict, False

    def step(self, action_dict):
        obs, rew, done, truncated, info = self.env.step(action_dict)
        return obs, rew, done, truncated, info

class RLExpertInterventionRG(gym.ActionWrapper): #TODO: test it
    def __init__(self, env):
        super().__init__(env)

        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False
        self.frame = "ee"

    def action(self, action_dict: dict):
        # print("EXPERT ACTS:", action_dict["expert_acts"])
        if action_dict["expert_acts"]:
            return action_dict, True
        return action_dict, False

    def step(self, action_dict):
        action_dict, replaced = self.action(action_dict)

        obs, rew, done, truncated, info = self.env.step(action_dict)
        if replaced:
            info["intervene_action"] = action_dict["action"]
        return obs, rew, done, truncated, info


class ActionFillWrapper(gym.Wrapper):
    """Train a policy in a *reduced* action space and zero-fill back to full.

    Both the policy and the (reduced) scripted expert emit actions on a subset
    of the full 6D/7D action vector ``[x, y, z, rx, ry, rz, (gripper)]``
    selected by ``active_indices`` (indices into that full EE-frame layout).
    This wrapper exposes the reduced ``action_space`` to everything above it
    (policy, replay buffer, ``env.action_space.sample()``) and scatters the
    reduced action into a full-width vector -- zeros on every non-active axis
    -- before the inner env / frame transforms.

    It must sit ABOVE ``RelativeFrameRG`` / ``RelativeFrame`` so the action is
    full-width by the time the adjoint transform (which needs the 6 spatial
    dims) runs.

    On the return path ``info["intervene_action"]`` (full, produced by the
    intervention wrappers below) is projected back down to the reduced space,
    so the transition the actor stores matches the policy's action space.

    Handles both the ``action_dict`` (robot-gated) and plain-array interfaces.
    """

    def __init__(self, env, active_indices):
        super().__init__(env)
        self.full_dim = int(env.action_space.shape[0])
        self.active_indices = list(active_indices)
        assert len(self.active_indices) > 0, "active_indices must be non-empty"
        assert all(0 <= i < self.full_dim for i in self.active_indices), (
            f"active_indices {self.active_indices} out of range for full "
            f"action dim {self.full_dim}"
        )
        assert len(set(self.active_indices)) == len(self.active_indices), (
            f"active_indices {self.active_indices} contains duplicates"
        )
        self.reduced_dim = len(self.active_indices)

        low = np.asarray(env.action_space.low, dtype=np.float32)[self.active_indices]
        high = np.asarray(env.action_space.high, dtype=np.float32)[self.active_indices]
        self.action_space = Box(low, high, dtype=np.float32)
        print_green(
            f"ActionFillWrapper: full_dim={self.full_dim}, "
            f"active_indices={self.active_indices}, reduced_dim={self.reduced_dim}"
        )

    def _to_full(self, action):
        """Scatter a reduced action into a full-width zero-filled vector."""
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.reduced_dim:
            raise ValueError(
                f"ActionFillWrapper expected a reduced action of dim "
                f"{self.reduced_dim}, got dim {action.shape[0]}"
            )
        full = np.zeros(self.full_dim, dtype=np.float32)
        full[self.active_indices] = action
        return full

    def _to_reduced(self, action):
        """Project a full action down to the reduced (active) subspace."""
        return np.asarray(action, dtype=np.float32).reshape(-1)[self.active_indices]

    def step(self, action):
        if isinstance(action, dict):
            action_dict = dict(action)
            action_dict["action"] = self._to_full(action_dict["action"])
            obs, rew, done, truncated, info = self.env.step(action_dict)
        else:
            obs, rew, done, truncated, info = self.env.step(self._to_full(action))

        if "intervene_action" in info:
            info["intervene_action"] = self._to_reduced(info["intervene_action"])
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
        if self.env.frame == "ee":
            original_action_dict, replaced = self.action(action_dict)
            if replaced:
                transformed_action_dict = original_action_dict
                transformed_action_dict["action"] = self.transform_action(transformed_action_dict["action"]) #convert ee action to base frame
                transformed_action_dict["expert_acts"] = True
                # print_green(f"TEST RELATIVE FRAME RG TRANSFORMED ACTION DICT: {transformed_action_dict}")
        obs, reward, done, truncated, info = self.env.step(transformed_action_dict)
        info['original_state_obs'] = copy.deepcopy(obs['state'])

        # this is to convert the spacemouse intervention action
        if "intervene_action" in info:
            info["intervene_action"] = self.transform_action_inv(info["intervene_action"])

        # Update adjoint matrix
        self.adjoint_matrix = construct_adjoint_matrix(obs["state"]["tcp_pose"])

        # Transform observation to spatial frame
        transformed_obs = self.transform_observation(obs)
        # print("TRANSFORMED OBS (RELATIVE FRAME RG Wrapper):", transformed_obs)
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

        #TODO: need to transform object observations here as well
        # Keys will be different for different object
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

class RelativeFrameRGWithExpert(RelativeFrameRG):
    def transform_observation(self, obs):
        """
        Transform observations from spatial(base) frame into body(end-effector) frame
        using the adjoint matrix
        """
        obs =super().transform_observation(obs)

        # KEY: cube_pos
        # KEY: cube_quat
        # KEY: gripper_to_cube_pos
        # merged_privileged_obs = []
        # for obs_name, obs_value in obs["state"]["privileged"].items():
        #     if "pose" in obs_name:
        #         print("TRANSFORM", obs_name)
        #         T_b_o = construct_homogeneous_matrix(obs_value)
        #         T_b_r = self.T_r_o_inv @ T_b_o

        #         # Reconstruct transformed tcp_pose vector
        #         p_b_r = T_b_r[:3, 3]
        #         theta_b_r = R.from_matrix(T_b_r[:3, :3]).as_quat()
        #         obs_value = np.concatenate((p_b_r, theta_b_r))
        #     else:
        #         print("NOT TRANSFORM", obs_name)
        #     merged_privileged_obs.extend(obs_value)
        
        # obs["state"]["privileged"] = np.array(merged_privileged_obs)

        # return obs

        if isinstance(obs["state"].get("privileged"), dict):
            for obs_name, obs_value in obs["state"]["privileged"].items():
                if "pose" in obs_name:
                    T_b_o = construct_homogeneous_matrix(obs_value)
                    T_b_r = self.T_r_o_inv @ T_b_o

                    p_b_r = T_b_r[:3, 3]
                    theta_b_r = R.from_matrix(T_b_r[:3, :3]).as_quat()
                    obs["state"]["privileged"][obs_name] = np.concatenate((p_b_r, theta_b_r))
        return obs


# based on "external/conrft/examples/experiments/task1_pick_banana/wrapper.py"
# obs["state"][0,0] because keys are sorted in lexicographical order (gripper_pose comes first)
class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, 0]
        return obs, info

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        action = copy.deepcopy(action)
        grasp_action = action[..., -1] #TODO
        
        grasp_action = np.where(grasp_action > 0.5, 1, np.where(grasp_action < -0.5, -1, 0) )
        action[..., -1] = grasp_action
        
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]

        # print("LAST GRIPPER POS:", self.last_gripper_pos, "1-open, 0-close")
        # print("GRIPPER ACTION:", action[-1], "1-to_open, -1-to_close")

        if (action[-1] < -0.5 and self.last_gripper_pos > 0.9) or (
            action[-1] > 0.5 and self.last_gripper_pos < 0.9
        ):
            info["grasp_penalty"] = self.penalty
        else:
            info["grasp_penalty"] = 0.0
        # print("GRASP PENALTY:", info["grasp_penalty"])

        self.last_gripper_pos = observation["state"][-1, 0]
        return observation, reward, terminated, truncated, info
    
class GripperPenaltyWrapperRG(gym.Wrapper):
    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, 0]
        return obs, info

    def step(self, action_dict): #TODO
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        action_dict = copy.deepcopy(action_dict)
        # print("ACTION DICT:", action_dict)
        grasp_action = action_dict["action"][..., -1] #TODO
        # print_green(f"TEST GRASP ACTION: {grasp_action}")
        
        grasp_action = np.where(grasp_action > 0.5, 1, np.where(grasp_action < -0.5, -1, 0) )
        action_dict["action"][..., -1] = grasp_action

        # print_green(f"TEST ACTION DICT: {action_dict}")
        
        # print("BEFORE STEP")
        observation, reward, terminated, truncated, info = self.env.step(action_dict)
        # print("AFTER STEP")
        # print("OBSERVATION:", observation["state"])
        if "intervene_action" in info:
            action_dict["action"] = info["intervene_action"]

        # print("LAST GRIPPER POS:", self.last_gripper_pos, "1-open, 0-close")
        # print("GRIPPER ACTION:", action_dict["action"][-1], "1-to_open, -1-to_close")
        if (action_dict["action"][-1] < -0.5 and self.last_gripper_pos > 0.9) or ( #TODO
            action_dict["action"][-1] > 0.5 and self.last_gripper_pos < 0.9
        ):
            info["grasp_penalty"] = self.penalty
        else:
            info["grasp_penalty"] = 0.0
        # print_green(f"LAST GRIPPER POS: {self.last_gripper_pos}")
        # print_green(f"GRASP PENALTY: {info['grasp_penalty']}")
        # if info["grasp_penalty"] != 0.0:
        #     time.sleep(5)

        self.last_gripper_pos = observation["state"][-1, 0]
        # print("NEW GRIPPER POS:", self.last_gripper_pos, "1-open, 0-close")
        return observation, reward, terminated, truncated, info

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

def print_green(text):
    print(f"\033[92m{text}\033[0m")


class RemoveTCPOrientationWrapper(gym.ObservationWrapper):
    """Drop orientation from selected proprio observations, keeping position only.

    For each key in ``keys`` that is present in the ``state`` observation, only
    the first three components are kept and the rest are discarded:
      - ``tcp_pose``: keep the xyz position, drop the orientation (euler/quat).
      - ``tcp_vel`` : keep the linear velocity, drop the angular velocity.

    Placement: this must sit ABOVE ``RelativeFrameRG`` (which needs the full
    6-DoF pose/vel to build the adjoint transform) and ABOVE ``Quat2EulerWrapper``
    (whose assert expects a 7-D ``tcp_pose``), and BELOW ``SERLObsWrapper`` (which
    flattens the ``state`` dict into a single vector). Keys that are absent from
    the observation -- or already have <= 3 dims -- are left untouched.
    """

    POSITION_DIM = 3

    def __init__(self, env, keys=("tcp_pose", "tcp_vel")):
        super().__init__(env)
        self.keys = tuple(keys)
        state_space = self.observation_space["state"]
        for key in self.keys:
            if key not in state_space.keys():
                continue
            box = state_space[key]
            if box.shape[0] <= self.POSITION_DIM:
                continue
            state_space[key] = spaces.Box(
                box.low[: self.POSITION_DIM],
                box.high[: self.POSITION_DIM],
                dtype=box.dtype,
            )
        print_green(
            f"RemoveTCPOrientationWrapper active: stripped orientation from {self.keys}"
        )

    def observation(self, observation):
        for key in self.keys:
            if key in observation["state"]:
                observation["state"][key] = observation["state"][key][: self.POSITION_DIM]
        return observation


class SERLObsWrapperWithExpert(gym.ObservationWrapper):
    """
    This observation wrapper treat the observation space as a dictionary
    of a flattened state space and the images.
    """

    def __init__(self, env, proprio_keys=None, expert_proprio_keys=None):
        super().__init__(env)
        self.proprio_keys = proprio_keys
        self.expert_proprio_keys = expert_proprio_keys

        if self.proprio_keys is None:
            self.proprio_keys = list(self.env.observation_space["state"].keys())
        if self.expert_proprio_keys is None:
            self.expert_proprio_keys = list(self.env.observation_space["state"].keys())

        self.proprio_space = gym.spaces.Dict(
            {key: self.env.observation_space["state"][key] for key in self.proprio_keys}
        )
        self.expert_proprio_space = gym.spaces.Dict(
            {key: self.env.observation_space["state"][key] for key in self.expert_proprio_keys}
        )
        # print_green(f"TEST EXPERT PROPRIO SPACE: {self.expert_proprio_space}")

        
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

        # self.expert_observation_space = gym.spaces.Dict(
        #     {
        #         "state": flatten_space(self.expert_proprio_space)
        #     }
        # )
        self.expert_observation_space = self.observation_space #TODO
        # print_green(f"SERL OBSERVATION SPACE: {self.observation_space}")

    def observation(self, obs):
        if "images" in obs.keys():
            policy_obs = {
                "state": flatten(
                    self.proprio_space,
                    {key: obs["state"][key] for key in self.proprio_keys},
                ),
                **(obs["images"]),
            }
        else:
            policy_obs = {
                "state": flatten(
                    self.proprio_space,
                    {key: obs["state"][key] for key in self.proprio_keys},
                )
            }
        # expert_obs = {
        #     "state": flatten(
        #         self.expert_proprio_space,
        #         {key: obs["state"][key] for key in self.expert_proprio_keys},
        #     )
        # }
        expert_obs = obs #TODO
        return policy_obs, expert_obs
    
    def step(self, action_dict):
        obs, reward, terminated, truncated, info = self.env.step(action_dict)
        obs, expert_obs = self.observation(obs)
        info["expert_obs"] = expert_obs
        # print("OBS ARE (SERL OBS WRAPPER):", obs)
        # print("EXPERT OBS ARE (SERL OBS WRAPPER):", expert_obs)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info =  self.env.reset(**kwargs)
        obs, expert_obs = self.observation(obs)
        info["expert_obs"] = expert_obs
        return obs, info

# class Quat2EulerWrapperWithExpert(Quat2EulerWrapper):
#     """
#     Convert the quaternion representation of the tcp pose to euler angles
#     """

#     def __init__(self, env: Env):
#         super().__init__(env)
#         # print("TEST SHAPE", env.observation_space["state"]["tcp_pose"].shape)
#         # assert env.observation_space["state"]["tcp_pose"].shape == (7,)
#         assert env.expert_observation_space["state"]["tcp_pose"].shape == (7,)
#         # from xyz + quat to xyz + euler
#         # self.observation_space["state"]["tcp_pose"] = spaces.Box(
#         #     -np.inf, np.inf, shape=(6,)
#         # )
#         self.expert_observation_space["state"]["tcp_pose"] = spaces.Box(
#             -np.inf, np.inf, shape=(6,)
#         )


class Quat2EulerWrapperWithExpert(gym.ObservationWrapper):
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

        #TODO:
        # init_privileged_len = env.observation_space["state"]["privileged"].shape[0]
        # self.observation_space["state"]["privileged"] = spaces.Box(-np.inf, np.inf, shape=(init_privileged_len - 1,)) #TODO: change for different env


        # print("TEST Quat2EulerWrapperWithExpert ASSERT", env.observation_space["state"]["privileged"])
        # for key in env.observation_space["state"]["privileged"].keys():
        #     if "pose" in key:
        #         print("Quat2EulerWrapperWithExpert ASSERT", key)
        #         assert env.observation_space["state"]["privileged"][key].shape == (7,)
        #         # from xyz + quat to xyz + euler
                # self.observation_space["state"]["privileged"][key] = spaces.Box(
                #     -np.inf, np.inf, shape=(6,)
                # )


    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        if isinstance(observation["state"].get("privileged"), dict):
            for key in observation["state"]["privileged"].keys():
                if "pose" in key:
                    pose = observation["state"]["privileged"][key]
                    observation["state"]["privileged"][key] = np.concatenate(
                        (pose[:3], R.from_quat(pose[3:]).as_euler("xyz"))
                    )
        return observation

# class MergePrivilegedObsWithExpert(gym.ObservationWrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self.observation_space["state"]["privileged"] = spaces.Box(-np.inf, np.inf, shape=(5,))

#     def observation(self, observation):
#         # convert tcp pose from quat to euler
#         # for cube
#         ee_pos = observation["state"]["tcp_pose"][:3]
#         ee_orn = observation["state"]["tcp_pose"][3:]
#         cube_pos = observation["state"]["privileged"]["cube_pose"][:3]
#         cube_orn = observation["state"]["privileged"]["cube_pose"][3:]

#         delta_pos = cube_pos - ee_pos #in initial ee frame
#         R_ee = R.from_euler("xyz", ee_orn)
#         delta_pos = R_ee.inv().apply(delta_pos) #in current ee frame

#         delta_yaw = cube_orn[-1] - ee_orn[-1]
#         delta_yaw_enc = np.array([np.sin(4 * delta_yaw), np.cos(4 * delta_yaw)])

#         observation["state"]["privileged"] = np.concatenate([delta_pos, delta_yaw_enc])

#         # merged_privileged_obs = []
#         # for key in observation["state"]["privileged"].keys():
#         #     print("KEY", key, observation["state"]["privileged"][key])
#         #     merged_privileged_obs.extend(observation["state"]["privileged"][key])
#         # observation["state"]["privileged"] = np.array(merged_privileged_obs)
#         # print("MERGED PRIVILEGED OBS ARE (MERGE PRIVILEGED OBS WITH EXPERT):", observation["state"]["privileged"])
#         return observation