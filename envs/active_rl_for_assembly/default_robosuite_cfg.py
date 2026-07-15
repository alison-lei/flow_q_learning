import numpy as np
from envs.active_rl_for_assembly.default_env_cfg import DefaultEnvConfig

class DefaultRobosuiteConfig(DefaultEnvConfig):
    MAX_EPISODE_LENGTH: int = 150
    IMAGE_OBS: bool = True
    SEED: int = 0

    ENV_NAME: str = None
    ROBOT: str = "Panda"
    GRIPPER: str = "Robotiq85Gripper"
    GRIPPER_IS_ACTIVE: bool = True
    INITIAL_GRIPPER_OPEN: bool = True
    CONTROLLER: str = "BASIC"
    CONTROL_FREQ: int = 10 # policy just executes for 0.1s
    RENDER_CAMERA: list[str] = ["robot0_robotview", "robot0_eye_in_hand"]
    RENDER_CAMERA_SIZE: tuple[int, int] = (640, 480)
    RENDERER: str = "mujoco"

    REWARD_SHAPING: bool = False
    reward_pos: float = 0.0
    reward_neg: float = -1.0

    REALSENSE_CAMERAS = ["wrist_1", "side_policy_256"]
    CAMERA_NAME_MAP = {
        "wrist_1": "robot0_eye_in_hand",
        "side_policy_256": "robot0_robotview",
    }
    # Available "camera" names = ('frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand').

    CAMERA_SIZE_MAP = {
        "wrist_1": (128, 128),
        "side_policy_256": (256, 256),
    }

    IMAGE_CROP = {
        "wrist_1": lambda image: image,
        "side_policy_256": lambda image: image
    }

    ACTION_SCALE = np.array([0.05, 0.1, 1]) #np.array([0.05, 0.06, 1])
    ABS_POSE_LIMIT_LOW = -np.inf * np.ones((6,))
    ABS_POSE_LIMIT_HIGH = np.inf * np.ones((6,))
    COMPLIANCE_PARAM = {}
    SLEEP: bool = True

    # IK parameters
    SOLVER: str = "quadprog"
    POS_THRESHOLD: float = 1e-4
    ORI_THRESHOLD: float = 1e-4
    MAX_ITERS: int = 200