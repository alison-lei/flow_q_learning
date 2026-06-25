import numpy as np
from typing import Dict

class DefaultEnvConfig:
    """Default configuration for FrankaEnv. Fill in the values below."""

    # SERVER_URL: str = "http://127.0.0.1:5000/"
    # REALSENSE_CAMERAS: Dict = {
    #     "wrist_1": "130322274175",
    #     "wrist_2": "127122270572",
    # }
    IMAGE_CROP: dict[str, callable] = {}
    TARGET_POSE = np.zeros((6,))
    # GRASP_POSE = np.zeros((6,))
    REWARD_THRESHOLD = np.zeros((6,))
    ACTION_SCALE = np.zeros((3,))
    RESET_POSE = np.zeros((6,))
    RANDOM_RESET: bool = False
    RANDOM_XY_RANGE: float = 0.0
    RANDOM_RZ_RANGE: float = 0.0
    ABS_POSE_LIMIT_HIGH = np.zeros((6,))
    ABS_POSE_LIMIT_LOW = np.zeros((6,))
    COMPLIANCE_PARAM: Dict[str, float] = {}
    # RESET_PARAM: Dict[str, float] = {}
    # PRECISION_PARAM: Dict[str, float] = {}
    # LOAD_PARAM: Dict[str, float] = {
    #     "mass": 0.0,
    #     "F_x_center_load": [0.0, 0.0, 0.0],
    #     "load_inertia": [0, 0, 0, 0, 0, 0, 0, 0, 0]
    # }
    BINARY_GRIPPER_THREASHOLD: float = 0.5 #is not present in HIL-SERL
    GRIPPER_PENALTY: float = -0.02 #is not present in HIL-SERL
    # DISPLAY_IMAGE: bool = True #TODO
    # GRIPPER_SLEEP: float = 0.6
    MAX_EPISODE_LENGTH: int = 100
    # JOINT_RESET_PERIOD: int = 0

    PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))
    SLEEP: bool = True