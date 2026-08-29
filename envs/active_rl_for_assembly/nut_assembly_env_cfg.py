import numpy as np
from envs.active_rl_for_assembly.default_robosuite_cfg import DefaultRobosuiteConfig

class NutAssemblyEnvConfig(DefaultRobosuiteConfig):
    ENV_NAME: str = "NutAssembly"
    MAX_EPISODE_LENGTH: int = 150
    RANDOM_RESET: bool = True
    ACTION_SCALE = np.array([0.05, 0.2, 1])

    TABLE_X = 0.56 #in base frame
    Z_OFFSET = 0.912 # robot base z position in world frame 
    TABLE_HEIGHT = 0.82 #table offset

    Y_RANGE = 0.0575 + 0.08
    RX_RY_RANGE = 0.01

    RANDOM_XY_RANGE: float = 0.05
    RANDOM_RZ_RANGE: float = np.pi / 6

    RESET_POSE = np.array( #in base frame
    [
        TABLE_X - 0.2,
        -0.1675,
        TABLE_HEIGHT + 0.1 - Z_OFFSET, #0.16
        np.pi,
        0,
        np.pi/2,
    ])

    # ABS_POSE_LIMIT numbers are in base frame
    ABS_POSE_LIMIT_LOW = np.array([
        TABLE_X - 0.21,
        RESET_POSE[1] - Y_RANGE, #-0.1675-0.0575 = -0.225
        TABLE_HEIGHT + 0.015 - Z_OFFSET,
        np.pi-RX_RY_RANGE,
        -RX_RY_RANGE,
        -np.pi,
    ])
    
    ABS_POSE_LIMIT_HIGH = np.array([
        # TABLE_X + 0.2,
        TABLE_X + 0.2, #np.inf, # TABLE_X + 0.25, #0.3,
        RESET_POSE[1] + Y_RANGE, #+ 0.2, #-0.1675+0.0575 = -0.11 #TODO
        # TABLE_HEIGHT + 0.18 - Z_OFFSET,
        TABLE_HEIGHT + 0.22 - Z_OFFSET,
        np.pi+RX_RY_RANGE,
        RX_RY_RANGE,
        np.pi,
    ])

    #IMAGE_CROP = {
    #    "wrist_1": lambda image: image,
    #    "side_policy_256": lambda image: image[90:-40, 70:-150, :]
    #}

    IMAGE_CROP = {
        "wrist_1": lambda image: image,
        "side_policy_256": lambda image: image[50:-105, :-190, :]
    }
