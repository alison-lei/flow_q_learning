import numpy as np
from envs.active_rl_for_assembly.default_robosuite_cfg import DefaultRobosuiteConfig

class LiftEnvConfig(DefaultRobosuiteConfig):
    ENV_NAME: str = "FixedCubeLift" #"Lift"
    MAX_EPISODE_LENGTH: int = 300
    RANDOM_RESET: bool = True
    INITIAL_GRIPPER_OPEN: bool = True #False#True

    TABLE_X = 0.56 #in base frame
    Z_OFFSET = 0.912 # robot base z position in world frame 
    TABLE_HEIGHT = 0.8 #table offset
    
    XY_RANGE = 0.08
    RX_RY_RANGE = 0.01

    RANDOM_XY_RANGE: float = 0.05
    RANDOM_RZ_RANGE: float = np.pi / 6

    RESET_POSE = np.array( #in base frame
    [
        TABLE_X,
        0,
        TABLE_HEIGHT + 0.08 - Z_OFFSET,
        np.pi,
        0,
        np.pi/2,
    ])

    # ABS_POSE_LIMIT numbers are in base frame
    ABS_POSE_LIMIT_LOW = np.array([
        TABLE_X - XY_RANGE - 0.01,    # x
        -XY_RANGE - 0.01,    # y  
        TABLE_HEIGHT + 0.015 - Z_OFFSET,  # z (just above table)
        np.pi-RX_RY_RANGE,     # rx (roll)
        -RX_RY_RANGE,     # ry (pitch)
        -np.pi,   # rz (yaw -- allow full rotation)
    ])
    
    ABS_POSE_LIMIT_HIGH = np.array([
        TABLE_X + XY_RANGE + 0.01,
        XY_RANGE + 0.01,
        TABLE_HEIGHT + 0.1 - Z_OFFSET,
        np.pi+RX_RY_RANGE,
        RX_RY_RANGE,
        np.pi,
    ])

    ALIGNMENT_REWARD_SCALE: float = 5 #0.1  # max per-step alignment bonus (1 = perfectly aligned, 0 = 45° off)

    IMAGE_CROP = {
        "wrist_1": lambda image: image,
        "side_policy_256": lambda image: image[50:-200, 170:-80, :]
    }