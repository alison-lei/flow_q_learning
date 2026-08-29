import numpy as np
from envs.active_rl_for_assembly.nut_assembly_env_cfg import NutAssemblyEnvConfig


class NutAssemblySquareEnvConfig(NutAssemblyEnvConfig):
    """Workspace config for the square-nut assembly task.

    The square nut and its target peg (peg1) live on the +y half of the
    table: robosuite spawns the square nut at world y in [0.11, 0.225] and
    peg1 sits at world y = +0.1. Here world_to_base is a pure translation
    (base = world + [0.56, 0, -0.912]), so base y == world y and those
    locations land at base y ~0.1..0.225.

    The round-nut config places the workspace on the -y half (reset y =
    -0.1675, y limits [-0.305, -0.03]). We mirror only the y-extent to +y so
    the EE clip box, reset pose, and OOB checks cover the square nut's native
    spawn; every magnitude is inherited from the round config unchanged, only
    the y sign flips. The reset orientation and x/z extents are identical to
    the round task (the Panda reaches +y as readily as -y from its base).
    """
    MAX_EPISODE_LENGTH: int = 150

    TABLE_X = NutAssemblyEnvConfig.TABLE_X
    Z_OFFSET = NutAssemblyEnvConfig.Z_OFFSET
    TABLE_HEIGHT = NutAssemblyEnvConfig.TABLE_HEIGHT
    Y_RANGE = NutAssemblyEnvConfig.Y_RANGE
    RX_RY_RANGE = NutAssemblyEnvConfig.RX_RY_RANGE

    RESET_POSE = np.array(  # in base frame
        [
            TABLE_X - 0.2,
            0.1675,  # +y mirror of the round config's -0.1675
            TABLE_HEIGHT + 0.1 - Z_OFFSET,
            np.pi,
            0,
            np.pi / 2,
        ]
    )

    # ABS_POSE_LIMIT numbers are in base frame. Derived from RESET_POSE so the
    # y-band stays centered on the (now +y) reset pose: y in [0.03, 0.305].
    ABS_POSE_LIMIT_LOW = np.array([
        TABLE_X - 0.21,
        RESET_POSE[1] - Y_RANGE,   # 0.1675 - 0.1375 = 0.03
        TABLE_HEIGHT + 0.015 - Z_OFFSET,
        np.pi - RX_RY_RANGE,
        -RX_RY_RANGE,
        np.pi / 2 - np.pi / 2# -np.pi, -----  -np.pi / 36
    ])

    ABS_POSE_LIMIT_HIGH = np.array([
        TABLE_X + 0.2,
        RESET_POSE[1] + Y_RANGE,   # 0.1675 + 0.1375 = 0.305
        TABLE_HEIGHT + 0.22 - Z_OFFSET,
        np.pi + RX_RY_RANGE,
        RX_RY_RANGE,
        np.pi / 2 + np.pi / 2# np.pi, ----- np.pi + np.pi / 36
    ])

    # The +y task makes robot0_robotview a horizontal mirror of the round
    # view (see PandaNutAssemblySquareGymEnv._change_camera_pose), so the
    # side-policy crop mirrors too: the round config drops the rightmost 190
    # cols (image[..., :-190]); here we drop the leftmost 190 (image[..., 190:])
    # to keep the gripper + nut + peg framed. Verified by render.
    IMAGE_CROP = {
        "wrist_1": lambda image: image,
        "side_policy_256": lambda image: image[100:-50, 220:, :],#[50:-105, 190:, :],
    }

    ACTION_SCALE = np.array([0.05, 0.2, 1])
