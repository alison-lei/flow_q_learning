from __future__ import annotations

from typing import Any, Dict, Literal, List

from lift_env_cfg import LiftEnvConfig
from sim_env.envs.robosuite_base_env import RobosuiteBaseEnv
import numpy as np

from scipy.spatial.transform import Rotation as R

import time
import robosuite
from robosuite.environments.manipulation.lift import Lift
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.placement_samplers import UniformRandomSampler


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))

class FixedCubeLift(Lift):
    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.cube = BoxObject(
            name="cube",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[1, 0, 0, 1],
            material=redwood,
            rng=self.rng,
        )

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube,
                x_range=[-0.03, 0.03],
                y_range=[-0.03, 0.03],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
                rng=self.rng,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cube,
        )

class PandaLiftGymEnv(RobosuiteBaseEnv):
    def __init__(
        self,
        config: LiftEnvConfig = LiftEnvConfig,
        fake_env: bool = False,
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        save_video: bool = False,
        video_path: str = "./videos",
        privileged_obs: bool = False,
        expert_extra_keys: List[str] = [],
        **kwargs,
    ):
        # print("PRIVILEGED OBS (LIFT):", privileged_obs)
        robosuite.environments.base.register_env(FixedCubeLift)
        super().__init__(config, fake_env, render_mode, save_video, video_path, privileged_obs=privileged_obs, **kwargs)
        self.expert_extra_keys = expert_extra_keys
        self.table_height = self._env.model.mujoco_arena.table_offset[2]
    
    def _change_camera_pose(self):
        cam_name = "robot0_robotview"
        cam_id = self._env.sim.model.camera_name2id(cam_name)

        cam_pos = [0, 0.3, 0.25]
        cam_quat = [0.4853543, 0.2473134, -0.41900563, -0.72643071]
        self._env.sim.model.cam_pos[cam_id] = cam_pos
        self._env.sim.model.cam_quat[cam_id] = cam_quat
    
    def _process_privileged_obs(
        self,
        obs: Dict[str, Any],
        out: Dict[str, Dict[str, np.ndarray]],
    ) -> Dict[str, Dict[str, np.ndarray]]:

        out["state"]["privileged"] = {}
        privileged_obs = self._extract_privileged_from_raw_obs(obs)

        # EE pose in base frame (position + quaternion)
        ee_pos = out["state"]["tcp_pose"][:3]
        ee_rot = R.from_quat(out["state"]["tcp_pose"][3:])
        ee_mat = ee_rot.as_matrix()

        # Cube pose: world → base frame
        cube_pos_world = privileged_obs["cube_pos"]
        cube_rot_world = R.from_quat(privileged_obs["cube_quat"])
        R_wb = self.world_to_base[:3, :3]
        cube_pos = R_wb @ cube_pos_world + self.world_to_base[:3, 3]
        cube_mat = (R_wb @ cube_rot_world.as_matrix())

        # --- 1. delta_pos in EE frame (3D) ---
        delta_pos_ee = ee_rot.inv().apply(cube_pos - ee_pos)

        # --- 2. symmetry-aware relative yaw (2D) ---
        # Find which cube axis is most aligned with base z = [0,0,1].
        # cube_mat columns are cube-local axes expressed in base frame;
        # dot with z = third row of cube_mat.
        z_dots = cube_mat[2, :]
        up_idx = int(np.argmax(np.abs(z_dots)))
        # Pick one of the two horizontal cube axes
        horiz_idx = 0 if up_idx != 0 else 1
        horiz_axis = cube_mat[:, horiz_idx]
        # Angle of horizontal cube axis in xy-plane (base frame)
        cube_angle = np.arctan2(horiz_axis[1], horiz_axis[0])
        # Angle of gripper x-axis in xy-plane (base frame)
        gripper_angle = np.arctan2(ee_mat[1, 0], ee_mat[0, 0])
        relative_angle = cube_angle - gripper_angle
        sym_yaw = np.array([
            np.sin(4 * relative_angle),
            np.cos(4 * relative_angle),
        ])

        # print_green(f"DELTA POS EE: {delta_pos_ee}")
        # print_green(f"SYM YAW: {sym_yaw}")

        out["state"]["privileged"] = np.concatenate([delta_pos_ee, sym_yaw]).astype(np.float32)
        return out


    def get_privileged_shape(self):
        # delta_pos_ee(3) + sym_yaw(2) = 5
        return (5,)

    def _cube_pos_base(self, body_name: str) -> np.ndarray:
        body_id = self._env.sim.model.body_name2id(body_name)
        pos_world = np.asarray(self._env.sim.data.body_xpos[body_id])
        R_wb = self.world_to_base[:3, :3]
        return R_wb @ pos_world + self.world_to_base[:3, 3]

    _OUT_OF_BOUNDS_XY_MARGIN: float = 0.01
    _GRIPPER_CLOSED_THRESH: float = 0.8
    _EMPTY_GRASP_THRESH: float = 0.4

    def _cube_out_of_bounds(self, body_name: str, grasping: bool) -> bool:
        pos = self._cube_pos_base(body_name)
        low = self.config.ABS_POSE_LIMIT_LOW[:3]
        high = self.config.ABS_POSE_LIMIT_HIGH[:3]
        m = self._OUT_OF_BOUNDS_XY_MARGIN
        # print("CUBE NAME: ", body_name)
        # print("CUBE POS: ", pos)
        # print()

        if pos[2] < low[2] - 0.1:
            return True

        if not grasping:
            if pos[0] < low[0] + m or pos[0] > high[0] - m:
                return True
            if pos[1] < low[1] + m or pos[1] > high[1] - m:
                return True

        return False

    def step(self, action: np.ndarray):
        start_time = time.time()
        action = np.asarray(action, dtype=np.float32) #delta action (euler angles in base frame)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # control gripper
        if self.config.GRIPPER_IS_ACTIVE:
            gripper_action = action[6] * self.config.ACTION_SCALE[2]
            self._run_gripper_command(gripper_action)

        # get next absolute pose (base frame)
        xyz_delta = action[:3]
        next_pos = self.curr_ee_pose[:3] + xyz_delta * self.config.ACTION_SCALE[0]
        next_quat = ( #delta applied in the base frame (Relative Frame wrapper transforms action from ee frame to base frame before using delta here)
            R.from_euler("xyz", action[3:6] * self.config.ACTION_SCALE[1])
            * R.from_quat(self.curr_ee_pose[3:])
        ).as_quat()
        next_pose = np.concatenate([next_pos, next_quat]) #in base frame
        next_pose = self._clip_safety_box(next_pose) #in base frame

        action = np.concatenate([
            next_pose[:3], 
            R.from_quat(next_pose[3:]).as_rotvec(),
        ])
        if self.config.GRIPPER_IS_ACTIVE:
            action = np.concatenate([action, [0]]) #gripper action is controlled separately
        step_out = self._env.step(action) #in base frame

        if len(step_out) == 4:
            raw_obs, _raw_reward, done, info = step_out
        else:
            raw_obs, _raw_reward, done, _truncated, info = step_out
        self.curr_path_length += 1

        if self.config.SLEEP:
            dt = time.time() - start_time
            time.sleep(max(0.0, self._env.control_timestep - dt))

        obs = self._process_obs(raw_obs)
        success = self._check_success(info)
        reward = self.config.reward_pos if success else self.config.reward_neg
        done = bool(done or self.curr_path_length >= self.config.MAX_EPISODE_LENGTH or success)

        grip = float(np.ravel(obs["state"]["gripper_pose"])[0])
        grasping = (grip < self._GRIPPER_CLOSED_THRESH) and (grip >= self._EMPTY_GRASP_THRESH)
        if self._cube_out_of_bounds("cube_main", grasping):
            done = True
            info["cube_out_of_bounds"] = "cube_main"

        if self.render_mode == "human":
            self._env.render()

        step_info = dict(info)
        step_info["succeed"] = int(success)
        # print_green(f"GRIPPER POSE MORE: {obs['state']['gripper_pose']}")
        # print_green(f"TEST BASE ENV OBS: {obs}")
        return obs, float(reward), done, False, step_info
    
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        # print_green(f"CUBE SIZE: {self._env.cube.size}")
        return obs, info
    


    # def _compute_cube_ee_geometry(self):
    #     """Returns (delta_pos_ee, relative_angle) for the cube relative to the EE.

    #     delta_pos_ee: cube position minus EE position, expressed in EE frame (3D).
    #     relative_angle: yaw misalignment between gripper and cube horizontal axis,
    #                     accounting for 4-fold cube symmetry.
    #     """
    #     # ee_pos = self.curr_ee_pose[:3]
    #     ee_rot = R.from_quat(self.curr_ee_pose[3:])
    #     ee_mat = ee_rot.as_matrix()

    #     # Cube pose: world → base frame
    #     # cube_pos_world = self._env.sim.data.body_xpos[self._env.cube_body_id].copy()
    #     cube_mat_world = self._env.sim.data.body_xmat[self._env.cube_body_id].reshape(3, 3).copy()
    #     R_wb = self.world_to_base[:3, :3]
    #     # cube_pos = R_wb @ cube_pos_world + self.world_to_base[:3, 3]
    #     cube_mat = R_wb @ cube_mat_world

    #     # delta_pos in EE frame
    #     # delta_pos_ee = ee_rot.inv().apply(cube_pos - ee_pos)

    #     # symmetry-aware relative yaw:
    #     # find which cube axis is most vertical, pick a horizontal axis, measure yaw delta
    #     z_dots = cube_mat[2, :]
    #     up_idx = int(np.argmax(np.abs(z_dots)))
    #     horiz_idx = 0 if up_idx != 0 else 1
    #     horiz_axis = cube_mat[:, horiz_idx]
    #     cube_angle = np.arctan2(horiz_axis[1], horiz_axis[0])
    #     gripper_angle = np.arctan2(ee_mat[1, 0], ee_mat[0, 0])
    #     relative_angle = cube_angle - gripper_angle

    #     return relative_angle

    # def _compute_alignment_reward(self) -> float:
        # """Per-step reward for gripper yaw alignment with cube.

        # Returns a value in [0, ALIGNMENT_REWARD_SCALE]:
        #   - alignment=1 when perfectly aligned, 0 when 45° off (4-fold cube symmetry)
        # """
        # relative_angle = self._compute_cube_ee_geometry()
        # # alignment = 0.5 * (1.0 + np.cos(4.0 * relative_angle))
        # alignment = 0.5 * (np.cos(4.0 * relative_angle))
        # print("ALIGNMENT REWARD:", self.config.ALIGNMENT_REWARD_SCALE * alignment)
        # return self.config.ALIGNMENT_REWARD_SCALE * alignment

    # #TODO: this is used only for expert training
    # def step(self, action):                                                                                                       
    #     obs, reward, done, truncated, info = super().step(action)                                                                 
    #     # reward += self._compute_alignment_reward()  
    #     if info["succeed"]:                                                                                                               
    #         relative_angle = self._compute_cube_ee_geometry()                                                                          
    #         alignment = 0.5 * (1.0 + np.cos(4.0 * relative_angle))  
    #         if alignment > 0.95:                                                                      
    #             reward += self.config.ALIGNMENT_REWARD_SCALE
                                                                                    
    #     return obs, reward, done, truncated, info


class PandaLiftGymEnvRG(PandaLiftGymEnv):
    def step(self, action_dict: dict):
        return super().step(action_dict["action"])


if __name__ == "__main__":
    from franka_env.envs.wrappers import Quat2EulerWrapper
    from serl_launcher.wrappers.chunking import ChunkingWrapper
    from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper

    from sim_env.envs.modified_wrappers import KeyboardInterventionRG, RelativeFrameRG

    env = PandaLiftGymEnvRG(config=LiftEnvConfig, render_mode="human")
    env = KeyboardInterventionRG(env)
    env = RelativeFrameRG(env)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env, proprio_keys=["tcp_pose", "tcp_vel", "gripper_pose", "wrist_force", "wrist_torque"])
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

    env.reset()
    for _ in range(10000):
        action = np.zeros(7, dtype=np.float32)
        action_dict = {"action": action, "expert_acts": True}
        env.step(action_dict)