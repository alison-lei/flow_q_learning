from __future__ import annotations

from typing import Any, Dict, Literal, Tuple

import cv2
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from envs.active_rl_for_assembly.default_robosuite_cfg import DefaultRobosuiteConfig

from scipy.spatial.transform import Rotation as R
from envs.active_rl_for_assembly.quat_scalar_transformations import scalar_first_to_scalar_last

import os
from datetime import datetime
import copy

import robosuite.utils.transform_utils as T

try: #TODO
    import robosuite as suite
    from robosuite.controllers import load_composite_controller_config
    try:
        from robosuite.controllers import load_controller_config as legacy_load_controller_config
    except ImportError:
        legacy_load_controller_config = None
except ImportError as exc:
    ROBOSUITE_IMPORT_ERROR = exc
else:
    ROBOSUITE_IMPORT_ERROR = None

import robosuite.macros as macros
macros.IMAGE_CONVENTION = "opencv" # to get images in opencv format (not flipped in contrast to opengl convention)

# import robosuite.utils.transform_utils as T
# from robosuite.utils.ik_utils import IKSolver
import mink
from envs.active_rl_for_assembly.converge_ik import converge_ik

def print_green(x):
    return print("\033[92m {}\033[00m".format(x))

#TODO: make boolean to choose image vs states observation
class RobosuiteBaseEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}
    def __init__(
        self,
        config: DefaultRobosuiteConfig,
        fake_env: bool = False,
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        save_video: bool = False,
        video_path: str = "./videos",
        privileged_obs: bool = False,
        **env_kwargs,
    ):
        # print("ENV_KWARGS:", env_kwargs)
        super().__init__()
        if ROBOSUITE_IMPORT_ERROR is not None:
            raise ImportError(
                "robosuite is required for PandaStackGymEnv. Install robosuite in this environment."
            ) from ROBOSUITE_IMPORT_ERROR

        self.config = config
        self.render_mode = render_mode
        self.privileged_obs = privileged_obs

        self.save_video = save_video
        if self.save_video:
            print("Saving videos!")
            self.recording_frames = []
        self.video_path = video_path

        self.curr_path_length = 0

        self._allow_retry_from_same_state = False
        self._saved_sim_state = None  # MuJoCo state for retrying from same initial state
        self._saved_env_rng_state = None  
        self._saved_robot_rng_state = None

        try: #TODO
            print("CONTROLLER:", config.CONTROLLER)
            controller_config = load_composite_controller_config(controller=config.CONTROLLER)
        except Exception:
            print("CONTROLLER CONFIG NOT FOUND")
            if legacy_load_controller_config is None:
                raise
            controller_config = legacy_load_controller_config(default_controller="OSC_POSE") 
        #based on controller_config, 'input_type': 'delta', 'input_ref_frame': 'base'
        controller_config["body_parts"]["right"]["input_type"] = "absolute"
        controller_config["body_parts"]["right"]["input_ref_frame"] = "base"
        
        # input_min, input_max, output_min, output_max are only for delta control OSC_POSE controller
        # if we want to use delta control, need to take into account that they impact action scale
        print("CONTROLLER CONFIG:", controller_config)
        print_green(f"REALSENSE_CAMERAS: {config.REALSENSE_CAMERAS}")
        # default init qpos is [0, np.pi / 16.0, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.2, np.pi / 4]
        self._env = suite.make(
            config.ENV_NAME,
            robots=config.ROBOT,
            gripper_types=config.GRIPPER,
            controller_configs=controller_config,
            has_renderer=(render_mode == "human" and not fake_env),
            has_offscreen_renderer=config.IMAGE_OBS,
            render_camera=config.RENDER_CAMERA,
            control_freq=config.CONTROL_FREQ,
            horizon=config.MAX_EPISODE_LENGTH,
            use_object_obs=True, #TODO
            use_camera_obs=config.IMAGE_OBS,
            camera_names=[config.CAMERA_NAME_MAP[k] for k in config.REALSENSE_CAMERAS],
            # camera_heights=[config.CAMERA_SIZE_MAP[k][0] for k in config.REALSENSE_CAMERAS],
            # camera_widths=[config.CAMERA_SIZE_MAP[k][1] for k in config.REALSENSE_CAMERAS],
            camera_widths=[config.RENDER_CAMERA_SIZE[0] for _ in config.REALSENSE_CAMERAS],
            camera_heights=[config.RENDER_CAMERA_SIZE[1] for _ in config.REALSENSE_CAMERAS],
            reward_shaping=config.REWARD_SHAPING,
            ignore_done=False,
            hard_reset=False,
            renderer=config.RENDERER,
            seed=config.SEED,
            # initialization_noise={
            #     "type": "gaussian",      # or "uniform"
            #     "magnitude": 0,       # scale factor
            # },
            **env_kwargs,
        )
        # Use robosuite RNG when available; fallback keeps reset randomization functional.
        self.rng = getattr(self._env, "rng", np.random.default_rng(config.SEED))

        # We add our own initialization noise to the robot end-effector pose instead of joint positions
        self._env.robots[0].initialization_noise["magnitude"] = 0

        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
        )

        state_space = {
            "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
            "tcp_vel": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
            "wrist_force": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            "wrist_torque": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
        }
        if self.config.GRIPPER_IS_ACTIVE:
            state_space["gripper_pose"] = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        image_space = {
            key: spaces.Box(
                low=0,
                high=255,
                shape=(config.CAMERA_SIZE_MAP[key][0], config.CAMERA_SIZE_MAP[key][1], 3),
                dtype=np.uint8,
            )
            for key in config.REALSENSE_CAMERAS
        }

        # print_green(f"PRIVILEGED OBS TEST: {self.privileged_obs}")
        if self.privileged_obs:
            privileged_shape = self.get_privileged_shape()
            state_space["privileged"] = spaces.Box(-np.inf, np.inf, shape=privileged_shape, dtype=np.float32)

        self.observation_space = spaces.Dict( #TODO
            {
                "state": spaces.Dict(state_space),
                "images": spaces.Dict(image_space),
            }
        )

        # print("PRIVILEGED OBS:", self.privileged_obs)
        # if self.privileged_obs:
        #     privileged_shape = self._env.observation_spec()["object-state"].shape
        #     state_space["privileged"] = spaces.Box(-np.inf, np.inf, shape=privileged_shape, dtype=np.float32)
        #     self.expert_observation_space = spaces.Dict( #TODO
        #         {"state": spaces.Dict(state_space)}
        #     )

        # print_green(f"OBSERVATION SPACE: {self.observation_space}")
        # print("EXPERT OBSERVATION SPACE:", self.expert_observation_space)

        if self.render_mode == "human":
            self._env.viewer.width = config.RENDER_CAMERA_SIZE[0]
            self._env.viewer.height = config.RENDER_CAMERA_SIZE[1]
        
        self.resetpos = np.concatenate(
            [config.RESET_POSE[:3], R.from_euler('xyz', config.RESET_POSE[3:]).as_quat()]
        )

        self.xyz_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[:3],
            config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        
        self.rpy_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[3:],
            config.ABS_POSE_LIMIT_HIGH[3:],
            dtype=np.float64,
        )

        self.world_to_base = self.get_world_to_base_transformation_matrix()
        self.base_to_world = self.get_base_to_world_transformation_matrix()
        print("WORLD TO BASE:", self.world_to_base)
        print("BASE TO WORLD:", self.base_to_world)

        self.curr_ee_pose = None
        self._warned_missing_table_body = False
        if self.config.GRIPPER_IS_ACTIVE:
            if self.config.INITIAL_GRIPPER_OPEN:
                # print("RESET GRIPPER TO OPEN")
                self._reset_gripper_to_open()
            else:
                # print("RESET GRIPPER TO CLOSE")
                self._reset_gripper_to_close()
        self.last_reset_pose = None
        self._change_camera_pose()
        # self.highest = 0

        # Create a Mink configuration
        model = self._env.sim.model._model
        # data = self._env.sim.data._data
        site_name = "gripper0_right_grip_site" #gripper0_right_grip_site
        self.ik_configuration = mink.Configuration(model)
        # Define tasks
        self.posture_task = mink.PostureTask(model=model,
                                             cost=1e-3) #1e-2
        self.end_effector_task = mink.FrameTask(
            frame_name=site_name,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )

        self.robosuite_home_qpos = np.array([0, np.pi / 16.0, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.2, np.pi / 4])

        # rng to sample robot ee initial pose
        if hasattr(self._env.rng, 'spawn'):
            # NumPy >= 1.25
            self.rng = self._env.rng.spawn(1)[0]
        else:
            # NumPy < 1.25
            seed = self._env.rng.integers(0, 2**32 - 1)
            # print("TEST SEED:", seed)

            self.rng = np.random.default_rng(seed)
        
    def _change_camera_pose(self):
        pass

    # def _process_privileged_obs(
    #     self,
    #     obs: Dict[str, Any],
    #     out: Dict[str, Dict[str, np.ndarray]],
    # ) -> Dict[str, Dict[str, np.ndarray]]:
    #     """Subclass hook for adding task-specific privileged observations."""
    #     return out

    def _extract_privileged_from_raw_obs(
        self,
        obs: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """Extract numeric non-image, non-robot proprio keys from robosuite raw observations."""
        privileged: Dict[str, np.ndarray] = {}
        for key, value in obs.items():
            # print("KEY:", key)
            if key.endswith("_image"):
                continue
            if key.startswith("robot0_"):
                continue

            try:
                arr = np.asarray(value, dtype=np.float32)
            except (TypeError, ValueError):
                continue

            if arr.size == 0:
                continue
            privileged[key] = arr.copy()
            # print("key, arr:", key, arr)
        return privileged

    def _compute_desired_reset_qpos(self, reset_pose=None):
        random_reset = self.config.RANDOM_RESET
        if isinstance(random_reset, (tuple, list, np.ndarray)):
            random_reset = bool(np.asarray(random_reset).reshape(-1)[0])

        if reset_pose is None:
            if random_reset:  # randomize reset position in xy plane
                reset_pose = self.resetpos.copy()
                reset_pose[:2] += self.rng.uniform(
                    -self.config.RANDOM_XY_RANGE, self.config.RANDOM_XY_RANGE, (2,)
                )
                euler_random = self.config.RESET_POSE[3:].copy()
                euler_random[-1] += self.rng.uniform(
                    -self.config.RANDOM_RZ_RANGE, self.config.RANDOM_RZ_RANGE
                )

                reset_pose[3:] = R.from_euler('xyz', euler_random).as_quat()
            else:
                reset_pose = self.resetpos.copy()

        self.last_reset_pose = reset_pose.copy()
        # print("CURRENT QPOS:", R.from_quat(self._get_tcp_pose(frame="base")[3:]).as_euler("xyz"))
        # print("RESET POSE:", R.from_quat(reset_pose[3:]).as_euler("xyz"))

        # print("CURRENT TCP POSE:", self._get_tcp_pose(frame="base"))
        # print("RESET TCP POSE:", reset_pose)

        # Convert reset_pose from base frame to world frame
        pos_base = reset_pose[:3]
        rot_base = R.from_quat(reset_pose[3:]).as_matrix()

        pos_world = self.base_to_world[:3, :3] @ pos_base + self.base_to_world[:3, 3]
        rot_world = self.base_to_world[:3, :3] @ rot_base

        target_so3 = mink.SO3.from_matrix(rot_world)
        target_se3 = mink.SE3.from_rotation_and_translation(target_so3, pos_world)
        
        
        # print("IK starts with qpos:", self._env.sim.data.qpos)
        # print("IK starts with qpos:", self.robosuite_home_qpos)

        full_qpos = self._env.sim.data.qpos.copy()
        # print("FULL QPOS:", full_qpos)
        # print("IDX:", self._env.robots[0]._ref_joint_pos_indexes)
        # full_qpos[self._env.robots[0].joint_indexes] = self.robosuite_home_qpos
        full_qpos[self._env.robots[0]._ref_joint_pos_indexes] = self.robosuite_home_qpos
        self.ik_configuration.update(full_qpos)
        self.posture_task.set_target_from_configuration(self.ik_configuration)
        self.end_effector_task.set_target(target_se3)
        converged = converge_ik(
                self.ik_configuration,
                [self.end_effector_task, self.posture_task],
                1e-2, #try 0.02
                self.config.SOLVER,
                self.config.POS_THRESHOLD,
                self.config.ORI_THRESHOLD,
                self.config.MAX_ITERS,
        )
        arm_joint_ids = self._env.robots[0]._ref_joint_pos_indexes
        if not converged:
            print("WARNING: IK reset did not converge. Falling back to current arm joint positions.")
            return self._env.sim.data.qpos[arm_joint_ids].copy()

        self._env.sim.data.qpos[arm_joint_ids] = self.ik_configuration.q[arm_joint_ids]
        return self.ik_configuration.q[arm_joint_ids]


    def get_env(self):
        return self._env
    
    def allow_retry_from_same_state(self):
        self._allow_retry_from_same_state = True

    #TODO: fix gripper
    # gripper_binary_state = 1 is closed, 0 is open
    def _reset_gripper_to_open(self):
        self.gripper_binary_state = 1 #from close to open
        self.curr_ee_pose = self._get_tcp_pose(frame="base")
        self._run_gripper_command(1, mode="binary")
    
    def _reset_gripper_to_close(self):
        self.gripper_binary_state = 0 #from open to close
        self.curr_ee_pose = self._get_tcp_pose(frame="base")
        self._run_gripper_command(-1, mode="binary")

    def _run_gripper_command(self, pos: float, mode="binary"):
        def command_gripper_n_steps(n_steps: int):
            if self.config.SLEEP:
                for i in range(n_steps):
                    self._env.timestep -= 1
                    self._env.step(gripper_action)
                    time.sleep(max(0, 0.04 * (i + 1) - (time.time() - start_time)))
            else:
                for _ in range(n_steps):
                    self._env.timestep -= 1
                    self._env.step(gripper_action)        

        if mode == "binary": # based on franka_env: _send_gripper_command
            start_time = time.time()
            if (
                pos <= -self.config.BINARY_GRIPPER_THREASHOLD
                and self.gripper_binary_state == 0
            ):  # close gripper
                # print("Closing gripper")
                # 1 is close, -1 is open #flipped sign because of the way gripper is controlled in robosuite
                gripper_action = np.concatenate([self.curr_ee_pose[:3], R.from_quat(self.curr_ee_pose[3:]).as_rotvec(), [1]])
                command_gripper_n_steps(15) #15
                self.gripper_binary_state = 1
            elif (
                pos >= self.config.BINARY_GRIPPER_THREASHOLD
                and self.gripper_binary_state == 1
            ):  # open gripper
                # print("Opening gripper")
                # 1 is close, -1 is open #flipped sign because of the way gripper is controlled in robosuite
                gripper_action = np.concatenate([self.curr_ee_pose[:3], R.from_quat(self.curr_ee_pose[3:]).as_rotvec(), [-1]])
                command_gripper_n_steps(15) #15
                self.gripper_binary_state = 0
            # print("TIME TAKEN:", time.time() - start_time)
            return
        elif mode == "continuous": # based on panda_pick_gym_env: step
            raise NotImplementedError("Continuous gripper control is optional")

    def _check_success(self, info: Dict[str, Any]) -> bool:
        if "success" in info:
            return bool(info["success"])
        if hasattr(self._env, "_check_success"):
            return bool(self._env._check_success())
        return False
    
    def _get_tcp_pose(self, frame: str = "base") -> np.ndarray:
        ref_id = self._env.sim.model.site_name2id("gripper0_right_grip_site")
        ee_pos_world = np.array(self._env.sim.data.site_xpos[ref_id])
        ee_matrix_world = np.array(self._env.sim.data.site_xmat[ref_id].reshape([3, 3]))
        if frame == "base":
            ee_pos_base = self.world_to_base[:3, :3] @ ee_pos_world + self.world_to_base[:3, 3]
            ee_quat_base = R.from_matrix(self.world_to_base[:3, :3] @ ee_matrix_world).as_quat()
            return np.concatenate([ee_pos_base, ee_quat_base])
        # in world frame
        # return np.concatenate([ee_pos_world, ee_matrix_world])

    #TODO: will be different for states vs images observation
    # The obs["robot0_eef_quat"] has a bug (using body instead of site in robosuite)
    # _hand_pose, _hand_quat etc (https://robosuite.ai/docs/simulation/robot.html) returns pose of the wrist, not tcp (in base frame)
    def _process_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"state": {}, "images": {}}

        out["state"]["tcp_pose"] = self._get_tcp_pose(frame="base")
        # print("TCP POS:", out["state"]["tcp_pose"][:3])
        # print("TCP ORN:", R.from_quat(out["state"]["tcp_pose"][3:]).as_euler("xyz"))
        # print("BIN 1 POS:", self._env.sim.data.body("bin1").xpos)
        # print("BIN 2 POS:", self._env.sim.data.body("bin2").xpos)
        # print("Z OFFSET:", self._env.z_offset)
        #print("TABLE POS:", self._env.sim.data.body("table").xpos)
        # See what the environment thinks the table height is
        #print("table_offset:", self._env.model.mujoco_arena.table_offset)
        #print("table_full_size:", self._env.model.mujoco_arena.table_full_size)
        # print("RoundNut_main POS:", self._env.sim.data.body("RoundNut_main").xpos)
        # print("RoundNut_main POS CHECK:", self._env.sim.data.get_joint_qpos(self._env.nuts[1].joints[0]))
        # print("peg1 POS:", self._env.sim.data.body("peg1").xpos)
        # print("peg2 POS:", self._env.sim.data.body("peg2").xpos)
        # print("ORN IS:", R.from_quat(out["state"]["tcp_pose"][3:]).as_euler("xyz"))
        # print("LIMITS ARE:", self.rpy_bounding_box.low, self.rpy_bounding_box.high)

        # bin_x_half = self._env.model.mujoco_arena.table_full_size[0] / 2 - 0.05
        # bin_y_half = self._env.model.mujoco_arena.table_full_size[1] / 2 - 0.05
        # print("BIN 1 x half:", bin_x_half)
        # print("BIN 1 y half:", bin_y_half)

        # tcp velocity in world frame; is the same for base frame because of no rotation between world and base frames
        lin_vel = self._env.sim.data.get_site_xvelp("gripper0_right_grip_site") 
        ang_vel = self._env.sim.data.get_site_xvelr("gripper0_right_grip_site")
        out["state"]["tcp_vel"] = np.concatenate([lin_vel, ang_vel]).astype(np.float32)
       
        if self.config.GRIPPER_IS_ACTIVE: #TODO: fix gripper
            # out["state"]["gripper_pose"] = np.array([self.gripper_binary_state], dtype=np.float32)
            gripper_pose = (obs["robot0_gripper_qpos"][0] + obs["robot0_gripper_qpos"][3]) / 2
            out["state"]["gripper_pose"] = 1.0 - np.clip(gripper_pose, 0.0, 0.8) / 0.8 
            # print_green(f"GRIPPER POSE: {out['state']['gripper_pose']}")
            # print_green(f"GRIPPER BINARY STATE: {self.gripper_binary_state}")
            # print_green(f"GRIPPER QPOS: {obs['robot0_gripper_qpos']}")

        wrist_force = self._env.robots[0].ee_force["right"]
        out["state"]["wrist_force"] = wrist_force.astype(np.float32)
        wrist_torque = self._env.robots[0].ee_torque["right"]
        out["state"]["wrist_torque"] = wrist_torque.astype(np.float32)

        for key in self.config.REALSENSE_CAMERAS:
            camera_name = self.config.CAMERA_NAME_MAP[key]
            img_key = f"{camera_name}_image"
            img = obs.get(img_key)
            if img is None:
                raise ValueError(f"Was not able to get image {img_key}")

            if key in self.config.IMAGE_CROP:
                img = self.config.IMAGE_CROP[key](img)
                target_hw = self.observation_space["images"][key].shape[:2][::-1]
                img = cv2.resize(img, target_hw)

            out["images"][key] = img.astype(np.uint8)

        if self.save_video:
            obs_for_video = {}
            for key in self.config.REALSENSE_CAMERAS:
                bgr_img = cv2.cvtColor(out["images"][key], cv2.COLOR_RGB2BGR)
                obs_for_video[key] = bgr_img
            self.recording_frames.append(obs_for_video)

        if self.privileged_obs:
            out = self._process_privileged_obs(obs, out) #TODO

        self.curr_ee_pose = out["state"]["tcp_pose"].copy()

        if self.render_mode == "human":
            self._render_step_overlay()

        return out
    
    # def _process_privileged_obs(
    #     self,
    #     obs: Dict[str, Any],
    #     out: Dict[str, Dict[str, np.ndarray]],
    # ) -> Dict[str, Dict[str, np.ndarray]]:
    #     out["state"]["privileged"] = self._extract_privileged_from_raw_obs(obs)["object-state"]
    #     return out
    
    def save_video_recording(self):
        # try:
        if len(self.recording_frames):
            if not os.path.exists(self.video_path):
                os.makedirs(self.video_path)

            # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
            # print("TIMESTAMP:", timestamp)
            for camera_key in self.recording_frames[0].keys():
                video_path = f"{self.video_path}/{camera_key}_{timestamp}.mp4"

                # Get the shape of the first frame for this camera
                first_frame = self.recording_frames[0][camera_key]
                height, width = first_frame.shape[:2]

                video_writer = cv2.VideoWriter(
                    video_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    10,
                    (width, height),
                )

                for frame_dict in self.recording_frames:
                    video_writer.write(frame_dict[camera_key])

                video_writer.release()
                # print(f"Saved video for camera {camera_key} at {video_path}")

        self.recording_frames.clear()
        # except Exception as e:
        #     print(f"Failed to save video: {e}")
    
    def _reset_env(self, reset_pose=None): #TODO: make robot reset deterministic (set initialization_noise magnitude to 0??)
        self._env.robots[0].init_qpos = self._compute_desired_reset_qpos(reset_pose) #TODO
        return self._env.reset()

    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:

        # print_green(f"CUBE SIZE: {self._env.cube.size}")
        # print_green(f"CUBE A SIZE: {self._env.cubeA.size}")
        # print_green(f"CUBE B SIZE: {self._env.cubeB.size}")

        if self.save_video:
            # print_green("SAVE VIDEO")
            self.save_video_recording()
        
        # print("RESET OPTIONS:", options)

        if self._allow_retry_from_same_state:
            retry_from_same_state = (
                options is not None
                and options.get("retry_from_same_state", False))
            if retry_from_same_state:
                # print("RETRY RESET")
                self._env.rng.bit_generator.state = copy.deepcopy(self._saved_env_rng_state)
                self.rng.bit_generator.state = copy.deepcopy(self._saved_robot_rng_state)

            else: #fresh reset
                # print("FRESH RESET")
                self._saved_env_rng_state = copy.deepcopy(self._env.rng.bit_generator.state)
                self._saved_robot_rng_state = copy.deepcopy(self.rng.bit_generator.state)
        
        if options is not None:
            reset_pose = options["reset_pose"]
            # test_o_1 = self._reset_env(reset_pose)
            self._reset_env(reset_pose)
        else:
            # test_o_1 = self._reset_env()
            self._reset_env()
        # test_o_1 = self._process_obs(test_o_1)

        if self.config.GRIPPER_IS_ACTIVE:
            if self.config.INITIAL_GRIPPER_OPEN:
                # print("RESET GRIPPER TO OPEN")
                self._reset_gripper_to_open()
            else:
                # print("RESET GRIPPER TO CLOSE")
                self._reset_gripper_to_close()
        reset_out = self._env._get_observations()
        
        raw_obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        obs = self._process_obs(raw_obs)
        # print_green(f"CUBE POS: {self._env.sim.data.body('cube_main').xpos}")
        # print_green(f"CUBE QUAT: {self._env.sim.data.body('cube_main').xquat}")
        # print_green(f"EE POSE: {obs['state']['tcp_pose']}")
        self.curr_path_length = 0

        # print("COMPARE OBS1 AND OBS2") #TODO: velocity will be non-zero if we run gripper action, need to change gripper with joints
        # print("OBS1 pose:", test_o_1["state"]["tcp_pose"])
        # print("OBS2 pose:", obs["state"]["tcp_pose"])
        # print("OBS1 vel:", test_o_1["state"]["tcp_vel"])
        # print("OBS2 vel:", obs["state"]["tcp_vel"])


        if self.render_mode == "human":
            self._env.render()
        
        return obs, {"succeed": False}

    def get_base_to_world_transformation_matrix(self) -> np.ndarray:
        # Get robot base position in world frame
        base_body_name = self._env.robots[0].robot_model.root_body
        base_pos_world = self._env.sim.data.get_body_xpos(base_body_name).copy()
        base_quat_world = self._env.sim.data.get_body_xquat(base_body_name).copy()
        #print("BASE POS WORLD:", base_pos_world)
        
        # MuJoCo uses scalar-first quaternion (w, x, y, z), convert to scalar-last (x, y, z, w)
        base_quat_world = scalar_first_to_scalar_last(base_quat_world)
        
        # Construct transformation matrix T_world_base (base pose in world frame)
        T_world_base = np.eye(4)
        T_world_base[:3, :3] = R.from_quat(base_quat_world).as_matrix()
        T_world_base[:3, 3] = base_pos_world

        return T_world_base
    
    def get_world_to_base_transformation_matrix(self) -> np.ndarray:
        return np.linalg.inv(self.get_base_to_world_transformation_matrix())
        
    def _clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box."""
        # print("POS BEFORE CLIP:", pose[:3])
        pose[:3] = np.clip(
            pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )
        # print("POS AFTER XYZ CLIP:", pose[:3])
        euler = R.from_quat(pose[3:]).as_euler("xyz")
        # print("EULER BEFORE CLIPPING:", euler)

        #Clip first euler angle separately due to discontinuity from pi to -pi
        sign = np.sign(euler[0])
        euler[0] = sign * (
            np.clip(
                np.abs(euler[0]),
                self.rpy_bounding_box.low[0],
                self.rpy_bounding_box.high[0],
            )
        )

        euler[1:] = np.clip(
            euler[1:], self.rpy_bounding_box.low[1:], self.rpy_bounding_box.high[1:]
        )
        # print("EULER AFTER CLIPPING:", euler)
        pose[3:] = R.from_euler("xyz", euler).as_quat()
        return pose

    # Robosuite controller:
    # https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/controllers/parts/controller.py
    # https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/controllers/parts/arm/osc.py
    def step(self, action: np.ndarray):
        # for cube_name in ["cubeA_main", "cubeB_main"]:
        # for cube_name in ["cube_main"]:
        #     cube_body_id = self._env.sim.model.body_name2id(cube_name)
        #     global_pos = self._env.sim.data.body_xpos[cube_body_id]
        #     print(f"Cube {cube_name} global position: {global_pos}")
        # print("TCP POSE:", self._get_tcp_pose(frame="base"))
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

        if self.render_mode == "human":
            self._env.render()

        step_info = dict(info)

        qacc = self._env.sim.data.qacc
        if not np.all(np.isfinite(qacc)) or not np.all(np.isfinite(
            np.concatenate([np.ravel(obs["state"]["tcp_pose"]),
                            np.ravel(obs["state"]["privileged"])])
        )):
            # sim went unstable — terminate this episode, don't poison the dataset
            done = True
            reward = self.config.reward_neg
            step_info["sim_unstable"] = 1
            step_info["succeed"] = 0
            return obs, float(reward), done, False, step_info

        step_info["succeed"] = int(success)
        # print_green(f"GRIPPER POSE MORE: {obs['state']['gripper_pose']}")
        # print_green(f"TEST BASE ENV OBS: {obs}")
        return obs, float(reward), done, False, step_info
    
    def _render_step_overlay(self):
        """Overlay remaining-step countdown on the MuJoCo human viewer."""
        steps_left = self.config.MAX_EPISODE_LENGTH - self.curr_path_length
        label = f"Steps left: {steps_left}"

        # MjviewerRenderer wraps the native passive viewer in .viewer
        mjviewer_renderer = getattr(self._env, "viewer", None)
        if mjviewer_renderer is None:
            return
        passive_viewer = getattr(mjviewer_renderer, "viewer", None)
        if passive_viewer is None:
            return  # passive viewer not yet initialised (first step)

        try:
            import mujoco
            scn = passive_viewer.user_scn
            scn.ngeom = 0
            g = scn.geoms[0]
            mujoco.mjv_initGeom(
                g,
                mujoco.mjtGeom.mjGEOM_LABEL,
                np.zeros(3),
                np.zeros(3),
                np.eye(3).flatten(),
                np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32),
            )
            g.pos[:] = [0.0, 0.0, 1.5]
            g.label = label
            scn.ngeom = 1
            passive_viewer.sync()
        except Exception as e:
            print(f"[overlay] failed: {e}")

    def close(self):
        self._env.close()

    def tune_camera(self, cam_name: str = "robot0_robotview"):
        """
        Interactive camera tuner using OpenCV sliders.
        Call this method after environment initialization to adjust camera pose.
        Press 'p' to print current values, 'q' or ESC to quit.
        """
        # while True:
        #     pass
        import cv2
        from scipy.spatial.transform import Rotation as R
        
        cam_id = self._env.sim.model.camera_name2id(cam_name)

        # Read current camera position and orientation
        init_pos = self._env.sim.model.cam_pos[cam_id].copy()
        init_quat_wxyz = self._env.sim.model.cam_quat[cam_id].copy()
        
        # Convert initial quaternion to rotation matrix for delta rotations
        init_quat_xyzw = T.convert_quat(init_quat_wxyz, to="xyzw")
        init_R_mat = T.quat2mat(init_quat_xyzw)
        
        # Position sliders initialized to current camera position
        # pos = (slider - 200) / 100 => slider = pos * 100 + 200
        init_x_slider = int(np.clip(init_pos[0] * 100 + 200, 0, 400))
        init_y_slider = int(np.clip(init_pos[1] * 100 + 200, 0, 400))
        init_z_slider = int(np.clip(init_pos[2] * 100 + 200, 0, 400))
        
        # Rotation sliders start at center (180 = 0° delta from original)
        init_pitch_slider = 180
        init_yaw_slider = 180
        init_roll_slider = 180

        print(f"Initial camera pos: {init_pos}")
        print(f"Initial camera quat (wxyz): {init_quat_wxyz}")
        print("Rotation sliders = delta from original (center = no change)")

        window_name = "Real-Time Camera Tuner"
        cv2.namedWindow(window_name)
        def nothing(x): pass

        # Create Translation Sliders initialized to current camera position
        cv2.createTrackbar("X (m)", window_name, init_x_slider, 400, nothing)
        cv2.createTrackbar("Y (m)", window_name, init_y_slider, 400, nothing)
        cv2.createTrackbar("Z (m)", window_name, init_z_slider, 400, nothing)

        # Create Rotation Sliders (delta from original, center=0°)
        cv2.createTrackbar("dPitch", window_name, init_pitch_slider, 360, nothing)
        cv2.createTrackbar("dYaw", window_name, init_yaw_slider, 360, nothing)
        cv2.createTrackbar("dRoll", window_name, init_roll_slider, 360, nothing)

        print("--- TUNER ACTIVE ---")
        print("Drag sliders to move camera. Press 'p' to print, 'q' or ESC to quit.")

        self._env.reset()
        
        # Track position and rotation sliders separately
        prev_pos_sliders = (init_x_slider, init_y_slider, init_z_slider)
        prev_rot_sliders = (init_pitch_slider, init_yaw_slider, init_roll_slider)
        
        # Start with original camera pose
        current_pos = init_pos.copy()
        current_quat = init_quat_wxyz.copy()

        while True:
            x_slider = cv2.getTrackbarPos("X (m)", window_name)
            y_slider = cv2.getTrackbarPos("Y (m)", window_name)
            z_slider = cv2.getTrackbarPos("Z (m)", window_name)
            pitch_slider = cv2.getTrackbarPos("dPitch", window_name)
            yaw_slider = cv2.getTrackbarPos("dYaw", window_name)
            roll_slider = cv2.getTrackbarPos("dRoll", window_name)

            # x_slider = 188
            # y_slider = 165
            # z_slider = 232
            # pitch_slider = 250
            # yaw_slider = 14
            # roll_slider = 182
            
            curr_pos_sliders = (x_slider, y_slider, z_slider)
            curr_rot_sliders = (pitch_slider, yaw_slider, roll_slider)
            
            # Only update position if position sliders changed
            if curr_pos_sliders != prev_pos_sliders:
                print("Updating position")
                pos_x = (x_slider - 200) / 100.0
                pos_y = (y_slider - 200) / 100.0
                pos_z = (z_slider - 200) / 100.0
                current_pos = np.array([pos_x, pos_y, pos_z])
                prev_pos_sliders = curr_pos_sliders
            
            # Only update rotation if rotation sliders changed
            if curr_rot_sliders != prev_rot_sliders:
                print("Updating rotation")
                # Delta angles from center (slider=180 means 0° delta)
                d_pitch = np.deg2rad(pitch_slider - 180)
                d_yaw = np.deg2rad(yaw_slider - 180)
                d_roll = np.deg2rad(roll_slider - 180)
                
                # Build rotation matrices for each axis
                c_p, s_p = np.cos(d_pitch), np.sin(d_pitch)
                c_y, s_y = np.cos(d_yaw), np.sin(d_yaw)
                c_r, s_r = np.cos(d_roll), np.sin(d_roll)
                
                # Pitch around X-axis
                R_pitch = np.array([[1, 0, 0], [0, c_p, -s_p], [0, s_p, c_p]])
                # Yaw around Y-axis
                R_yaw = np.array([[c_y, 0, s_y], [0, 1, 0], [-s_y, 0, c_y]])
                # Roll around Z-axis
                R_roll = np.array([[c_r, -s_r, 0], [s_r, c_r, 0], [0, 0, 1]])
                
                # Apply delta rotations to initial rotation
                R_final = init_R_mat @ R_pitch @ R_yaw @ R_roll
                # print("R_final:", R_final)
                # print("EULER xyz:", T.mat2euler(R_final))
                # print("MAT:", T.euler2mat(T.mat2euler(R_final)))
                
                quat_xyzw = T.mat2quat(R_final)
                current_quat = T.convert_quat(quat_xyzw, to="wxyz")
                prev_rot_sliders = curr_rot_sliders
            
            self._env.sim.model.cam_pos[cam_id] = current_pos
            self._env.sim.model.cam_quat[cam_id] = current_quat
            print("CAMERA POS:", self._env.sim.model.cam_pos[cam_id])
            print("CAMERA QUAT:", self._env.sim.model.cam_quat[cam_id])

            # Step with zero action to render
            # obs, _, _, _ = self._env.step(np.zeros(self._env.action_dim))
            action = np.zeros(7)
            action = np.asarray(action, dtype=np.float32) #delta action (euler angles in base frame)
            action = np.clip(action, self.action_space.low, self.action_space.high)

            # # control gripper
            # if self.config.GRIPPER_IS_ACTIVE:
            #     gripper_action = action[6] * self.config.ACTION_SCALE[2]
            #     self._run_gripper_command(gripper_action)

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
            if self.config.GRIPPER_IS_ACTIVE: #TODO: fix gripper
                action = np.concatenate([action, [-1]]) #gripper action is controlled separately
            step_out = self._env.step(action) #in base frame
            #
            obs, _, _, _ = step_out
            img_rgb = obs[f"{cam_name}_image"]
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name, img_bgr)

            key = cv2.waitKey(50) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('p'):
                print("\n=== COPY THESE VALUES ===")
                print(f"cam_pos  = [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
                print(f"cam_quat = [{current_quat[0]:.4f}, {current_quat[1]:.4f}, {current_quat[2]:.4f}, {current_quat[3]:.4f}]")
                print("=========================\n")

        cv2.destroyAllWindows()