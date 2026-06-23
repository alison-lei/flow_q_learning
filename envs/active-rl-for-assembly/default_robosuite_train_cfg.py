from re import T
from turtle import window_width
from modified_wrappers import (
    RLExpertInterventionRG,
    RelativeFrameRG,
    GripperPenaltyWrapper,
    GripperPenaltyWrapperRG,
    SpacemouseInterventionHG,
    SpacemouseInterventionRG,
    ActionFillWrapper,
    RemoveTCPOrientationWrapper,
)
# do i need to import this, or delete this
from serl_launcher.wrappers.chunking import ChunkingWrapper
from modified_wrappers import SERLObsWrapper

from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.wrappers import GripperCloseEnv, Quat2EulerWrapper

from train.configs.default_train_cfg import DefaultTrainingConfig
from sim_env.configs.default_robosuite_cfg import DefaultRobosuiteConfig
from sim_env.envs.robosuite_base_env import RobosuiteBaseEnv

from heuristics import (
    RandomRobotGatedHeuristicGaussianLength,
    RandomRobotGatedHeuristicEmpirical,
    RandomRobotGatedHeuristicEmpiricalCurriculum,
    RobotGatedHeuristicQValueSlopeRecover,
    HumanGatedHeuristicWithButtonPresses,
    RobotGatedHeuristicQValueSlopeFinishAfterStart,
)

from sim_env.envs.modified_wrappers import Quat2EulerWrapperWithExpert, SERLObsWrapperWithExpert

def print_green(x):
    return print("\033[92m {}\033[00m".format(x))

class DefaultRobosuiteTrainConfig(DefaultTrainingConfig):
    max_traj_length = 100
    discount = 0.97
    checkpoint_period = 2000
    buffer_period = 0
    synchronized_checkpoint_period = 0

    image_keys = ["wrist_1", "side_policy_256"]
    #proprio_keys = ["tcp_pose", "tcp_vel", "gripper_pose", "wrist_force", "wrist_torque"] #for most tasks
    # proprio_keys = ["tcp_pose", "tcp_vel", "wrist_force", "wrist_torque"] #for wipe
    proprio_keys = ["tcp_pose", "gripper_pose"] #for flow
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"
    task_desc = ""
    octo_path = "hf://rail-berkeley/octo-small"

    batch_size = 256
    grasp_penalty = -0.2

    state_expert = True #False #True
    scripted_expert = True #False#True
    RL_expert = False

    intervention_heuristic = RobotGatedHeuristicQValueSlopeRecover #HumanGatedHeuristicWithButtonPresses #RobotGatedHeuristicQValueSlopeRecover #RobotGatedHeuristicQValueSlopeFinishAfterStart #HumanGatedHeuristicWithButtonPresses #RobotGatedHeuristicQValueSlopeRecover #RandomRobotGatedHeuristicEmpiricalCurriculum #RandomRobotGatedHeuristicGaussianLength, RobotGatedHeuristicQValueSlopeRecover, HumanGatedHeuristicWithButtonPresses
    if intervention_heuristic == RandomRobotGatedHeuristicGaussianLength:
        prob_from_robot_to_expert = None
        intervention_length_mean = None
        intervention_length_std = None
    elif intervention_heuristic == RobotGatedHeuristicQValueSlopeRecover:
        window_size = 20
        threshold = 0
        min_intervention_steps = 15
        max_interventions_per_episode = None
        clear_q_values_on_intervention_end = False
    elif intervention_heuristic == RandomRobotGatedHeuristicEmpirical or intervention_heuristic == RandomRobotGatedHeuristicEmpiricalCurriculum:
        window_size = 10
        min_intervention_steps = 15
        chunk_size = 500


    def get_environment(
        self,
        fake_env=False,
        save_video=False,
        classifier=False,
        render_mode="human",
        eval_env=False,
        env_seed=None,
        # init_heuristic=False,
        video_path="./videos",
        obs_horizon=None,
        reward_pos=None,
        reward_neg=None,
        # Indices into the full EE-frame action [x, y, z, rx, ry, rz, (gripper)]
        # to train on; remaining dims are zero-filled by ActionFillWrapper so
        # the policy/expert/demos live in this reduced space. None -> full action.
        active_action_indices=None,
        # If True, strip orientation from tcp_pose (keep xyz) and tcp_vel (keep
        # linear vel) in the proprio observation, reducing the flattened state
        # dim. False -> keep the full pose & velocity.
        remove_tcp_orientation: bool = False,
        allow_retry_from_same_state=False,
        env_config: DefaultRobosuiteConfig = DefaultRobosuiteConfig,
        env_class: RobosuiteBaseEnv = RobosuiteBaseEnv,
        env_rg_class: RobosuiteBaseEnv = RobosuiteBaseEnv,
        human_action_frame: str = "base",
        privileged_obs: bool = False,
    ):
        if env_seed is not None:
            env_config.SEED = env_seed
        if reward_pos is not None:
            env_config.reward_pos = reward_pos
        if reward_neg is not None:
            env_config.reward_neg = reward_neg

        if self.state_expert:
            self.proprio_keys.append("privileged")
        print("PROPRIO KEYS IN CONFIG ARE", self.proprio_keys)

        # if init_heuristic:
        if self.intervention_heuristic is not None:
            env = env_rg_class(
                config=env_config,
                fake_env=fake_env,
                render_mode=render_mode,
                save_video=save_video,
                video_path=video_path,
                privileged_obs=privileged_obs,
            )
            if allow_retry_from_same_state:
                env.allow_retry_from_same_state()
            if not fake_env and not eval_env:
                if self.RL_expert or self.scripted_expert:
                    env = RLExpertInterventionRG(env)
                    env = SpacemouseInterventionHG(env, frame=human_action_frame) #just to debug
                elif self.intervention_heuristic is not None:
                    env = SpacemouseInterventionRG(env, frame=human_action_frame)
                else:
                    env = SpacemouseInterventionHG(env, frame=human_action_frame)

            env = RelativeFrameRG(env)
            if active_action_indices is not None:
                env = ActionFillWrapper(env, active_indices=active_action_indices)
            env = Quat2EulerWrapper(env)
            if remove_tcp_orientation:
                env = RemoveTCPOrientationWrapper(env)
            env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
            env = ChunkingWrapper(env, obs_horizon=obs_horizon, act_exec_horizon=None)
            # if env_config.GRIPPER_IS_ACTIVE: # need to be careful with how gripper actions are handled
            #     env = GripperPenaltyWrapperRG(env, penalty=self.grasp_penalty) #will impact only HIL-SERL, not ConRFT
            return env

        env = env_class(
            config=env_config,
            fake_env=fake_env,
            render_mode=render_mode,
            save_video=save_video,
            video_path=video_path,
            privileged_obs=privileged_obs,
        )
        if allow_retry_from_same_state:
            env.allow_retry_from_same_state()
        if not fake_env and not eval_env:
            env = SpacemouseInterventionHG(env, frame=human_action_frame)
        env = RelativeFrame(env)
        if active_action_indices is not None:
            env = ActionFillWrapper(env, active_indices=active_action_indices)
        env = Quat2EulerWrapper(env)
        if remove_tcp_orientation:
            env = RemoveTCPOrientationWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=obs_horizon, act_exec_horizon=None)
        # if env_config.GRIPPER_IS_ACTIVE: # need to ba careful with how gripper actions are handled
        #     env = GripperPenaltyWrapper(env, penalty=self.grasp_penalty) #will impact only HIL-SERL, not ConRFT
        return env