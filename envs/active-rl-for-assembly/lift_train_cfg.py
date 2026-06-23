from default_robosuite_train_cfg import DefaultRobosuiteTrainConfig
from lift_env_cfg import LiftEnvConfig
from lift_env import PandaLiftGymEnv, PandaLiftGymEnvRG

class TrainConfig(DefaultRobosuiteTrainConfig):
    max_traj_length = LiftEnvConfig.MAX_EPISODE_LENGTH
    discount = 0.98
    checkpoint_period = 0 #14000
    buffer_period = 0
    synchronized_checkpoint_period = 2000 #0
    task_desc = "pick up the red block"
    octo_path = "hf://rail-berkeley/octo-small" # "hf://rail-berkeley/octo-small"
                                                # "hf://rail-berkeley/octo-small-1.5"

    def get_environment(self, **kwargs):
        return super().get_environment(**kwargs,
            env_config=LiftEnvConfig,
            env_class=PandaLiftGymEnv,
            env_rg_class=PandaLiftGymEnvRG,
            human_action_frame="ee",
            privileged_obs=True,)
    
    