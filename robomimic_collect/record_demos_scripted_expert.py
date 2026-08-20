# Collect demonstration data using a scripted expert.
#
# Usage:
#   python -m sim_env.collect_data.record_demos_scripted_expert \
#       --exp_name lift --successes_needed 50 --env_seed 0
#   python -m sim_env.collect_data.record_demos_scripted_expert \
#       --exp_name lift --successes_needed 50 --apply_noise --noise_seed 42

import os
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time

from train.mappings import CONFIG_MAPPING
from sim_env.experts import get_scripted_expert
from sim_env.experts.expert_utils import strip_privileged, compute_privileged_indices

from external.conrft.examples.data_util import (
    add_mc_returns_to_trajectory,
    add_embeddings_to_trajectory,
    add_next_embeddings_to_trajectory,
)
from octo.model.octo_model import OctoModel

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")
flags.DEFINE_float("reward_scale", 1.0, "reward_scale")
flags.DEFINE_float("reward_bias", 0.0, "reward_bias")
flags.DEFINE_integer("env_seed", 0, "Environment seed.")
flags.DEFINE_boolean("apply_noise_perception", False, "Apply per-episode perception noise.")
flags.DEFINE_boolean("apply_noise_action", False, "Apply per-step action noise.")
flags.DEFINE_integer("noise_seed", None, "Noise RNG seed for reproducibility.")
flags.DEFINE_boolean("no_render", False, "Disable rendering.")
flags.DEFINE_boolean("no_retry", False, "Disable retry from same state on failure.")
flags.DEFINE_boolean("verbose", False, "Print expert phase transitions.")
flags.DEFINE_boolean("forward_only", False, "Wipe expert only: forward-only passes (default forward/backward).")
flags.DEFINE_boolean("sleep", False, "")
flags.DEFINE_boolean("save_video", False, "Save video.")

obs_horizon = 2 

from sim_env.experts.expert_utils import print_green

def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    # train_proprio_keys = list(config.proprio_keys)
    # config.proprio_keys = config.expert_proprio_keys

    render_mode = "rgb_array" if FLAGS.no_render else "human"
    allow_retry = not FLAGS.no_retry
    save_video = FLAGS.save_video

    try:
        print("RETRY IS ALLOWED")
        env = config.get_environment(
            fake_env=False,
            save_video=save_video,
            video_path=f"./videos/CORL_data/scripted_expert_data_{FLAGS.exp_name}",
            classifier=False,
            render_mode=render_mode,
            env_seed=FLAGS.env_seed,
            # init_heuristic=True,
            # with_gripper_close_env=False,
            obs_horizon=obs_horizon,
            reward_pos=10,
            reward_neg=-0.05,
            allow_retry_from_same_state=allow_retry,
            active_action_indices=[0,1,2] if FLAGS.exp_name == "wipe" else None
        )
    except TypeError:
        env = config.get_environment(
            fake_env=False,
            save_video=save_video,
            video_path=f"./videos/CORL_data/scripted_expert_data_{FLAGS.exp_name}",
            classifier=False,
            render_mode=render_mode,
            env_seed=FLAGS.env_seed,
            # init_heuristic=True,
            # with_gripper_close_env=False,
            obs_horizon=obs_horizon,
            reward_pos=10,
            reward_neg=-0.05,
            active_action_indices=[0,1,2] if FLAGS.exp_name == "wipe" else None
        )

    env.config.SLEEP = FLAGS.sleep

    # Compute privileged slice indices in the flattened (alphabetically sorted) state
    priv_shape = env.unwrapped.get_privileged_shape()
    priv_start, priv_end, total_dim = compute_privileged_indices(
        # config.expert_proprio_keys, priv_shape
        config.proprio_keys, priv_shape
    )
    print(f"Privileged obs at state[{priv_start}:{priv_end}], "
          f"will strip before saving (expert: {total_dim}d -> train: {total_dim - (priv_end - priv_start)}d)")

    expert = get_scripted_expert(
        task_name=FLAGS.exp_name,
        proprio_keys=config.proprio_keys,
        env=env,
        obs_horizon=obs_horizon,
        privileged_shape=priv_shape,
        apply_noise_perception=FLAGS.apply_noise_perception,
        apply_noise_action=FLAGS.apply_noise_action,
        forward_only=FLAGS.forward_only,
        seed=FLAGS.noise_seed,
        verbose=FLAGS.verbose,
    )

    model = OctoModel.load_pretrained(config.octo_path)
    tasks = model.create_tasks(texts=[config.task_desc])
    print("Model is:", config.octo_path)
    print("Task desc is:", config.task_desc)
    if FLAGS.apply_noise_perception:
        print(f"Perception noise enabled, seed={FLAGS.noise_seed}")
    if FLAGS.apply_noise_action:
        print(f"Action noise enabled, seed={FLAGS.noise_seed}")

    #for i in range(7):
    obs, info = env.reset()
    # obs, info = env.reset()
    last_reset_pose = env.last_reset_pose
    print("Reset done")

    transitions = []
    success_count = 0
    fail_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    trajectory = []
    returns = 0
    lengths = []

    while success_count < success_needed:
        actions = expert.sample_actions(observations=obs)
        print("ACTIONS: ", actions)
        print("ACTIONS SHAPE: ", actions.shape)
        action_dict = {"expert_acts": False, "action": actions} #TODO
        next_obs, rew, done, truncated, info = env.step(action_dict)
        returns += rew

        if "intervene_action" in info:
            actions = info["intervene_action"]
            print_green("RESET EXPERT")
            expert.reset()
        else:
            print("NO RESET EXPERT")
        
        if actions[-1] != 0:
            info["grasp_penalty"] = -0.02 #config.grasp_penalty #TODO: temp for expert data collection
        else:
            info["grasp_penalty"] = 0.0

        if np.all(actions == 0):
            print("WARNING: actions are zero")
        else:
            transition = copy.deepcopy(
                dict(
                    observations=strip_privileged(obs, priv_start, priv_end),
                    actions=actions,
                    next_observations=strip_privileged(next_obs, priv_start, priv_end),
                    rewards=rew,
                    masks=1.0 - done,
                    dones=done,
                    infos=info,
                    reset_pose=last_reset_pose,
                )
            )
            trajectory.append(transition)

        pbar.set_description(f"Return: {returns}")
        obs = next_obs

        if done:
            traj_len = len(trajectory)
            if info["succeed"]:
                trajectory = add_mc_returns_to_trajectory(
                    trajectory, config.discount, FLAGS.reward_scale,
                    FLAGS.reward_bias, reward_neg=-0.05, is_sparse_reward=True,
                )
                # trajectory = add_embeddings_to_trajectory( #TODO
                #     trajectory, model, tasks=tasks,
                # )
                # trajectory = add_next_embeddings_to_trajectory(trajectory)

                print(f"SUCCESS (length {traj_len})")
                for transition in trajectory:
                    transitions.append(copy.deepcopy(transition))
                success_count += 1
                lengths.append(traj_len)
                pbar.update(1)
            else:
                fail_count += 1
                print(f"FAILURE (length {traj_len})")

            trajectory = []
            returns = 0
            if info["succeed"]:
                obs, info = env.reset()
            else:
                obs, info = env.reset(
                    options={"reset_pose": last_reset_pose,
                             "retry_from_same_state": True}
                )
            if hasattr(expert, "reset"):
                expert.reset()
            last_reset_pose = env.last_reset_pose

    env.close()

    # Print stats
    print(f"\nCollection complete: {success_count} successes, {fail_count} failures")
    print(f"Trajectory lengths: {lengths}")
    print(f"Average length: {np.mean(lengths):.1f} "
          f"(std: {np.std(lengths):.1f}, "
          f"min: {np.min(lengths)}, max: {np.max(lengths)})")

    # Save
    if not os.path.exists("./demos/scripted_policy_no_noise_demos/"):
        os.makedirs("./demos/scripted_policy_no_noise_demos")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")

    noise_parts = []
    if FLAGS.apply_noise_perception:
        noise_parts.append("percnoise")
    if FLAGS.apply_noise_action:
        noise_parts.append("actnoise")
    noise_tag = "_" + "_".join(noise_parts) if noise_parts else ""
    file_name = (f"./demos/scripted_policy_no_noise_demos/{FLAGS.exp_name}_{success_needed}_demos"
                 f"_scripted{noise_tag}_{uuid}.pkl")
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"Saved {success_needed} demos to {file_name}")


if __name__ == "__main__":
    app.run(main)