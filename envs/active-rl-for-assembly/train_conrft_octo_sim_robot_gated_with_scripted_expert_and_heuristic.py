#!/usr/bin/env python3

import glob
import time
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
from flax.core import frozen_dict
import os
import copy
import pickle as pkl
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from natsort import natsorted

from serl_launcher.agents.continuous.conrft_single_octo_cp import ConrftCPOctoAgentSingleArm
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from external.conrft.examples.data_util import add_mc_returns_to_trajectory, add_next_embeddings_to_trajectory

from serl_launcher.utils.launcher import (
    # make_conrft_octo_cp_pixel_agent_single_arm,
    make_trainer_config,
    make_wandb_logger,
)
from train.conrft.conrft_pixel_agent import make_conrft_octo_cp_pixel_agent_single_arm
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from train.mappings import CONFIG_MAPPING
from train.configs.utils import save_configs

from octo.model.octo_model import OctoModel

from train.robot_gated_heurisitcs.heuristics import (
    RandomRobotGatedHeuristicEmpirical,
    RandomRobotGatedHeuristicEmpiricalCurriculum,
    RobotGatedHeuristicQValueSlopeRecover,
    HumanGatedHeuristicWithButtonPresses,
)

from sim_env.experts.expert_utils import compute_privileged_indices, strip_privileged, strip_privileged_space

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")

flags.DEFINE_float("reward_scale", 1.0, "reward_scale ")
flags.DEFINE_float("reward_bias", 0.0, "reward_bias")
flags.DEFINE_float("q_weight", 0.1, "q_weight ")
flags.DEFINE_float("bc_weight", 1.0, "bc_weight")

flags.DEFINE_integer("pretrain_steps", 2000, "Number of pretrain steps.")
flags.DEFINE_boolean("debug", False, "Debug mode.")  # debug mode will disable wandb logging

flags.DEFINE_integer("critic_ensemble_size", 2, "")
flags.DEFINE_integer("critic_subsample_size", None, "")

flags.DEFINE_string("render_mode", "human", "If we want to render the environment.")
flags.DEFINE_boolean("verbose", False, "If we want to print robot/expert switches.")

flags.DEFINE_integer("port_number", 5588, "")
flags.DEFINE_integer("broadcast_port", 5589, "")

flags.DEFINE_float("cql_alpha", None, "")

flags.DEFINE_integer("train_seed", 42, "")
flags.DEFINE_integer("env_seed", 42, "Random seed.")
flags.DEFINE_integer("training_starts", 100, "")
flags.DEFINE_integer("expert_seed", 42, "")
flags.DEFINE_boolean("expert_argmax", False, "")

flags.DEFINE_integer("cta_ratio", 2, "")
flags.DEFINE_integer("max_steps_actor", 30001, "")
flags.DEFINE_integer("max_steps_learner", 150000, "")

flags.DEFINE_boolean("save_video", False, "")
flags.DEFINE_string("video_path", "./videos", "")

flags.DEFINE_integer("obs_horizon", 1, "")
flags.DEFINE_float("reward_pos", None, "")
flags.DEFINE_float("reward_neg", None, "")

# flags.DEFINE_list(
#     "active_action_indices", None,
#     "Comma-separated indices into the full action [x,y,z,rx,ry,rz,(gripper)] "
#     "to train on (e.g. 0,1,2,6); remaining dims are zero-filled by "
#     "ActionFillWrapper. Unset -> full action.")

# flags.DEFINE_string("heuristic", None, "") # RG_SLOPE, RG_RANDOM, HG

flags.DEFINE_integer("dagger_dump_every_n_episodes", 100,
                     "Flush completed dagger_stats to a chunk file every N episodes "
                     "to bound actor memory. Each flush writes a separate file.")
flags.DEFINE_integer("window_size", 20, "")
flags.DEFINE_integer("min_intervention_steps", 10, "")

flags.DEFINE_boolean("curriculum", False, "")

devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))

def print_red(x):
    return print("\033[31m {}\033[00m".format(x))

def print_blue(x):
    return print("\033[34m {}\033[00m".format(x))

def critic_fingerprint(params):
    """Cheap scalar fingerprint of the critic params for sync verification."""
    critic_leaves = jax.tree_util.tree_leaves(params["modules_critic"])
    return float(jnp.sum(critic_leaves[0]))

def get_new_dagger_dict(heuristic):
    fresh_dagger_stats = {"successful": False,
                            "episode_len": 0,
                            "intervention_count": 0,
                            "intervention_steps": 0,
                            # Per-step who-acted flag, consumed by analysis
                            # notebooks (e.g. processing_4_0_finetuning_robosuite.ipynb)
                            # for intervention-span shading and filtering.
                            "expert_acts": []}
    if isinstance(heuristic, RandomRobotGatedHeuristicEmpirical) or isinstance(heuristic, RandomRobotGatedHeuristicEmpiricalCurriculum):
        fresh_dagger_stats["requested_n_interventions"] = {} #key is step idx, value is length of intervention
    elif isinstance(heuristic, RobotGatedHeuristicQValueSlopeRecover):
        fresh_dagger_stats["q_values"] = []
    elif isinstance(heuristic, HumanGatedHeuristicWithButtonPresses):
        fresh_dagger_stats["button_presses_idxs"] = []
        fresh_dagger_stats["q_values"] = []
    return fresh_dagger_stats

##############################################################################


def actor(tasks, agent, data_store, intvn_data_store, env, sampling_rng,
          heuristic, expert):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """

    start_step = (
        int(os.path.basename(natsorted(glob.glob(os.path.join(
            FLAGS.checkpoint_path, "buffer/*.pkl")))[-1])[12:-4]) + 1
        if FLAGS.checkpoint_path and os.path.exists(os.path.join(FLAGS.checkpoint_path, "buffer"))
        else 0
    )

    datastore_dict = {
        f"actor_env_{FLAGS.port_number}": data_store,
        f"actor_env_intvn_{FLAGS.port_number}": intvn_data_store,
    }

    client = TrainerClient(
        f"actor_env_{FLAGS.port_number}",
        FLAGS.ip,
        make_trainer_config(port_number=FLAGS.port_number, broadcast_port=FLAGS.broadcast_port),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=3000,
    )

    # Function to update the agent with new params
    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    transitions = []
    demo_transitions = []

    obs, _ = env.reset()
    stripped_obs = strip_privileged(obs, priv_start, priv_end)

    # training loop
    timer = Timer()
    running_return = 0.0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0
    trajectory = []

    # log dagger stats
    dagger_stats = dict()
    episode_idx = 0
    dagger_stats[episode_idx] = get_new_dagger_dict(heuristic)

    pbar = tqdm.tqdm(range(start_step, FLAGS.max_steps_actor), dynamic_ncols=True)
    prev_step = -1
    print_green("POLICY") #TODO
    for step in pbar:
        # action_embeddings = None #to avoid possible bugs
        policy_executes_action = None
        timer.tick("total")

        with timer.context("sample_actions"):
            if step < config.random_steps:
                executed_actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                stripped_obs_dev = jax.device_put(stripped_obs)

                # action_embeddings = jax.device_get(action_embeddings) #to optimize time
                if isinstance(heuristic, RandomRobotGatedHeuristicEmpirical) or isinstance(heuristic, RandomRobotGatedHeuristicEmpiricalCurriculum):
                    policy_actions, action_embeddings = agent.sample_actions(observations=stripped_obs_dev,
                                                            tasks=jax.device_put(tasks),
                                                            seed=key)
                    expert_acted = heuristic.expert_acts
                    heuristic.evaluate_state()
                    expert_acts = heuristic.expert_acts
                    if expert_acts:
                        if not expert_acted:
                            # start counting from 0
                            dagger_stats[episode_idx]["requested_n_interventions"][step - prev_step - 1] = heuristic.intervention_length
                            expert.reset()
                        executed_actions = expert.sample_actions(observations=obs)
                        policy_executes_action = False
                    else:
                        executed_actions = np.asarray(jax.device_get(policy_actions))
                        policy_executes_action = True
                elif isinstance(heuristic, RobotGatedHeuristicQValueSlopeRecover):
                    # implementation where we track expert action Q-values for recovery
                    # expert_acted = heuristic.expert_acts

                    # if expert_acted:
                    #     # sample expert action
                    #     executed_actions = expert.sample_actions(observations=obs)
                    #     expert_q_values = agent.eval_q_values(observations=stripped_obs_dev,
                    #                                           actions=jax.device_put(jnp.expand_dims(executed_actions, 0)))
                    #     expert_q_values = jax.device_get(expert_q_values)
                    #     heuristic.add_q_value(expert_q_values)
                    #     # evaluate recovery
                    #     heuristic.evaluate_state()
                    #     #even if it is recovered, we want to execute this last step that leads to recovery
                    #     policy_executes_action = False
                    #     dagger_stats[episode_idx]["q_values"].append(expert_q_values)
                    # else:
                    #     policy_q_values = agent.eval_q_values(observations=stripped_obs_dev,
                    #                                           actions=jnp.expand_dims(policy_actions, 0))
                    #     policy_q_values = jax.device_get(policy_q_values)
                    #     heuristic.add_q_value(policy_q_values)
                    #     dagger_stats[episode_idx]["q_values"].append(policy_q_values)
                    #     heuristic.evaluate_state()
                    #     expert_acts = heuristic.expert_acts
                    #     if expert_acts:
                    #         expert.reset()
                    #         executed_actions = expert.sample_actions(observations=obs)
                    #         policy_executes_action = False
                    #     else:
                    #         executed_actions = jax.device_get(policy_actions)
                    #         policy_executes_action = True

                    # implementation where we track policy action Q-values for recovery
                    policy_actions, action_embeddings, policy_q_values = agent.sample_actions_and_q_values(observations=stripped_obs_dev,
                                                                                                            tasks=jax.device_put(tasks),
                                                                                                            seed=key)
                    expert_acted = heuristic.expert_acts
                    policy_q_values = jax.device_get(policy_q_values)
                    heuristic.add_q_value(policy_q_values)
                    dagger_stats[episode_idx]["q_values"].append(policy_q_values)
                    heuristic.evaluate_state()
                    expert_acts = heuristic.expert_acts

                    if expert_acts:
                        if not expert_acted:
                            expert.reset()
                            print_red("HUMAN")
                        executed_actions = expert.sample_actions(observations=obs)
                        policy_executes_action = False
                    else:
                        executed_actions = np.asarray(jax.device_get(policy_actions))
                        policy_executes_action = True
                        if expert_acted:
                            print_green("POLICY")
                elif isinstance(heuristic, HumanGatedHeuristicWithButtonPresses):
                    # policy_actions, action_embeddings = agent.sample_actions(observations=stripped_obs_dev,
                    #                                         tasks=jax.device_put(tasks),
                    #                                         seed=key)
                    policy_actions, action_embeddings, policy_q_values = agent.sample_actions_and_q_values(observations=stripped_obs_dev,
                                                                                                            tasks=jax.device_put(tasks),
                                                                                                            seed=key)
                    policy_q_values = jax.device_get(policy_q_values)
                    dagger_stats[episode_idx]["q_values"].append(policy_q_values)
                    # Heuristic polls keyboard internally; returns True if
                    # a press was just consumed (state flipped).
                    toggled = heuristic.evaluate_state()
                    # Handle toggle side effects first (log + per-transition reset).
                    if toggled:
                        dagger_stats[episode_idx]["button_presses_idxs"].append(step - prev_step - 1)
                        if heuristic.expert_acts:
                            # robot -> expert transition: reset expert state
                            expert.reset()
                            print_red("HUMAN") #TODO
                        else:
                            print_green("POLICY")
                    # Then route the action based on the (possibly just-flipped) state.
                    if heuristic.expert_acts:
                        executed_actions = expert.sample_actions(observations=obs)
                        policy_executes_action = False
                    else:
                        executed_actions = jax.device_get(policy_actions)
                        policy_executes_action = True
                    
                    # if policy_executes_action:
                    #     print_green("POLICY")
                    # else:
                    #     print_red("HUMAN")
                                
                assert policy_executes_action is not None, "Policy executes action is not found"
                # if FLAGS.verbose:
                #     print("POLICY" if policy_executes_action else "EXPERT")
     
        # Step environment
        with timer.context("step_env"):
            # Per-step expert_acts log for offline analysis (mirrors the
            # expert_acts value sent to the env wrapper).
            dagger_stats[episode_idx]["expert_acts"].append(not policy_executes_action)
            if heuristic is not None:
                action_dict = {"action": executed_actions,
                                "expert_acts": not policy_executes_action}

                next_obs, reward, done, truncated, info = env.step(action_dict)
            else:
                # print("RG HEURISTIC IS NONE")
                next_obs, reward, done, truncated, info = env.step(executed_actions)
            
            stripped_next_obs = strip_privileged(next_obs, priv_start, priv_end)

            # next_obs, reward, done, truncated, info = env.step(actions)
            if "left" in info:
                info.pop("left")
            if "right" in info:
                info.pop("right")

            # override the action with the intervention action
            if "intervene_action" in info:
                # print_green("EXPERT INTERVENTION")
                executed_actions = info.pop("intervene_action")
                intervention_steps += 1
                if not already_intervened:
                    intervention_count += 1
                already_intervened = True
            else:
                # print_green("NO EXPERT INTERVENTION")
                already_intervened = False

            # if FLAGS.verbose:
                # print("INFO IS:", info)
                # print("REWARD:", reward)
            running_return += reward

            # assert action_embeddings is not None, "Action embeddings are not found"
            transition = dict(
                observations=stripped_obs,
                actions=executed_actions,
                next_observations=stripped_next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done,
                intervened=already_intervened,
                embeddings=action_embeddings,
            )
            if 'grasp_penalty' in info:
                transition['grasp_penalty'] = info['grasp_penalty']

            trajectory.append(transition)

            obs = next_obs
            stripped_obs = stripped_next_obs
            if done or truncated:
                if reward > 0:
                    print_blue("SUCCESS")
                else:
                    print_blue("FAILURE")
                # if FLAGS.verbose:
                #     if reward > 0:
                #         print("SUCCESS", step - prev_step)
                #     else:
                #         print("FAILURE", step - prev_step)
                prev_step = step
                
                #log dagger stats
                dagger_stats[episode_idx]["successful"] = info['succeed']
                dagger_stats[episode_idx]["episode_len"] = info["episode"]["l"]
                dagger_stats[episode_idx]["intervention_count"] = intervention_count
                dagger_stats[episode_idx]["intervention_steps"] = intervention_steps

                # Flush completed episodes to a chunk file every N episodes,
                # then clear the dict so memory stays bounded.
                if (episode_idx + 1) % FLAGS.dagger_dump_every_n_episodes == 0:
                    dagger_stats_path = os.path.join(
                        FLAGS.video_path,
                        f"dagger_stats_finetuning_ep_{episode_idx + 1}.pkl",
                    )
                    with open(dagger_stats_path, "wb") as f:
                        pkl.dump(dagger_stats, f)
                    dagger_stats.clear()
                # print(f"EPISODE IDX: {episode_idx}")
                episode_idx += 1
                dagger_stats[episode_idx] = get_new_dagger_dict(heuristic)

                trajectory = add_mc_returns_to_trajectory(trajectory, config.discount,
                                                          FLAGS.reward_scale, FLAGS.reward_bias,
                                                          env.config.reward_neg, is_sparse_reward=False
                                                          )
                trajectory = add_next_embeddings_to_trajectory(trajectory)
                for transition in trajectory:
                    data_store.insert(transition)
                    transitions.append(copy.deepcopy(transition))
                    if transition['intervened']:
                        intvn_data_store.insert(transition)
                        demo_transitions.append(copy.deepcopy(transition))

                info["episode"]["intervention_count"] = intervention_count
                info["episode"]["intervention_steps"] = intervention_steps
                info["episode"]["succeed"] = int(info['succeed'])
                info["episode"]["total_steps"] = step
                # send stats to the learner to log
                stats = {"environment": info}
                client.request("send-stats", stats)
                pbar.set_description(f"last return: {running_return}")
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                already_intervened = False
                client.update()
                trajectory = []
                obs, _ = env.reset()
                stripped_obs = strip_privileged(obs, priv_start, priv_end)
                expert.reset()
                if heuristic is not None:
                    heuristic.reset()
                
                print_green("POLICY") #TODO

        if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
            # dump to pickle file
            buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
            demo_buffer_path = os.path.join(
                FLAGS.checkpoint_path, "demo_buffer")
            if not os.path.exists(buffer_path):
                os.makedirs(buffer_path)
            if not os.path.exists(demo_buffer_path):
                os.makedirs(demo_buffer_path)
            with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                pkl.dump(transitions, f)
                transitions = []
            with open(
                os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb"
            ) as f:
                pkl.dump(demo_transitions, f)
                demo_transitions = []
        
        if config.buffer_period == 0 and step % 1000 == 0: #this is to avoid OOM errors
            transitions = []
            demo_transitions = []
        
        if step > 0 and config.synchronized_checkpoint_period > 0 and step % config.synchronized_checkpoint_period == 0:
            stats = {"environment_step": step}
            client.request("send-stats", stats)

        timer.tock("total")

        if step % config.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)
        
    # Final flush of any remaining episodes. The current episode_idx entry
    # didn't hit done/truncated, so its counters were never written by the
    # episode-end block — record them here and mark it partial.
    if episode_idx in dagger_stats:
        dagger_stats[episode_idx]["intervention_count"] = intervention_count
        dagger_stats[episode_idx]["intervention_steps"] = intervention_steps
        dagger_stats[episode_idx]["partial"] = True
    dagger_stats_path = os.path.join(
        FLAGS.video_path,
        f"dagger_stats_finetuning_ep_{episode_idx}_final.pkl",
    )
    with open(dagger_stats_path, "wb") as f:
        pkl.dump(dagger_stats, f)

    client.request("send-stats", {"stop_training": True})
    client.stop()


##############################################################################


def learner(rng, tasks, agent, replay_buffer, demo_buffer, wandb_logger=None, restored_checkpoint=False):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    start_step = (
        int(os.path.basename(checkpoints.latest_checkpoint(
            FLAGS.checkpoint_path))[11:]) + 1
        # if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
        if restored_checkpoint
        else 0
    )
    step = start_step
    stop_training = False
    environment_step = 0
    online_start_step = start_step

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if payload.pop("stop_training", False):
            print("STOP TRAINING REQUEST RECEIVED")
            nonlocal stop_training
            stop_training = True
        if "environment_step" in payload:
            nonlocal environment_step
            environment_step = payload.pop("environment_step")
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(make_trainer_config(port_number=FLAGS.port_number,
                                               broadcast_port=FLAGS.broadcast_port),
                            request_callback=stats_callback)
    server.register_data_store(f"actor_env_{FLAGS.port_number}", replay_buffer)
    server.register_data_store(f"actor_env_intvn_{FLAGS.port_number}", demo_buffer)
    server.start(threaded=True)

    train_critic_networks_to_update = frozenset({"critic"})
    train_actor_networks_to_update = frozenset({"actor"})
    train_networks_to_update = frozenset({"critic", "actor"})

    def create_batch_tasks(data_dict, batch_size):
        batch_dict = {}
        for key, value in data_dict.items():
            if isinstance(value, dict):  # Handling nested dictionary (e.g., language_instruction)
                batch_dict[key] = {k: np.tile(
                    v, (batch_size, *([1] * (v.ndim - 1)))) for k, v in value.items()}
            else:
                # For non-dictionary values, repeat along batch dimension (axis=0)
                batch_dict[key] = np.tile(
                    value, (batch_size, *([1] * (value.ndim - 1))))  # Repeat along axis 0

        return batch_dict

    # Pretrain the model with the demo data
    if step < FLAGS.pretrain_steps:
        print_green("Pretraining the model with demo data")
        for step in tqdm.tqdm(range(start_step, FLAGS.pretrain_steps + 1), desc="pretraining"):
            for _ in range(FLAGS.cta_ratio - 1):
                batch = next(demo_buffer.get_iterator(
                    sample_args={"batch_size": config.batch_size,
                                 "pack_obs": True, },
                    device=sharding.replicate(),
                ))

                batch = {
                    **batch,
                    "tasks": create_batch_tasks(tasks, config.batch_size),
                }
                batch = frozen_dict.freeze(batch)
                agent, critics_info = agent.update_calql(
                    batch, networks_to_update=train_critic_networks_to_update,)

            batch = next(demo_buffer.get_iterator(
                sample_args={"batch_size": config.batch_size,
                             "pack_obs": True, },
                device=sharding.replicate(),
            ))

            batch = {
                **batch,
                "tasks": create_batch_tasks(tasks, config.batch_size),
            }
            batch = frozen_dict.freeze(batch)
            agent, update_info = agent.update_calql(
                    batch, networks_to_update=train_networks_to_update,)

            if step % config.log_period == 0 and wandb_logger:
                wandb_logger.log(update_info, step=step)

            if step > 0 and config.checkpoint_period and step % config.checkpoint_period == 0:
                checkpoints.save_checkpoint(
                    os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=100)

        print_green("Pretraining done")
        return  # after pretraining, return and exit
    else:
        print_green(
            "Existing pretrained checkpoint model found. Skipping pretraining")

    agent = jax.block_until_ready(agent)
    server.publish_network(agent.state.params)

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=FLAGS.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < FLAGS.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # send the initial network to the actor
    server.publish_network(agent.state.params) 
    print_green("sent initial network to actor")

    # 50/50 sampling from RLPD, half from demo and half from online experience
    replay_iterator = replay_buffer.get_iterator(
        sample_args={"batch_size": config.batch_size // 2, "pack_obs": True, },
        device=sharding.replicate(),
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={"batch_size": config.batch_size // 2, "pack_obs": True, },
        device=sharding.replicate(),
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()

    # Start online training after offline pretraining
    online_start_step = FLAGS.pretrain_steps + \
        1 if online_start_step < FLAGS.pretrain_steps else online_start_step
    for step in tqdm.tqdm(range(online_start_step, FLAGS.max_steps_learner), dynamic_ncols=True, desc="learner"):
        if stop_training:
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=100)
            return
        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
        for critic_step in range(FLAGS.cta_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)

                batch = {
                    **batch,
                    "tasks": create_batch_tasks(tasks, config.batch_size),
                }
            batch = frozen_dict.freeze(batch)

            with timer.context("train_critics"):
                agent, critics_info = agent.update_ql(
                    batch, networks_to_update=train_critic_networks_to_update,)

        with timer.context("train"):
            batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch = concat_batches(batch, demo_batch, axis=0)
            batch = {
                **batch,
                "tasks": create_batch_tasks(tasks, config.batch_size),
            }
            batch = frozen_dict.freeze(batch)
            agent, update_info = agent.update_ql(
                batch, networks_to_update=train_networks_to_update,)

        # publish the updated network
        if step > 0 and step % (config.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        if (step > 0 and config.checkpoint_period and step % config.checkpoint_period == 0):
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=100)
        
        if step > 0 and environment_step > 0:
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=100)
            correspondences_path = os.path.join(os.path.abspath(FLAGS.checkpoint_path), "correspondences.txt")
            if not os.path.exists(correspondences_path):
                with open(correspondences_path, "w") as f:
                    f.write("env_step,learner_step\n")
            with open(correspondences_path, "a") as f:
                f.write(f"{environment_step},{step}\n")
            environment_step = 0


##############################################################################


def main(_):
    import os
    print("MUJOCO_GL", os.environ.get("MUJOCO_GL")) 
    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]() # FLAGS.exp_name = lift, which returns LiftTrainConfigTrainConfig which parent is DefaultRobosuiteTrainConfig
    #TODO: need to be tested
    print_green(f"WINDOW SIZE IS {config.window_size}")
    config.window_size = FLAGS.window_size
    print_green(f"UPDATED WINDOW SIZE IS {config.window_size}")
    print_green(f"MIN INTERVENTION STEPS IS {config.min_intervention_steps}")
    config.min_intervention_steps = FLAGS.min_intervention_steps
    print_green(f"UPDATED MIN INTERVENTION STEPS IS {config.min_intervention_steps}")

    # print_green(f"HEURISTIC IS {config.intervention_heuristic}")
    # if FLAGS.curriculum:
    #    config.intervention_heuristic = RandomRobotGatedHeuristicEmpiricalCurriculum
    # else:
    #    config.intervention_heuristic = RandomRobotGatedHeuristicEmpirical
    # print_green(f"UPDATED HEURISTIC IS {config.intervention_heuristic}")
        
    print_green(f"IMAGE KEYS ARE {config.image_keys}")
    print_green(f"LEN OF IMAGE KEYS IS {len(config.image_keys)}")

    assert config.batch_size % num_devices == 0
    print("CHECKPOINT PATH ", FLAGS.checkpoint_path)
    print("BATCH SIZE IS ", config.batch_size)
    print("TRAIN SEED IS ", FLAGS.train_seed)
    print("PROPRIO KEYS ARE:", config.proprio_keys)

    rng = jax.random.PRNGKey(FLAGS.train_seed)
    rng, sampling_rng = jax.random.split(rng)

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    # active_action_indices = (
    #     [int(i) for i in FLAGS.active_action_indices]
    #     if FLAGS.active_action_indices else None
    # )
    env = config.get_environment(
        fake_env=FLAGS.learner,
        save_video=FLAGS.save_video,
        video_path=FLAGS.video_path,
        classifier=False,
        render_mode=FLAGS.render_mode,
        env_seed=FLAGS.env_seed,
        # init_heuristic=config.heuristic is not None, #FLAGS.heuristic is not None,
        obs_horizon=FLAGS.obs_horizon,
        reward_pos=FLAGS.reward_pos,
        reward_neg=FLAGS.reward_neg,
        # active_action_indices=active_action_indices,
    )
    env = RecordEpisodeStatistics(env)

    rng, sampling_rng = jax.random.split(rng)

    include_octo_embeddings = False
    include_mc_returns = True

    octo_model = OctoModel.load_pretrained(config.octo_path)
    tasks = octo_model.create_tasks(texts=[config.task_desc])

    if FLAGS.actor:
        expert = None
        # if FLAGS.heuristic is not None:
        if config.intervention_heuristic is not None:
            print("ROBOT GATED INTERVENTIONS")
            if config.intervention_heuristic is RandomRobotGatedHeuristicEmpirical:
                stats_path = f"./stats_for_random_emperical/{FLAGS.exp_name}_random_empirical_stats_ws_{config.window_size}_min_step_{config.min_intervention_steps}_FINAL_OPT_maxlen_140_seed_{FLAGS.train_seed}.pkl"
                intervention_seed = FLAGS.train_seed + 1000
                heuristic = config.intervention_heuristic(stats_path, intervention_seed)
            elif config.intervention_heuristic is RandomRobotGatedHeuristicEmpiricalCurriculum:
                stats_path = f"./stats_for_random_emperical/{FLAGS.exp_name}_random_empirical_curriculum_stats_ws_{config.window_size}_min_step_{config.min_intervention_steps}_chunk_{config.chunk_size}_FINAL_OPT_maxlen_140_seed_{FLAGS.train_seed}.pkl"
                intervention_seed = FLAGS.train_seed + 1000
                heuristic = config.intervention_heuristic(stats_path, intervention_seed)
            elif config.intervention_heuristic is RobotGatedHeuristicQValueSlopeRecover:
                heuristic = config.intervention_heuristic(config)
            elif config.intervention_heuristic is HumanGatedHeuristicWithButtonPresses:
                # Heuristic owns its own KeyboardExpert and polls SPACE in
                # evaluate_state() to toggle expert_acts (no env wrapper).
                heuristic = config.intervention_heuristic(env)
            else:
                heuristic = config.intervention_heuristic()
            if config.scripted_expert:
                from sim_env.experts import get_scripted_expert
                expert = get_scripted_expert(
                    task_name=FLAGS.exp_name,
                    proprio_keys=config.proprio_keys,
                    env=env,
                    obs_horizon=FLAGS.obs_horizon,
                    privileged_shape=env.unwrapped.get_privileged_shape(),
                    seed=FLAGS.expert_seed,
                )
                print_green("Loaded scripted expert for task: " + FLAGS.exp_name)
        else:
            heuristic = None
    
    priv_shape = env.unwrapped.get_privileged_shape()
    global priv_start, priv_end
    priv_start, priv_end, total_dim = compute_privileged_indices(
        config.proprio_keys, priv_shape, env=env
    )

    state_dim = env.observation_space["state"].shape[-1]
    assert total_dim == state_dim, (
        f"compute_privileged_indices total_dim={total_dim} "
        f"but env state last-dim={state_dim}; "
        f"check DEFAULT_KEY_SHAPES, env.get_privileged_shape(), and the wrapper chain"
    )

    stripped_obs_space = strip_privileged_space(env.observation_space, priv_start, priv_end)
    print("TEST OBSERVATION SPACE:", stripped_obs_space)
    print("TEST OBSERVATION HORIZON:", env.obs_horizon)

    sample_obs_full = env.observation_space.sample()
    sample_obs_stripped = strip_privileged(sample_obs_full, priv_start, priv_end)

    if config.setup_mode == 'single-arm-fixed-gripper':
        agent: ConrftCPOctoAgentSingleArm = make_conrft_octo_cp_pixel_agent_single_arm(
            seed=FLAGS.train_seed,
            sample_obs=sample_obs_stripped,
            sample_action=env.action_space.sample(),
            sample_tasks=tasks,
            octo_model=octo_model,
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
            fix_gripper=True,
            q_weight=FLAGS.q_weight,
            bc_weight=FLAGS.bc_weight,
            critic_ensemble_size=FLAGS.critic_ensemble_size,
            critic_subsample_size=FLAGS.critic_subsample_size,
            cql_alpha=FLAGS.cql_alpha,
            dagger=config.intervention_heuristic is not None
        )
        include_grasp_penalty = False
        include_octo_embeddings = True
    elif config.setup_mode == 'single-arm-learned-gripper':
        agent: ConrftCPOctoAgentSingleArm = make_conrft_octo_cp_pixel_agent_single_arm(
            seed=FLAGS.train_seed,
            sample_obs=sample_obs_stripped,
            sample_action=env.action_space.sample(),
            sample_tasks=tasks,
            octo_model=octo_model,
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
            q_weight=FLAGS.q_weight,
            bc_weight=FLAGS.bc_weight,
            critic_ensemble_size=FLAGS.critic_ensemble_size,
            critic_subsample_size=FLAGS.critic_subsample_size,
            cql_alpha=FLAGS.cql_alpha,
            dagger=config.intervention_heuristic is not None
        )
        include_grasp_penalty = False #True #ConRFT does not use grasp penalty
        include_octo_embeddings = True
    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")
    
    print("INCLUDE GRASP PENALTY:", include_grasp_penalty)
        

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(jax.tree_map(
        jnp.array, agent), sharding.replicate())

    restored_checkpoint = False
    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
        folders = [f for f in os.listdir(FLAGS.checkpoint_path) if os.path.isdir(os.path.join(FLAGS.checkpoint_path, f))]
        if len(folders) != 0:
            ckpt = checkpoints.restore_checkpoint(
                FLAGS.checkpoint_path, agent.state,)
            # agent = agent.replace(state=ckpt)

            # Update params only, ignore the optimizer states
            new_params = ckpt.params
            new_target_params = ckpt.target_params

            agent = agent.replace(state=agent.state.replace(
                params=new_params, target_params=new_target_params))

            ckpt_number = os.path.basename(
                checkpoints.latest_checkpoint(FLAGS.checkpoint_path))[11:]
            print_green(f"Loaded previous checkpoint at step {ckpt_number}.")
            restored_checkpoint = True

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            stripped_obs_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
            include_octo_embeddings=include_octo_embeddings,
            include_mc_returns=include_mc_returns,
        )

        # set up wandb and logging
        wandb_logger = make_wandb_logger(
            project="conrft",
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
        )

        return replay_buffer, wandb_logger

    if FLAGS.learner:
        sampling_rng = jax.device_put(
            sampling_rng, device=sharding.replicate())
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            stripped_obs_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
            include_octo_embeddings=include_octo_embeddings,
            include_mc_returns=include_mc_returns,
        )
        print("OBSERVATION SPACE:", stripped_obs_space)
        assert FLAGS.demo_path is not None
        for path in FLAGS.demo_path:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    if 'infos' in transition and 'grasp_penalty' in transition['infos']:
                        transition['grasp_penalty'] = transition['infos']['grasp_penalty']
                    demo_buffer.insert(transition)
        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(os.path.join(FLAGS.checkpoint_path, "buffer")):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        replay_buffer.insert(transition)
            print_green(
                f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "demo_buffer")
        ):
            for file in glob.glob(
                os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl")
            ):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        demo_buffer.insert(transition)
            print_green(
                f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}") #TODO: check this print

        # learner loop
        step = (
            int(os.path.basename(checkpoints.latest_checkpoint(
                FLAGS.checkpoint_path))[11:]) + 1
            if restored_checkpoint
            else 0
        )
        if step < FLAGS.pretrain_steps:
            print_green("starting pretrain loop")
            process_name = "pretrain"
        else:
            print_green("starting learner loop")
            process_name = "learner"
        save_configs(process_name=process_name,
                     config=config,
                     flags=FLAGS,
                     config_mapping=CONFIG_MAPPING,
                     wandb_logger=wandb_logger,
                     env_config=env.unwrapped.config)
        learner(sampling_rng,
                tasks,
                agent,
                replay_buffer,
                demo_buffer=demo_buffer,
                wandb_logger=wandb_logger,
                restored_checkpoint=restored_checkpoint,
                )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(50000)  # the queue size on the actor
        intvn_data_store = QueuedDataStore(50000)

        # actor loop
        print_green("starting actor loop")
        save_configs(process_name="actor",
                     config=config,
                     flags=FLAGS,
                     config_mapping=CONFIG_MAPPING,
                     env_config=env.unwrapped.config)
        actor(tasks,
              agent,
              data_store,
              intvn_data_store,
              env,
              sampling_rng,
              heuristic,
              expert
              )

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)