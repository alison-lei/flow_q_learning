# image + proprio

import os
import platform

import json
import random
import time

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, ReplayBuffer
from utils.evaluation import evaluate, flatten
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

from octo.model.octo_model import OctoModel

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'finetune', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('eval_seed', None, 'seed used for evaluation')
# flags.DEFINE_string('env_name', 'robomimic-lift', 'Environment (dataset) name.')
flags.DEFINE_string('env_name', 'nut_assembly_square', 'Environment (dataset) name.')

flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('offline_steps', 0, 'Number of offline steps.')
flags.DEFINE_integer('online_steps', 100000, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 20000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 500, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 20000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 20000, 'Saving interval.')

flags.DEFINE_integer('eval_episodes', 20, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 5, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 1, 'Frame skip for videos.')

flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')
flags.DEFINE_integer('balanced_sampling', 1, 'Whether to use balanced sampling for online fine-tuning.')

config_flags.DEFINE_config_file('agent', 'agents/fql.py', lock_config=False)

flags.DEFINE_boolean('debug_eval', False,
                      'If True, restore the agent and render one debug episode, then exit '
                      '(does not run offline/online training).')

# TASK_DESC = "pick up the red block" # for lift
TASK_DESC = "pick up the brown square nut and insert it onto the square peg" # nut assembly square



def main(_):
    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project='nut', group=FLAGS.run_group,
    name=exp_name)
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Make environment and datasets.
    config = FLAGS.agent
    config['encoder'] = 'image_proprio'
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, frame_stack=FLAGS.frame_stack)

    # for the robomimic-based tasks (robomimic-lift, nut_assembly_square),
    # make_env_and_datasets always returns env=None and builds exactly one live
    # sim, returned as eval_env. Fold that single instance into `env` so the rest
    # of this file only ever deals with one variable for "the live sim we
    # currently have open" -- and so env and eval_env are never both pointing at
    # separate live render contexts at once (that's what was corrupting recorded
    # video frames). For OGBench/D4RL, env is already non-None here, so this is a
    # no-op and their existing dedicated training env is left untouched.
    if (FLAGS.online_steps > 0):
        if env is None:
            env = eval_env
        eval_env = None

    # added the restore_path guard so this can never accidentally close
    # the sim the debug/restore_path branch below needs, if someone ever runs
    # that branch with --online_steps=0 explicitly.
    if FLAGS.online_steps == 0 and FLAGS.restore_path is None and env is not None:
        env.close()
        env = None

    # Load Octo once -- used for both the standalone debug eval below and the
    # periodic in-training eval later in the loop.
    octo_model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small")
    tasks = octo_model.create_tasks(texts=[TASK_DESC])

    # Same helper used inside utils/evaluation.py's evaluate() and
    # run_single_episode_video(). Kept identical so online rollouts see the
    # exact same observation representation (state + images + Octo embedding)
    # the agent was trained/evaluated on.
    def add_embeddings(observation):
        image_primary = observation["side_policy_256"][np.newaxis, ...]
        image_wrist = observation["wrist_1"][np.newaxis, ...]
        timestep_pad_mask = np.array([[True, True]])
        octo_obs = {
            "image_primary": image_primary,
            "image_wrist": image_wrist,
            "timestep_pad_mask": timestep_pad_mask,
        }

        transformer_outputs = octo_model.sample_transformer(octo_obs, tasks, train=False)
        action_embeddings = transformer_outputs[:, -1, :]  # (1, embed_dim)

        observation = dict(observation)
        observation["embeddings"] = action_embeddings[0]
        return observation

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    # Set up datasets.
    train_dataset = Dataset.create(**train_dataset)

    if FLAGS.balanced_sampling:
        example_transition = jax.tree_util.tree_map(lambda x: x[0], dict(train_dataset))
        replay_buffer = ReplayBuffer.create(example_transition, size=FLAGS.buffer_size)
    else:
        train_dataset = ReplayBuffer.create_from_initial_dataset(
            dict(train_dataset), size=FLAGS.buffer_size + train_dataset.size
        )
        replay_buffer = train_dataset

    # Set p_aug and frame_stack.
    for dataset in [train_dataset, val_dataset, replay_buffer]:
        if dataset is not None:
            dataset.p_aug = FLAGS.p_aug
            dataset.frame_stack = FLAGS.frame_stack
            if config['agent_name'] == 'rebrac':
                dataset.return_next_actions = True

    # Create agent.
    example_batch = train_dataset.sample(1)
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Restore agent if requested -- used both for the standalone debug-eval path
    # below and for resuming/fine-tuning an already-trained checkpoint.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)
        

    # Standalone single-episode debug eval: `CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl python main.py --restore_path=exp/.../sd000_.../ --restore_epoch=300000 --eval_seed=345 --debug_eval=True`
    if FLAGS.debug_eval:
        assert FLAGS.restore_path is not None, "debug_eval requires --restore_path."
        from utils.evaluation import run_single_episode_video
        # have_success = have_failure = False
        # out_dir = "q_check/epoch{}".format(FLAGS.restore_epoch)
        # os.makedirs(out_dir, exist_ok=True)
        # ep_out = "q_check/epoch{}/tmp.mp4".format(FLAGS.restore_epoch)
        # while have_success == False or have_failure == False:
        #     success, length, q_values = run_single_episode_video(
        #         agent=agent, env=env, octo_model=octo_model, tasks=tasks,
        #         out_path=ep_out,
        #     )
        #     if success:
        #         have_success = True
        #         os.rename(ep_out, f"{out_dir}/success.mp4")
        #     else:
        #         have_failure = True
        #         os.rename(ep_out, f"{out_dir}/failure.mp4")
        
        success, length, q_values = run_single_episode_video(
            agent=agent, env=env, octo_model=octo_model, tasks=tasks,
            out_path="epoch{}_my_nut_video.mp4".format(FLAGS.restore_epoch),
        )
        os.rename(f"epoch{FLAGS.restore_epoch}_my_nut_video.mp4", f"epoch{FLAGS.restore_epoch}_{success}_my_nut_video.mp4")
        env.close()
        print(f"success={success}, length={length}")
        wandb.finish()
        return

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    step = 0
    done = True
    expl_metrics = dict()
    online_rng = jax.random.PRNGKey(FLAGS.seed)

    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + FLAGS.online_steps + 1), smoothing=0.1, dynamic_ncols=True):
        if i <= FLAGS.offline_steps:
            batch = train_dataset.sample(config['batch_size'])
            if config['agent_name'] == 'rebrac':
                agent, update_info = agent.update(batch, full_update=(i % config['actor_freq'] == 0))
            else:
                # CHANGED: pass batch as both args -- every transition sampled
                # here is trusted demo data during the offline phase, so the
                # actor's BC-flow regression is safe to see all of it.
                agent, update_info = agent.update(batch, batch)
        else:
            online_rng, key = jax.random.split(online_rng)
            if done and env is not None:
                step = 0
                ob, _ = env.reset()
                ob = add_embeddings(ob)  # match eval -- embed right after reset, before the loop

            # match eval: batch the observation, since agent.sample_actions and
            # the critic both expect a leading batch dim.
            batched_ob = jax.tree_util.tree_map(lambda x: x[None], ob)
            action = agent.sample_actions(observations=batched_ob, temperature=1, seed=key)
            action = np.clip(np.array(action)[0], -1, 1)  # match eval's action clipping

            next_ob, reward, terminated, truncated, info = env.step(action.copy())
            next_ob = add_embeddings(next_ob)  # match eval -- embed again every step
            done = terminated or truncated
            if 'antmaze' in FLAGS.env_name and (
                'diverse' in FLAGS.env_name or 'play' in FLAGS.env_name or 'umaze' in FLAGS.env_name
            ):
                reward = reward - 1.0
            replay_buffer.add_transition(
                dict(
                    observations=ob,  # full embedded dict: state, images, embeddings
                    actions=action,
                    rewards=reward,
                    terminals=float(done),
                    masks=1.0 - terminated,
                    next_observations=next_ob,  # embedded next_ob, not raw env output
                )
            )
            ob = next_ob
            if done:
                # skip non-numeric info fields (e.g. a string
                # termination reason) instead of crashing np.mean on them.
                flat_info = flatten(info)
                expl_metrics = {}
                for k, v in flat_info.items():
                    arr = np.asarray(v)
                    if np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_):
                        expl_metrics[f'exploration/{k}'] = np.mean(arr)
            step += 1

            if FLAGS.balanced_sampling:
                dataset_batch = train_dataset.sample(config['batch_size'] // 2)
                replay_batch = replay_buffer.sample(config['batch_size'] // 2)
                # CHANGED: dataset_batch/replay_batch contain nested dicts (observations,
                # next_observations hold {state, side_policy_256, wrist_1, embeddings}).
                # A flat dict-comprehension concatenation over top-level keys tries to
                # np.concatenate the nested dicts themselves; tree_map recurses into the
                # structure and concatenates the matching leaf arrays instead.
                batch = jax.tree_util.tree_map(
                    lambda a, b: np.concatenate([a, b], axis=0), dataset_batch, replay_batch
                )
            else:
                batch = replay_buffer.sample(config['batch_size'])
                # CHANGED: no separate demo-only source exists when
                # balanced_sampling=0 (train_dataset was merged into
                # replay_buffer above), so demo_batch falls back to the same
                # batch here. This means the demo_batch split provides no
                # protection under balanced_sampling=0 -- use
                # balanced_sampling=1 to actually get a clean demo source for
                # the actor's BC-flow loss.
                dataset_batch = batch
            if config['agent_name'] == 'rebrac':
                agent, update_info = agent.update(batch, full_update=(i % config['actor_freq'] == 0))
            else:
                # CHANGED: pass dataset_batch (demo-only under
                # balanced_sampling=1) as the second arg.
                agent, update_info = agent.update(batch, dataset_batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                # CHANGED: pass val_batch as both args -- val_dataset is
                # entirely demo data (and is None for the robomimic tasks
                # anyway, so this branch is currently unreachable for them).
                _, val_info = agent.total_loss(val_batch, val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            train_metrics.update(expl_metrics)
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if FLAGS.eval_interval != 0 and (i == 1 or i % FLAGS.eval_interval == 0):
            renders = []
            eval_metrics = {}

            # Close the training rollout env first so only one live render
            # context exists at a time.
            if env is not None:
                env.close()
                env = None

            # now it will be the same 30 episdoes, starting with same reset position for all evaluation
            _, eval_env, _, _ = make_env_and_datasets(FLAGS.env_name, frame_stack=FLAGS.frame_stack)

            eval_info, trajs, cur_renders = evaluate(
                agent=agent,
                env=eval_env,
                octo_model=octo_model,
                tasks=tasks,
                config=config,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )

            renders.extend(cur_renders)
            for k, v in eval_info.items():
                eval_metrics[f'evaluation/{k}'] = v

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders)
                eval_metrics['video'] = video
            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

            eval_env.close() # frees up the mujoco sim for next eval_interval steps of training
            eval_env = None

            # reopen using the same env/eval_env merge idiom as the
            # initial setup -- for nut_assembly_square this function always
            # returns env=None, so grabbing the first return value alone
            # would have silently left `env` as None for the rest of training.
            if FLAGS.online_steps > 0:
                env, eval_env, _, _ = make_env_and_datasets(FLAGS.env_name, frame_stack=FLAGS.frame_stack)
                if env is None:
                    env = eval_env
                eval_env = None
                done = True

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)