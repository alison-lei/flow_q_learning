import numpy as np
import pickle
from envs import robomimic_utils
import jax.numpy as jnp
from agents.fql import FQLAgent
from scipy.spatial.transform import Rotation as R

file_path = "robomimic_datasets/lift_300_demos_scripted_flow_2026-06-10_14-29-49-254753.pkl"
with open(file_path, "rb") as f:
    data = pickle.load(f)


count = 0
for i in range(len(data)):
    # if data[i]["infos"]["succeed"]:
    #     count += 1
    if data[i]["rewards"] == 0:
        count += 1

print(count)

# env = robomimic_utils.make_eval_env("robomimic-lift")
# obs, _ = env.reset()

t = data[0]
env = robomimic_utils.make_eval_env("robomimic-lift")
obs, _ = env.reset(options={"reset_pose": t["reset_pose"],
                            "retry_from_same_state": True})
print("Observation once the environment just reset: ", obs)
print("First observation of the 300 collected demos: ", data[0]["observations"]["state"][0])

next_obs, *_ = env.step(t["actions"])
print("stored next obs:", t["next_observations"]["state"].squeeze())
print("reproduced next obs:", next_obs)
print("max abs diff:", np.abs(next_obs - t["next_observations"]["state"].squeeze()).max())

q = env.unwrapped._get_tcp_pose(frame="base")[3:]
print("quat from env:", q)

q = env.unwrapped._get_tcp_pose(frame="base")[3:]
print("as scalar-last euler:", R.from_quat(q).as_euler("xyz"))
print("as scalar-first euler:", R.from_quat(q, scalar_first=True).as_euler("xyz"))
obs, _ = env.reset()
print("expected (RESET_POSE[3:]):", obs[3:])  # [pi, 0, pi/2] + yaw noise


# /////////////////////////////////////////////////////////////////////

def _flatten_through_wrappers(env, processed_obs):
    # SERLObsWrapper flattens {"state": {...}} -> {"state": (13,)} alphabetically.
    # Easiest: call the SERL wrapper's observation() on the processed dict.
    # Walk down to the SERLObsWrapper instance and apply its transform.
    e = env
    while e is not None and type(e).__name__ != "SERLObsWrapper":
        e = getattr(e, "env", None)
    if e is None:
        raise RuntimeError("SERLObsWrapper not found in stack")
    flat = e.observation(processed_obs)     # -> {"state": (13,)}
    return np.asarray(flat["state"], dtype=np.float32)

def probe_value_vs_cube_height(agent, env, lift_heights=(0.0, 0.02, 0.04, 0.08, 0.12)):
    """Check whether the critic assigns higher Q to a lifted cube.

    Resets the env, then for each height sets the cube that many meters
    above the table, holds the arm/gripper state fixed, and reads the
    agent's Q-value for the resulting observation. If the reward signal
    reached the critic, Q should increase with cube height.
    """
    raw = env.unwrapped._env          # the FixedCubeLift robosuite env
    cube_joint = raw.cube.joints[0]   # free joint name for the cube

    obs, _ = env.reset()

    # Cube's current pose right after reset (this is "on the table")
    base_qpos = raw.sim.data.get_joint_qpos(cube_joint).copy()  # [x,y,z, qw,qx,qy,qz]
    table_z = base_qpos[2]

    results = []
    for h in lift_heights:
        # Set cube height = table position + h, keep xy and orientation fixed
        qpos = base_qpos.copy()
        qpos[2] = table_z + h
        raw.sim.data.set_joint_qpos(cube_joint, qpos)
        raw.sim.forward()             # propagate the state change (no dynamics step)

        # Rebuild the observation the agent would see for this cube pose.
        # Re-run the env's own obs pipeline so the 13-vector matches training.
        raw_obs = raw._get_observations()
        proc = env.unwrapped._process_obs(raw_obs)     # PandaLiftGymEnv/base processing
        # Push through the wrapper stack's observation transform to get the flat (13,)
        state = _flatten_through_wrappers(env, proc)

        # Sample the agent's own action for this state, then read Q(s, a)
        action = np.array(agent.sample_actions(observations=state,
                                               seed=__import__("jax").random.PRNGKey(0),
                                               temperature=0))
        action = np.clip(action, -1, 1)
        qs = agent.network.select('critic')(state[None], actions=action[None])
        q = float(jnp.mean(qs))

        # Does the env consider this a success at this height?
        success = raw._check_success()
        results.append((h, q, success))
        print(f"lift height {h:.3f} m | Q = {q:8.4f} | success = {success}")

    print("\nQ should increase with height if the reward reached the critic.")
    return results

# env = robomimic_utils.make_eval_env("robomimic-lift")
# agent = FQLAgent()
# agent.load
# print(probe_value_vs_cube_height(env))

