import numpy as np, h5py
from os.path import expanduser

f = h5py.File(expanduser("~/.robomimic/lift/mh/low_dim_v15.hdf5"), "r")

demos = list(f["data"].keys())
demo = demos[0]
print(f"{len(demos)} demos, first = {demo}")

# what's stored per demo
print("top level:", list(f[f"data/{demo}"].keys()))       # actions, obs, next_obs, rewards, dones, states
print("obs keys :", list(f[f"data/{demo}/obs"].keys()))

# shapes of every observation key
for k in f[f"data/{demo}/obs"]:
    print(f"  {k:24s} {f[f'data/{demo}/obs/{k}'].shape}")

# from envs.robomimic_utils import low_dim_keys

# keys = low_dim_keys["low_dim"]   # ('robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos','object')
# flat = np.concatenate([np.array(f[f"data/{demo}/obs/{k}"]) for k in keys], axis=-1)
# print(flat.shape)   # (T, 19) for lift

obj = np.array(f[f"data/{demo}/obs/object"])
eef = np.array(f[f"data/{demo}/obs/robot0_eef_pos"])
print("obj[0]:", np.round(obj[0], 4))
print("eef[0]:", np.round(eef[0], 4))
print("obj[0:3] - eef  :", np.round(obj[0, 0:3] - eef[0], 4))
print("obj[7:10]       :", np.round(obj[0, 7:10], 4))   # should match one of the above