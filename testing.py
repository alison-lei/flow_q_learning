import numpy as np, h5py
from os.path import expanduser
from envs.robomimic_utils import make_env, low_dim_keys

env = make_env("lift-mh-low_dim")
f = h5py.File(expanduser("~/.robomimic/lift/mh/low_dim_v15.hdf5"), "r")
demo = list(f["data"].keys())[0]

env.reset()
env.env.reset_to({"states": np.array(f[f"data/{demo}/states"])[0]})  # exact recorded state
live = env.env.get_observation()                                     # obs computed by robosuite 1.5.1

for k in low_dim_keys["low_dim"]:
    stored = np.array(f[f"data/{demo}/obs/{k}"])[0]
    got = np.asarray(live[k])
    print(k)
    print("  stored:", np.round(stored, 4))
    print("  live  :", np.round(got, 4))
    print("  |diff|:", np.round(np.abs(got - stored), 4).max())