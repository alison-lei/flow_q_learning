import robosuite
import numpy as np
import pickle
import itertools
import matplotlib.pyplot as plt
from utils.datasets import Dataset
from octo.model.octo_model import OctoModel
from scipy.spatial.transform import Rotation as R
from envs.robomimic_utils import make_eval_env


with open('robomimic_datasets/gripOpenFalse_20_demos.pkl', 'rb') as f:
    data = pickle.load(f)
episode_lengths = []
counter = 0
octo_model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small")
tasks = octo_model.create_tasks(texts=["pick up the red block"])

def add_embeddings(observation):
    image_primary = observation["side_policy_256"][np.newaxis, ...]
    image_wrist = observation["wrist_1"][np.newaxis, ...]
    timestep_pad_mask = np.array([[True, True]])
    octo_obs = {
        "image_primary": image_primary,
        "image_wrist": image_wrist,
        "timestep_pad_mask": timestep_pad_mask
    }

    transformer_outputs = octo_model.sample_transformer(octo_obs, tasks, train=False)
    action_embeddings = transformer_outputs[:, -1, :]                                  # (1, embed_dim)
    observation = dict(observation)
    observation["embeddings"] = action_embeddings[0]
    return action_embeddings[0]

def add_embeddings_6(observation):
    octo_obs = {
        "image_primary": observation["side_policy_256"][np.newaxis, ...],
        "image_wrist": observation["wrist_1"][np.newaxis, ...],
    }
    timestep_pad_mask = np.array([[True, True]])
    transformer_outputs = octo_model.run_transformer(octo_obs, tasks, timestep_pad_mask, train=False)
    action_embeddings = transformer_outputs["readout_action"].tokens.mean(axis=-2)  # (1, 2, embed_dim)
    action_embeddings = action_embeddings[:, -1, :]                                  # (1, embed_dim)
    observation = dict(observation)
    observation["embeddings"] = action_embeddings[0]
    return action_embeddings[0]


# training_transition = data[0]
# stored = training_transition["embeddings"]
# reconstructed = add_embeddings(training_transition["observations"])["embeddings"]
# print(np.allclose(stored, reconstructed, atol=1))

# info = training_transition["infos"]["original_state_obs"]
# state = training_transition["next_observations"]["state"][-1]   # last window frame, (19,)

# for name, val in sorted(info.items()):
#     val = np.atleast_1d(val)
#     print(f"{name}: dim={val.shape}, value={val[:3]}...")

# print("state:", state)

# print(len(data))
first_transition = data[0]   # first step of a trajectory
print(first_transition)

# for i in range(466):
#     counter += 1
#     if data[i]["infos"]["succeed"] == 1:
#         episode_lengths.append(counter)
#         counter = 0

# print(episode_lengths)


import jax
# load first transition
d = pickle.load(open("robomimic_datasets/gripOpenFalse_20_demos.pkl","rb"))[0]  # adjust indexing
stored = np.asarray(d["embeddings"])              # 512-dim, baked in at collection
imgs   = d["observations"]                          # side_policy_256 + wrist_1, window=2

# recompute from the SAME images using your CURRENT env's octo path
recomputed = add_embeddings_6(imgs)             # exact call eval uses
recomputed = np.asarray(jax.device_get(recomputed)).ravel()
s = stored.ravel()
print("cosine:", np.dot(s,recomputed)/(np.linalg.norm(s)*np.linalg.norm(recomputed)))
print("max|Δ|:", np.abs(s-recomputed).max())

def cosine(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else np.nan

def load(p):
    d = pickle.load(open(p, "rb"))
    return [t for ep in d for t in ep] if isinstance(d[0], list) else d

a = load("robomimic_datasets/gripOpenFalse_noLR_mujoco3_20_demos.pkl")
b = load("robomimic_datasets/lift_20_demos_scripted_2026-05-11_15-23-51-466619_CORL.pkl")

# for idx in [183, 303, 282, 208, 424]:
#     fig, ax = plt.subplots(1, 2, figsize=(8, 4))
#     ax[0].imshow(a[idx]["observations"]["side_policy_256"][0]); ax[0].set_title(f"A idx {idx}")
#     ax[1].imshow(b[idx]["observations"]["side_policy_256"][0]); ax[1].set_title(f"B idx {idx}")
#     plt.savefig(f"compare_{idx}.png"); plt.close()

img = b[181]["observations"]  # same as b[181]'s image, confirmed identical
current_emb = add_embeddings(img)

print("current vs A stored:", cosine(current_emb, a[181]["embeddings"]))
print("current vs B stored:", cosine(current_emb, b[181]["embeddings"]))
print("A gripper_pose:", a[181]["infos"]["original_state_obs"]["gripper_pose"])
print("B gripper_pose:", b[181]["infos"]["original_state_obs"]["gripper_pose"])

diff = np.abs(a[181]["observations"]["side_policy_256"].astype(int)
             - b[181]["observations"]["side_policy_256"].astype(int)).sum(axis=-1)
plt.imshow(diff[0]); plt.colorbar(); plt.savefig("diffmap_wrist_frame0.png"); plt.close()
plt.imshow(diff[1]); plt.colorbar(); plt.savefig("diffmap_wrist_frame1.png"); plt.close()# print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
# with open('robomimic_datasets/lift_20_demos_scripted_2026-05-11_15-23-51-466619_CORL.pkl', 'rb') as f:
#     data = pickle.load(f)

# print(data[0])

# reset_pose = first_transition["reset_pose"]

# env = make_eval_env("robomimic-lift")
# obs, info = env.reset(options={"reset_pose": reset_pose, "retry_from_same_state": True})

# live_state = obs["state"]
# stored_state = first_transition["observations"]["state"]

# print("live: ", live_state)
# print("stored:", stored_state)
# print("match:", np.allclose(live_state, stored_state, atol=1e-3))
# candidates = [
#     "hf://rail-berkeley/octo-small", // this
#     "hf://rail-berkeley/octo-small-1.5",
#     "hf://rail-berkeley/octo-base",
#     "hf://rail-berkeley/octo-base-1.5",
# ]

# for path in candidates:
#     m = OctoModel.load_pretrained(path)
#     t = m.create_tasks(texts=["pick up the red block"])
#     obs = {
#         "image_primary": training_transition["observations"]["side_policy_256"][np.newaxis, ...],
#         "image_wrist": training_transition["observations"]["wrist_1"][np.newaxis, ...],
#     }
#     out = m.run_transformer(obs, t, np.array([[True, True]]), train=False)
#     recon = out["readout_action"].tokens.mean(axis=-2)[:, -1, :][0]
#     print(path, "-> matches:", np.allclose(stored, recon, atol=1e-2),
#           "| max abs diff:", np.max(np.abs(stored - recon)))

# print(t)
# # obs = t['observations']['state'].squeeze()
# # orig = t['infos']['original_state_obs']

# # print("full obs dim:", obs.shape)
# # print("full obs values:", obs)
# # print()
# # print("gripper:", np.array([orig['gripper_pose']]).shape, orig['gripper_pose'])
# # print("privileged:", orig['privileged'].shape, orig['privileged'])
# # print("tcp_vel:", orig['tcp_vel'].shape, orig['tcp_vel'])
# # print("tcp_pose:", orig['tcp_pose'].shape, orig['tcp_pose'])
# # print("wrist_force:", orig['wrist_force'].shape, orig['wrist_force'])
# # print("wrist_torque:", orig['wrist_torque'].shape, orig['wrist_torque'])
# # print()
# # print("sum of all fields:", 
# #     1 + orig['privileged'].shape[0] + orig['tcp_vel'].shape[0] + 
# #     orig['tcp_pose'].shape[0] + orig['wrist_force'].shape[0] + orig['wrist_torque'].shape[0])

# # train_dataset = Dataset.create(**train_dataset)
# # # add this right after
# # print("example_batch observations shape:", train_dataset.sample(1)['observations'].shape)
# # print("example_batch actions shape:", train_dataset.sample(1)['actions'].shape)
# # example_batch = train_dataset.sample(1)
# # print("example_batch observations shape:", example_batch['observations'].shape)
# # print("example_batch actions shape:", example_batch['actions'].shape)

# import numpy as np
# import pickle
# from envs import robomimic_utils

# file_path = "robomimic_datasets/lift_300_demos_scripted_flow_2026-06-10_14-29-49-254753.pkl"
# with open(file_path, "rb") as f:
#     data = pickle.load(f)
# print(data[0])

# t = data[0]
# obs = t['observations']['state'][0]
# orig = t['infos']['original_state_obs']

# print("full obs:", obs)
# print("gripper:", orig['gripper_pose'])
# print("privileged:", orig['privileged'])
# print("tcp_vel:", orig['tcp_vel'])
# print("wrist_force:", orig['wrist_force'])
# print("wrist_torque:", orig['wrist_torque'])

# print(type(orig['gripper_pose']))
# print("!!!!!", np.array(orig['gripper_pose']).shape)

# print(data[1]["observations"]["state"][0])

# print("dataset privileged:", t['infos']['original_state_obs']['privileged'])
# print("dataset privileged shape:", t['infos']['original_state_obs']['privileged'].shape)

# print("next data element", data[1])

# print("gripper pose", data[1]["infos"]["original_state_obs"]["gripper_pose"])


# print("////////////////////////")


# env = robomimic_utils.make_eval_env("robomimic-lift")
# obs, _ = env.reset()
# print(obs)
# dataset_action = t["actions"]
# # pick a dataset transition collected from a comparable reset, feed its stored action
# next_obs, *_ = env.step(dataset_action)   # dataset_action = t["actions"], the (7,) EE-frame vector
# # next_obs should track t["next_observations"]["state"] on gripper(0) and privileged(1:7);
# # tcp_pose(7:13) should now match too, since the frame is reproduced
# print(next_obs)
# print(data[1]["observations"]["state"])

# for i in range(150, 300):
#     gripper_pose = data[i]["observations"]["state"][0][0]
#     # print(gripper_pose)
#     arr = data[i]["next_observations"]["state"][0][0]
#     gripper_val = data[i]["infos"]["original_state_obs"]["gripper_pose"]
#     # print(gripper_val)
#     # if (abs(gripper_val - arr)) > 0.1:
#     #     print("/////////////////////////////////////")
#     #     print(data[i])
#         # print("//////////////////////////////////")
#         # print(data[i-1])
#         # print("not the same as pose")
#         # print("arr: ", arr)
#         # print("gripper_val: ", gripper_val)
#         # gripper_pose is equal to index 0 of next_observations
#     # if arr != 1:
#     #     print("hello: ", arr)
    
#     # gaction = data[i]['actions'][6]
    
#     # if gaction != 0: 
#     #     print("action gripper:", gaction, i)
#     #     break

# obs = next_obs
# print(obs)
# dataset_action = data[17]["actions"]
# dataset_action[6] = -1
# # pick a dataset transition collected from a comparable reset, feed its stored action
# next_obs, *_ = env.step(dataset_action)   # dataset_action = t["actions"], the (7,) EE-frame vector
# # next_obs should track t["next_observations"]["state"] on gripper(0) and privileged(1:7);
# # tcp_pose(7:13) should now match too, since the frame is reproduced
# print(next_obs)
# dataset_action[6] = 1
# for i in range(4):
#     next_obs, *_ = env.step(dataset_action)
#     print(next_obs)
# # action controlling gripper works and self.gripper_binary_state works as a latch to ensure that once or closed or
# # open it wont close or open further respectively, so when you set the action then it just stays that value
# print("////////////////////////////////////")

# obs, _ = env.reset()
# a = np.zeros(7, dtype=np.float32); a[0] = 1.0
# for _ in range(5):
#     obs, *_ = env.step(a)
#     print(np.round(obs[7:10], 4))

# for i in range(300):
#     if data[i]["infos"]["succeed"] == 1:
#         print(data[i]["infos"]["succeed"]) 
#         print(data[i])
#         break