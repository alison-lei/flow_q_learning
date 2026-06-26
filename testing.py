# import numpy as np
# import pickle
# from utils.datasets import Dataset

# with open('robomimic_datasets/lift_300_demos_scripted_flow_2026-06-10_14-29-49-254753.pkl', 'rb') as f:
#     data = pickle.load(f)

# t = data[0]
# obs = t['observations']['state'].squeeze()
# orig = t['infos']['original_state_obs']

# print("full obs dim:", obs.shape)
# print("full obs values:", obs)
# print()
# print("gripper:", np.array([orig['gripper_pose']]).shape, orig['gripper_pose'])
# print("privileged:", orig['privileged'].shape, orig['privileged'])
# print("tcp_vel:", orig['tcp_vel'].shape, orig['tcp_vel'])
# print("tcp_pose:", orig['tcp_pose'].shape, orig['tcp_pose'])
# print("wrist_force:", orig['wrist_force'].shape, orig['wrist_force'])
# print("wrist_torque:", orig['wrist_torque'].shape, orig['wrist_torque'])
# print()
# print("sum of all fields:", 
#     1 + orig['privileged'].shape[0] + orig['tcp_vel'].shape[0] + 
#     orig['tcp_pose'].shape[0] + orig['wrist_force'].shape[0] + orig['wrist_torque'].shape[0])

# train_dataset = Dataset.create(**train_dataset)
# # add this right after
# print("example_batch observations shape:", train_dataset.sample(1)['observations'].shape)
# print("example_batch actions shape:", train_dataset.sample(1)['actions'].shape)
# example_batch = train_dataset.sample(1)
# print("example_batch observations shape:", example_batch['observations'].shape)
# print("example_batch actions shape:", example_batch['actions'].shape)

import numpy as np
import pickle

file_path = "robomimic_datasets/lift_300_demos_scripted_flow_2026-06-10_14-29-49-254753.pkl"
with open(file_path, "rb") as f:
    data = pickle.load(f)
print(data[0])

t = data[0]
obs = t['observations']['state'][0]
orig = t['infos']['original_state_obs']

print("full obs:", obs)
print("gripper:", orig['gripper_pose'])
print("privileged:", orig['privileged'])
print("tcp_vel:", orig['tcp_vel'])
print("wrist_force:", orig['wrist_force'])
print("wrist_torque:", orig['wrist_torque'])

print(type(orig['gripper_pose']))
print(np.array(orig['gripper_pose']).shape)

print(data[1]["observations"]["state"][0])

print("dataset privileged:", t['infos']['original_state_obs']['privileged'])
print("dataset privileged shape:", t['infos']['original_state_obs']['privileged'].shape)

print("next data element", data[1])

# print("gripper shape:", gripper.shape, gripper)
# print("privileged shape:", privileged.shape)
# print("tcp_vel shape:", tcp_vel.shape)
# print("privileged values", privileged)
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
# print(np.array(orig['gripper_pose']).shape)
# print(data[1]["observations"]["state"][0])
# print("dataset privileged:", t['infos']['original_state_obs']['privileged'])
# print("dataset privileged shape:", t['infos']['original_state_obs']['privileged'].shape)
