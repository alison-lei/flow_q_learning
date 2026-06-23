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