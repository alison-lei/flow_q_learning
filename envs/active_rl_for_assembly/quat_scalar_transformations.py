import numpy as np

def scalar_last_to_scalar_first(scalar_last_quat):
    scalar_last_quat= np.asarray(scalar_last_quat)
    return scalar_last_quat[..., [3, 0, 1, 2]]

def scalar_first_to_scalar_last(scalar_first_quat):
    scalar_first_quat = np.asarray(scalar_first_quat)
    return scalar_first_quat[..., [1, 2, 3, 0]]