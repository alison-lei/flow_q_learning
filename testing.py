import numpy as np

data = np.load('your_dataset.npz')
print(data.files)  # see all keys
print(data['observations'].shape)  # total shape
print(data['observations'][0])     # first observation
print(data['actions'].shape)