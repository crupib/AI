import torch
import numpy as np

numpy_array = np.array([1,2,3])

tensor_from_numpy = torch.from_numpy(numpy_array)
tensor_copy = torch.tensor(numpy_array)
print(f"Numpy array: {numpy_array}")
print(f"Tensor from NumPy: {tensor_from_numpy}")
print(f"Tensor copy: {tensor_copy}")

