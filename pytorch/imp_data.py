import torch
if torch.cuda.is_available():
   device = torch.device('cuda')
else:
  device = torch.device('cpu')
list_data = [1,2,3,4,5]
tensor_from_list = torch.tensor(list_data)
print(f"Tensor from list: {tensor_from_list}")
tuple_data = (6,7,8,9,10)
tensor_from_tuple = torch.tensor(tuple_data)
print(f"Tensor from tuple: {tensor_from_tuple}")
import numpy as np
numpy_array = np.array([11,12,13,14,15])
tensor_from_numpy = torch.tensor(numpy_array)
print(f"Tensor from Numpy: {tensor_from_numpy}")
float_tensor = torch.tensor([1,2,3],dtype=torch.float32)
print(f"Float Tensor: {float_tensor}")
