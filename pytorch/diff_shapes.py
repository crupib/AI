import torch

# Parameters
mean = 2.0
std = 0.5

# Create tensor on GPU
tensor = torch.randn(3, 3, device='cuda') * std + mean

# Reshape to 1D tensor
tensor_1d = tensor.view(-1)

# Or alternatively:
# tensor_1d = tensor.reshape(-1)

print(tensor_1d)
