import torch

mean = 2.0
std = 0.5
torch.set_printoptions(threshold=float('inf'))
tensor = torch.randn(100, 100, device='cuda') * std + mean
print(tensor)
x = tensor.to('cpu')

 