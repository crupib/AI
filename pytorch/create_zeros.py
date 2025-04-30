import torch

# assuming a is already defined as torch.rand(3,3)
a = torch.rand(3, 3)

# create a zero tensor with the same shape and dtype as a
zeros = torch.zeros_like(a)

print("Tensor a:\n", a)
print("Zero tensor:\n", zeros)

