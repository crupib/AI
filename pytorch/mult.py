import torch

# Create two random 3×3 tensors
a = torch.rand(3, 3)
b = torch.rand(3, 3)

# Element-wise multiplication
elem_mul = a * b

# Matrix multiplication
mat_mul = a @ b  # equivalent to torch.matmul(a, b)

print("Tensor a:\n", a)
print("Tensor b:\n", b)
print("Element-wise multiply:\n", elem_mul)
print("Matrix multiply:\n", mat_mul)

