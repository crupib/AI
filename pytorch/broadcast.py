import torch

# Tensor A: shape (3, 1)
A = torch.tensor([[1.0], [2.0], [3.0]])

# Tensor B: shape (1, 4)
B = torch.tensor([[10.0, 20.0, 30.0, 40.0]])

# Broadcasting addition: A + B => shape (3, 4)
C = A + B

print("Tensor A (3x1):")
print(A)
print("\nTensor B (1x4):")
print(B)
print("\nResult of A + B (broadcasted to 3x4):")
print(C)
