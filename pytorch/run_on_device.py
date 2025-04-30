import torch

# 1) Pick device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2) Create two random 3×3 tensors on that device
a = torch.rand(3, 3, device=device)
b = torch.rand(3, 3, device=device)

# 3) Perform element-wise and matrix multiplication
elem_mul = a * b
mat_mul  = a @ b  # or torch.matmul(a, b)

# 4) Print results (they’ll live on GPU if available)
print("Tensor a:\n", a)
print("Tensor b:\n", b)
print("Element-wise multiply (a * b):\n", elem_mul)
print("Matrix multiply    (a @ b):\n", mat_mul)

