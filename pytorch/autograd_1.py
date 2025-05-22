import torch
grad=True
x = torch.tensor([1.0,2.0,3.0], requires_grad=True)
print(f"Tensor:{x}")
print(f"Requires Gradient:{x.requires_grad}")
