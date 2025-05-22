import torch
grad=False
x = torch.tensor([4.0,5.0,6.0])
print(f"Requires Gradient:{x.requires_grad}")
x.requires_grad_(True)
print(f"Requires Gradient after change:{x.requires_grad}")

