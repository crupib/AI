import torch
if torch.cuda.is_available():
   device = torch.device('cuda')
else:
  device = torch.device('cpu')
zeros_tensor = torch.zeros((2,3))
print(f"Zeros tensor:\n {zeros_tensor}")
ones_tensor = torch.ones((3,2))
print(f"Ones tensor:\n {ones_tensor}")
full_tensor = torch.full((2,2),7.0)
print(f"Full tensor:\n{full_tensor}")
identity_matrix = torch.eye(3)
print(f"Identity matrix:\n{identity_matrix}")

