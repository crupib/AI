import torch
if torch.cuda.is_available():
   device = torch.device('cuda')
else:
   device = torch.device('cpu')
arange_tensor = torch.arange(0,10,2)
print(f"arange tensor:\n {arange_tensor}")
linspace_tensor = torch.linspace(0,1,5)
print(f"linspace Tensor:\n {linspace_tensor}")
