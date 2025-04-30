import torch
if torch.cuda.is_available():
   device = torch.device('cuda')
else:
  device = torch.device('cpu')
x = torch.tensor([1,2,3],dtype=torch.float32).to(device)
y = torch.tensor([4,5,6],dtype=torch.float32).to(device)
z = x + y
print(z)
