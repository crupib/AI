import torch
if torch.cuda.is_available():
   device = torch.device('cuda')
   print("using cuda\n")
else:
  device = torch.device('cpu')
x = torch.tensor([1,2,3]).to(device)
y = torch.tensor([4,5,6]).to(device)
z = x + y
print(z)
