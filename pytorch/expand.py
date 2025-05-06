import torch
if torch.cuda.is_available():
   device = torch.device('cuda')
else:
   device = torch.device('cpu')

x = torch.tensor([1,2,3])
y = 2
result = x + y

print(f"Broadcasting addtion:\n{result}")
A = torch.tensor(([[1,2,3],[4,5,6]]))
b = torch.tensor([10,20,30])
result = A + b
print(f"Broadcasting matrix:\n{result}")
