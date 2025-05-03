import torch
if torch.cuda.is_available():
   device = torch.device('cuda')
else:
   device = torch.device('cpu')
x = torch.tensor([1,2,3])
y = torch.tensor([4,5,6])
addtion = x + y
print(f"Addtion:\n {addtion}")
subtraction = x - y
print(f"subtraction:\n {subtraction}")
multiplication = x * y
print(f"multiplication:\n {multiplication}")
division = x / y
print(f"division:\n {division}")
expo = x**2
print(f"expo:\n {expo}")
