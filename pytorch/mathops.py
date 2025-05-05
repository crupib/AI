import torch
if torch.cuda.is_available():
   device = torch.device('cuda')
else:
   device = torch.device('cpu')
x = torch.tensor([1,2,3])
y = torch.tensor([4,5,6])
print(x)
print(y)
addtion = x + y
print(f"Addtion:\n {addtion}")
print(x)
print(y)
subtraction = x - y
print(f"subtraction:\n {subtraction}")
print(x)
print(y)
multiplication = x * y
print(f"multiplication:\n {multiplication}")
print(x)
print(y)
division = x / y
print(f"division:\n {division}")
print(x)
expo = x**2
print(f"expo:\n {expo}")
