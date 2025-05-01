import torch
if torch.cuda.is_available():
   device = torch.device('cuda')
else:
   device = torch.device('cpu')
rand_tensor = torch.rand((2,3))
print(f"Random tensor:\n {rand_tensor}")
randn_tensor = torch.randn((3,2))
print(f"Random tensor(normal):\n {randn_tensor}")
randint_tensor = torch.randint(0,10,(2,2))
print(f"Random integer tensor:\n {randint_tensor}")
