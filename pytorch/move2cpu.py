import torch
x = torch.tensor([1,2,3])
print(f"Tensor on CPU:{x.device}")
if torch.cuda.is_available():
   print("CUDA Available")
   device = torch.device('cuda')
else:
   print("CUDA is not available. Using CPU.")
   device = torch.device('cuda')
x = x.to(device)
print(f"Tensor on GPU: {x.device}")
x = x.to('cpu')
print(f"Tensor back on CPU: {x.device}")