import torch
if torch.cuda.is_available():
   print("CUDA Available")
   device = torch.device('cuda')
else:
   print("CUDA is not available. Using CPU.")
   device = torch.device('cpu')
x = torch.tensor([1.0,2.0,3.0],requires_grad=False)
y = x * 2
loss = y.sum()

print(f"Gradients: {x.grad}")
