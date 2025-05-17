import torch
if torch.cuda.is_available():
   print("CUDA Available")
   device = torch.device('cuda')
else:
   print("CUDA is not available. Using CPU.")
   device = torch.device('cuda')
x = torch.tensor([[1.0,2.0],[3.0,4.0]], requires_grad=True)
y = x * 2
gradient = torch.tensor([[1.0,1.0],[0.5,0.5]])
y.backward(gradient)
print(f"Gradients of y with respect to x: {x.grad}")
