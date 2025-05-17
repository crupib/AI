import torch
if torch.cuda.is_available():
   print("CUDA Available")
   device = torch.device('cuda')
else:
   print("CUDA is not available. Using CPU.")
   device = torch.device('cpu')
x = torch.tensor([1.0,2.0,3.0],requires_grad=True)
y = torch.tensor([4.0,5.0,6.0],requires_grad=True)

z = x * y
loss = z.sum()
loss.backward()

print(f"Gradients of loss with respect to x: {x.grad}")

print(f"Gradients of loss with respect to y: {y.grad}")

