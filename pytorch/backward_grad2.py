import torch
if torch.cuda.is_available():
   print("CUDA Available")
   device = torch.device('cuda')
else:
   print("CUDA is not available. Using CPU.")
   device = torch.device('cuda')
x = torch.tensor([1.0,2.0,3.],requires_grad=True)
y = x + 2
z = y.mean()
z.backward(retain_graph=True)

w = x * 3
loss = w.sum()
loss.backward()

print(f"Gradients: {x.grad}")
