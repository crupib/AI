import torch
grad=True
X = torch.tensor([[1.0],[2.0],[3.0]], requires_grad=False)
Y = torch.tensor([[2.0],[4.0],[6.0]],requires_grad=False)
w = torch.randn((1,1),requires_grad=True)
b = torch.randn((1),requires_grad=True)
learning_rate = 0.001
num_epochs = 100
for epoch in range(num_epochs):
  Y_pred = torch.matmul(X,w)+b
  loss=torch.mean((Y_pred - Y)**2)
  loss.backward()
  with torch.no_grad():
    w -= learning_rate*w.grad
    b -= learning_rate*b.grad
    w.grad.zero_()
    b.grad.zero_()
  if (epoch+1) % 10 == 0:
   print(f'Epoch[{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
  print(f"Learned weight:{w.item():.4f}, bias:{b.item():.4f}")
print(f"Tensor:{X}")
print(f"Requires Gradient:{X.requires_grad}")
