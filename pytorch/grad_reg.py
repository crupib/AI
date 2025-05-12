import torch
X = torch.tensor([[1.0, 2.0, 3.0]],requires_grad=False)
Y = torch.tensor([[2.0, 4.0, 6.0]],requires_grad=False)
w = torch.randn((1,1), requires_grad=True)
b = torch.randn(1, requires_grad=True)

lr = 0.01
for epoch in range(1, 101):
    # element-wise scaling
    Y_pred = X * w + b
    loss = torch.mean((Y_pred - Y)**2)

    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}  Loss: {loss.item():.4f}")

print(f"\nLearned w = {w.item():.4f}, b = {b.item():.4f}")

