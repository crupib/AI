import torch
import torch.nn as nn

class SimpleNet(nn.Module):
   def __init__(self):
     super(SimpleNet,self).__init__()
     self.linear = nn.Linear(10,5)
   def forward(self,x):
     return self.linear(x)
model = SimpleNet()

if torch.cuda.is_available():
   device = torch.device('cuda')
else:
   device = torch.device('cuda')

model.to(device)

input_data = torch.randn(1,10).to(device)
output = model(input_data)
print(f"Output on {output.device}")