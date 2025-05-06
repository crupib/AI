import torch
if torch.cuda.is_available():
   device = torch.device('cuda')
else:
   device = torch.device('cpu')

x = torch.arange(12)
reshaped_matrix = x.reshape((3,4))
print(f"Reshaped matrix:\n{reshaped_matrix}")
reshaped_matrix_2 = x.reshape((4,3))
print(f"Reshaped matrix 2:\n{reshaped_matrix_2}")
reshaped_matrix_3 = x.reshape((-1,3))
print(f"Inferred shape:\n{reshaped_matrix_3}")
