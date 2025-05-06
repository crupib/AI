import torch
if torch.cuda.is_available():
   device = torch.device('cuda')
else:
   device = torch.device('cpu')

matrix = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
print(f"Matrix \n{matrix}")
element = matrix[0,1]
print(f"Single element: {element}")
row = matrix[1,:]
print(f"Row: {row}")
column = matrix[:,2]
print(f"Column: {column}")
submatrix = matrix[0:2,0:2]
print(f"Submatrix:\n{submatrix}")

