import torch


tensor2d = torch.tensor([[1, 2, 3],
                         [4, 5, 6]])

print(tensor2d)

print(tensor2d.shape) # [2, 3], 2 rows by 3 columns

# print(tensor2d.reshape(3, 2)) # [3, 2], 3 rows by 2 columns, reshaping tensor

print(tensor2d.view(3, 2)) # [3, 2], 3 rows by 2 columns, more common command for reshaping tensors in PyTorch

print("Transpose: \n ", tensor2d.T, "\n") # Transpose, flip across its diagonal

print(tensor2d.matmul(tensor2d.T)) # Matrix multiplication

print(tensor2d @ tensor2d.T) # Matrix multiplication (compact syntax)