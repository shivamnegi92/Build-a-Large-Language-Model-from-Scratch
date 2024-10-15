import torch


tensor0d = torch.tensor(1) # Creates a zero-dimensional tensor (scalar) from a Python integer

tensor1d = torch.tensor([1, 2, 3]) # Creates a one-dimensional tensor (vector) from a Python list

tensor2d = torch.tensor([[1, 2], # Creates a two-dimensional tensor from a nested Python list
                         [3, 4]])

tensor3d = torch.tensor([[[1, 2], [3, 4]], # Creates a three-dimensional tensor from a nested Python list
                         [[5, 6],  [7, 8]]])


print("0d tensor: \n", tensor0d, "\n")
print("1d tensor: \n", tensor1d, "\n")
print("2d tensor: \n", tensor2d, "\n")
print("3d tensor: \n", tensor3d, "\n")