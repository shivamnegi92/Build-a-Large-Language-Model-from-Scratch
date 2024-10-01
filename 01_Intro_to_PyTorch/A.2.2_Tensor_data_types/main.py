import torch


tensor1d = torch.tensor([1, 2, 3])
print(tensor1d.dtype) # torch.int64

floatvec = torch.tensor([1.0, 2.0, 3.0])
print(floatvec.dtype) # torch.float32

floatvec = tensor1d.to(torch.float32)
print(floatvec.dtype) # torch.float32