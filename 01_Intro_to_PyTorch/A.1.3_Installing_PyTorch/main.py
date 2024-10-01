import torch

print(torch.cuda.is_available()) # False if no GPU

print(torch.__version__) # Version #

print(torch.backends.mps.is_available()) # Check whether your Mac supports PyTorch acceleration with its Apple Silicon chip