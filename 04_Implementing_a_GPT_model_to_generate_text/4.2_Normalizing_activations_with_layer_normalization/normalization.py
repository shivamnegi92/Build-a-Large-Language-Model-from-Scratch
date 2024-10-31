import torch
import torch.nn as nn


torch.manual_seed(123)

batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)

print("\nLayer outputs:\n", out)

# Mean values for both row 1 and row 2
mean = out.mean(dim=-1, keepdim=True)

var = out.var(dim=-1, keepdim=True)
print("\nMean:\n", mean)
print("Variance:\n", var)

print("\n------------------------")
# Layer Normalization
out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
print("\nNormalized layer outputs:\n", out_norm)
print("Mean:\n", mean)
print("Variance:\n", var)

print("\n------------------------")
# Removing scientific notation
torch.set_printoptions(sci_mode=False)
print("Mean:\n", mean)
print("Variance:\n", var)

