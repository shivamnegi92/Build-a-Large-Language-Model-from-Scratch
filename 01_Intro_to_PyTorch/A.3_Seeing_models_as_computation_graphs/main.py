import torch
import torch.nn.functional as F


y = torch.tensor([1.0]) # true label
x1 = torch.tensor([1.1]) # input feature
w1 = torch.tensor([2.2]) # weight parameter
b = torch.tensor([0.0]) # bias unit
z = x1 * w1 + b # net input
a = torch.sigmoid(z) # activation and output

print(z)
print(a)

loss = F.binary_cross_entropy(a, y)
print(loss)
