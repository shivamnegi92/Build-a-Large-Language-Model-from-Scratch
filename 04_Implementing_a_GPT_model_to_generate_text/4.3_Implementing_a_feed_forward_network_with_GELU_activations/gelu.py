import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


gelu = GELU()
relu = nn.ReLU()

x = torch.linspace(-3, 3, 100) # creates 100 sample data points in the range -3 to 3
y_gelu = gelu(x)
y_relu = relu(x)

plt.figure(figsize=(8, 3))

for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    # plt.plot(x, y, marker='o', linestyle='-', markersize=3)  # Add marker='o' and markersize=3
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)

plt.tight_layout()
plt.show()