import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


torch.manual_seed(123)

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(

            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # Output layer
            torch.nn.Linear(20, num_outputs)
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits

model = NeuralNetwork(2, 2)
model.load_state_dict(torch.load("model.pth"))

# If GPU present
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# If Apple Silicon Chip, MPS = Metal Performance Shader
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(device)

# CPU addition
tensor_1 = torch.tensor([1., 2., 3.])
tensor_2 = torch.tensor([4., 5., 6.])
print(tensor_1 + tensor_2)


tensor_1 = tensor_1.to("mps")
tensor_2 = tensor_2.to("mps")
print(tensor_1 + tensor_2)

# tensor_2 = tensor_2.to("cpu") # Will crash, tensors need to be on the same device
# print(tensor_1 + tensor_2)

