import torch


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

torch.manual_seed(123)
model = NeuralNetwork(50, 3)
print(model)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable model parameters: ", num_params)

print(model.layers[0].weight)
# print(len(model.layers[0].weight))
# print(model.layers[0].weight.shape)
# print(model.layers[0].bias)

X = torch.rand((1, 50))
# out = model(X)
# print(out)

#  When we use a model for inference (for instance, making predictions) rather than training, 
#  the best practice is to use the torch.no_grad() con- text manager. This tells PyTorch that it doesn’t
#  need to keep track of the gradients, which can result in significant savings in memory and computation
# with torch.no_grad():
#     out = model(X)

# If we want to compute class-membership probabilities for our predictions, we have to call the softmax function explicitly.
with torch.no_grad():
    out = torch.softmax(model(X), dim=1)

print(out)