import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5 # small constant "epsilon" added to the variance, prevents division by zero during normalization
        self.scale = nn.Parameter(torch.ones(emb_dim)) # trainable param, LLM will adjust this during training
        self.shift = nn.Parameter(torch.zeros(emb_dim)) # trainable param, LLM will adjust this during training
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


torch.manual_seed(123)
torch.set_printoptions(sci_mode=False)

batch_example = torch.randn(2, 5)
print(batch_example)

ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)

# Verification that the mean = 0 and variance = 1
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)

print("Mean:\n", mean)
print("Variance:\n", var)