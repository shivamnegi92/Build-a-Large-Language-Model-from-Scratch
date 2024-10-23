import torch.nn as nn
import torch


class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    
    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec


inputs = torch.tensor([
    [0.43, 0.15, 0.89], # Your      (x^1)
    [0.55, 0.87, 0.66], # journey   (x^2)
    [0.57, 0.85, 0.64], # starts    (x^3)
    [0.22, 0.58, 0.33], # with      (x^4)
    [0.77, 0.25, 0.10], # one       (x^5)
    [0.05, 0.80, 0.55]  # step      (x^6)
])

torch.manual_seed(789)
d_in = inputs.shape[1] # 3
d_out = 2
sa_v2 = SelfAttention_v2(d_in, d_out)

# sa_v2(inputs)

# Manually Getting weights
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print("\nAttention Weights:\n", attn_weights)

# Creating Mask
context_length = attn_scores.shape[0] # 6
mask_simple = torch.tril(torch.ones(context_length, context_length))
print("\nMask Simple:\n", mask_simple)

# Multiply mask with attention weights to zero-out the values above the diagonal
masked_simple = attn_weights * mask_simple
print("\nMasked Simple (zero-out values above diag):\n", masked_simple)

# Normalize the attention weights to sum up to 1 again in each row
row_sums = masked_simple.sum(dim=-1, keepdim=True)
print("\nRow Sums:\n", row_sums)
masked_simple_norm = masked_simple / row_sums
print("\nNormalized Masked Attention Weights:\n", masked_simple_norm)

# Masking with 1's above the diagonal and replacing the 1s with negativt infinity (-inf) values, more efficient
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print("\nAlternate Masking Method with -inf:\n", masked)

# Normalizing alternate masked attention scores
attn_weights_different = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
print("\nAlternate Normalized Masked Attention Weights:\n", attn_weights_different)