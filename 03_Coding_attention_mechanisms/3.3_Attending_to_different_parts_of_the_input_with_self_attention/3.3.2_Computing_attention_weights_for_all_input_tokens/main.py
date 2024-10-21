import torch


inputs = torch.tensor([
    [0.43, 0.15, 0.89], # Your      (x^1)
    [0.55, 0.87, 0.66], # journey   (x^2)
    [0.57, 0.85, 0.64], # starts    (x^3)
    [0.22, 0.58, 0.33], # with      (x^4)
    [0.77, 0.25, 0.10], # one       (x^5)
    [0.05, 0.80, 0.55]  # step      (x^6)
])

print(inputs.shape)

attn_scores = torch.empty(6, 6)

# For Loop Method
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

# Matrix Multiplication
attn_scores = inputs @ inputs.T
print("\nAttention Scores:\n", attn_scores)

# Normalized
attn_weights = torch.softmax(attn_scores, dim=-1)
print("Attention Weights:\n", attn_weights, "\n")

# Verification of Sum to 1
row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 Sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=-1))

# All Context Vectors
all_context_vecs = attn_weights @ inputs
print("\nAll Context Vectors:\n", all_context_vecs)


# Verification of 2nd context vector
query = inputs[1] # journey   (x^2)
attn_scores_2 = torch.empty(inputs.shape[0]) # shape[0] = 6

for i, x_i, in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)

attn_weights_2_torch = torch.softmax(attn_scores_2, dim=0)

# Calculating the context vector for the 2nd input (journey (x^2))
query = inputs[1]
context_vec_2 = torch.zeros(query.shape) # shape = 3
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2_torch[i] * x_i

print("\nPrevious 2nd Context Vector:", context_vec_2)
