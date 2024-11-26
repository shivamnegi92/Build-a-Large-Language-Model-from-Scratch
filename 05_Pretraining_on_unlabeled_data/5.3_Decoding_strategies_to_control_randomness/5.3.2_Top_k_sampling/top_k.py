import torch


# Assume the LLM is given the start context "every effort moves you" and
# generates the following next-token logits:
next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)

top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)
print("Top logits:", top_logits)
print("Top positions:", top_pos)

new_logits = torch.where(
    condition=next_token_logits < top_logits[-1], # identifies logits less than the minimum in the top 3
    input=torch.tensor(float("-inf")), # assigns -inf to these lower logits
    other=next_token_logits # retains the original logits for all other tokens
)

print(new_logits)

# An alternative, slightly more efficient implementation of the previous code
new_logits_alt = torch.full_like( # create tensor containing -inf values
    next_token_logits, -torch.inf
)
new_logits_alt[top_pos] = next_token_logits[top_pos] # copy top k values into the -inf tensor

print(new_logits_alt)

# -----
topk_probas = torch.softmax(new_logits, dim=0)
print(topk_probas)

