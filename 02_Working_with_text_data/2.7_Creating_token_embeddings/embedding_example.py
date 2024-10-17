import torch


input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

print(embedding_layer.weight)

print("\n", embedding_layer(torch.tensor([3])))

print("\n", embedding_layer(input_ids))

# Understanding the Difference Between Embedding Layers and Linear Layers (Bonus)
# https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/03_bonus_embedding-vs-matmul/embeddings-and-linear-layers.ipynb
