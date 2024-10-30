
GPT_CONFIG_124M = {
    "vocab_size": 50257,        # Vocabulary size
    "context_length": 1024,     # Context length
    "emb_dim": 768,             # Embedding dimension
    "n_heads": 12,              # Number of attention heads
    "n_layers": 12,             # Number of layers
    "drop_rate": 0.1,           # Dropout rate
    "qkv_bias": False           # Query-Key-Value bias
}


# vocab_size: refers to a vocabulary of 50,257 words, as used by the BPE tokenizer (see chapter 2).

# context_length: denotes the maximum number of input tokens the model can handle via the positional embeddings (see chapter 2).

# emb_dim: represents the embedding size, transforming each token into a 768- dimensional vector.

# n_heads: indicates the count of attention heads in the multi-head attention mechanism (see chapter 3).

# n_layers: specifies the number of transformer blocks in the model, which we will cover in the upcoming discussion.

# drop_rate: indicates the intensity of the dropout mechanism (0.1 implies a 10% random drop out of hidden units) to prevent overfitting (see chapter 3).

# qkv_bias: determines whether to include a bias vector in the Linear layers of the multi-head attention for query, key, and value computations. 
    # We will initially disable this, following the norms of modern LLMs, but we will revisit it in chapter 6 when we load pretrained GPT-2 weights 
    # from OpenAI into our model (see chapter 6).
