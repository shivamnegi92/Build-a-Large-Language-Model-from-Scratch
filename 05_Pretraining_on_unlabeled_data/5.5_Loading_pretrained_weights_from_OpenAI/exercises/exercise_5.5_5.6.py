import tiktoken
import torch
from exercise_main import (
    init_model, 
    create_dataloader_v1, 
    calc_loss_loader,
    generate,
    text_to_token_ids,
    token_ids_to_text)


# Exercise 5.5: Training and validation set losses of the pretrained model

# We can use the following code to calculate the training and validation set losses of the GPT model:
#   train_loss = calc_loss_loader(train_loader, gpt, device)
#   val_loss = calc_loss_loader(val_loader, gpt, device)

# The resulting losses for the 124M parameter are as follows:
#   Training loss: 3.754748503367106
#   Validation loss: 3.559617757797241

# The main observation is that the training and validation set performances are in the same ballpark
# This can have multiple explanations:
# 1. The Verdict was not part of the pretraining dataset when OpenAI trained GPT-2. 
#    Hence, the model is not explicitly overfitting to the training set and performs similarly well on The Verdict's training and validation set portions. 
#    (The validation set loss is slightly lower than the training set loss, which is unusual in deep learning. 
#    However, it's likely due to random noise since the dataset is relatively small. In practice, if there is no overfitting, 
#    the training and validation set performances are expected to be roughly identical).

# 2. The Verdict was part of GPT-2's training dataset. In this case, we can't tell whether the model is overfitting the training data 
#    because the validation set would have been used for training as well. To evaluate the degree of overfitting, we'd need a new dataset 
#    generated after OpenAI finished training GPT-2 to make sure that it couldn't have been part of the pretraining.


GPT_CONFIG_124M = {
    "vocab_size": 50257,        # Vocabulary size
    "context_length": 256,      # Context length (shortened, orig: 1024)
    "emb_dim": 768,             # Embedding dimension
    "n_heads": 12,              # Number of attention heads
    "n_layers": 12,             # Number of layers
    "drop_rate": 0.1,           # Dropout rate
    "qkv_bias": False           # Query-Key-Value bias
}

torch.manual_seed(123)

tokenizer = tiktoken.get_encoding("gpt2")

gpt, device, NEW_CONFIG = init_model() # Default: model_size="124M", model_name="gpt2-small (124M)"
# gpt, device, NEW_CONFIG = init_model(model_size="1558M", model_name="gpt2-xl (1558M)") # Model will be a 6 gig file.

file_path = "the-verdict.txt"

with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

# Train/validation ratio
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]


train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

# Takes a long time to execute on 1558M model
# train_loss = calc_loss_loader(train_loader, gpt, device)
# val_loss = calc_loss_loader(val_loader, gpt, device)

# print("Training loss:", train_loss)
# print("Validation loss:", val_loss)

#  --------  Exercise 5.6: Trying larger models  --------
# In the main chapter, we experimented with the smallest GPT-2 model, which has only 124M parameters
# The reason was to keep the resource requirements as low as possible
# However, you can easily experiment with larger models with minimal code changes
# For example, instead of loading the 1558M instead of 124M model in chapter 5, the only 2 lines of code that we have to change are

torch.manual_seed(123)

token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# 124M model output text:
# " Every effort moves you toward finding an ideal new way to practice something!

# What makes us want to be on top of that?"

# -----------------------------------------------------

# 1558M model output text:
# "Every effort moves you toward finding an ideal life. You don't have to accept your current one at once, because if you do you'll never"