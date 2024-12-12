import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import tiktoken


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        # Pre-tokenize texts
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # Truncate sequences if they are longer than max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]
        
        # Pad sequences to the longest sequence
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]
    
    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )
    
    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


tokenizer = tiktoken.get_encoding("gpt2")

# Exercise 6.1: Increasing the context length
# Simply set max_length to 1024 which is the max context length size in which the original GPT-2 was trained on
# max_length = 1024

train_dataset = SpamDataset(
    csv_file="train.csv",
    max_length=None,
    tokenizer=tokenizer
)

# print(train_dataset.max_length) # 120

val_dataset = SpamDataset(
    csv_file="validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

test_dataset = SpamDataset(
    csv_file="test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

# Setting up data loaders
num_workers = 0
batch_size = 8

torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False
)

# Verifying that the batches in the data loaders contain 8 training examples each, where each training example consists of 120 tokens
print("Train loader:")
num = 0
for train_input_batch, train_target_batch in train_loader:
    pass

print("Input batch dimensions:", train_input_batch.shape)
print("Label batch dimensions:", train_target_batch.shape)

print("\nValidation loader:")
for val_input_batch, val_target_batch in val_loader:
    pass

print("Input batch dimensions:", val_input_batch.shape)
print("Label batch dimensions:", val_target_batch.shape)

print("\nTest loader:")
for test_input_batch, test_target_batch in test_loader:
    pass

print("Input batch dimensions:", test_input_batch.shape)
print("Label batch dimensions:", test_target_batch.shape)

# Total number of batches and samples in each dataset
print(f"\nTraining batches: {len(train_loader)} \nTraining samples: {len(train_loader.dataset)}\n")
print(f"\nValidation batches: {len(val_loader)} \nValidation samples: {len(val_loader.dataset)}\n")
print(f"\nTest batches: {len(test_loader)} \nTest samples: {len(test_loader.dataset)}\n")


