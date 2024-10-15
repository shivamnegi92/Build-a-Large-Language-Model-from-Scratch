import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


class GPTDataSetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Create dataset
    dataset = GPTDataSetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    
    return dataloader


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print("\nFirst Batch:", first_batch)

second_batch = next(data_iter)
print("Second Batch:", second_batch)

# Exercise 2.2
dataloader2 = create_dataloader_v1(raw_text, batch_size=1, max_length=2, stride=2, shuffle=False)
data_iter2 = iter(dataloader2)
first_batch2 = next(data_iter2)
print("\nFirst Batch 2:", first_batch2)

second_batch2 = next(data_iter2)
print("Second Batch 2:", second_batch2)

dataloader3 = create_dataloader_v1(raw_text, batch_size=1, max_length=8, stride=2, shuffle=False)
data_iter3 = iter(dataloader3)
first_batch3 = next(data_iter3)
print("\nFirst Batch 3:", first_batch3)

second_batch3 = next(data_iter3)
print("Second Batch 3:", second_batch3)

# --------------------------------------------------------------------------------

dataloader_final = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iter_final = iter(dataloader_final)
inputs, targets = next(data_iter_final)
print("\nInputs:\n", inputs)
print("\nTargets:\n", targets)