import torch
from torch.utils.data import Dataset
import tiktoken


# batching process.png -> 
#   Steps: 
#       2.1) Format data using prompt template.
#       2.2) Tokenize formatted data.
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))
    
    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


# batching process.png -> 
#   Steps: 
#       2.1) Format data using prompt template.
def format_input(entry):
    # Alpaca-style prompt formatting
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


# batching process.png -> 
#   Steps: 
#       2.3) Adjust to the same length with padding tokens.
def custom_collate_draft_1(batch, pad_token_id=50256, device="cpu"):
    
    # Find the longest sequence in the batch and increase 
    # the max length by +1, which will add one extra padding token below
    batch_max_length = max(len(item) +1 for item in batch)

    # Pad and prepare inputs
    inputs_list = []

    for item in batch:
        new_item = item.copy()
        
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        
        # Pad sequences to batch_max_length
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        
        # Via padded[:-1], we remove the extra padded token
        # that has been added via the +1 setting in batch_max_length
        # (the extra padding token will be relevant in later codes)
        inputs = torch.tensor(padded[:-1])
        inputs_list.append(inputs)

    # Convert list of inputs to tensor and transfer to target device
    inputs_tensor = torch.stack(inputs_list).to(device)
    return inputs_tensor


# batching process.png -> 
#   Steps: 
#       2.3) Adjust to the same length with padding tokens.
#       2.4) Create target token IDs for training.
def custom_collate_draft_2(batch, pad_token_id=50256, device="cpu"):
    
    # Find the longest sequence in the batch
    batch_max_length = max(len(item) +1 for item in batch)

    # Pad and prepare inputs
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        
        # Pad sequences to max_length
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )

        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs to tensor and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


# batching process.png -> 
#   Steps: 
#       2.3) Adjust to the same length with padding tokens.
#       2.4) Create target token IDs for training.
#       2.5) Replace padding tokens with placeholders.
def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"):
    
    # Find the longest sequence in the batch
    batch_max_length = max(len(item) +1 for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()

        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        
        # Pad sequences to max_length
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )

        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        # New: Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # New: Optionally truncate to maximum sequence length
        # This is useful if datasets are used that exceed the
        # 1,024-token context size supported by the GPT-2 model.
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


tokenizer = tiktoken.get_encoding("gpt2")

# Padding token
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})) # [50256]

# Testing the functions custom_collate_draft 1 & 2 and final custom_collate_fn
inputs_1 = [0, 1, 2, 3, 4]
inputs_2 = [5, 6]
inputs_3 = [7, 8, 9]

batch = (
    inputs_1,
    inputs_2,
    inputs_3,
)

print("*** Custom Collate Draft 1 ***")

print(custom_collate_draft_1(batch))
# tensor([[    0,     1,     2,     3,     4],
        # [    5,     6, 50256, 50256, 50256],
        # [    7,     8,     9, 50256, 50256]])


# Testing custom_collate_draft_2 function
print("\n*** Custom Collate Draft 2 ***")

inputs, targets = custom_collate_draft_2(batch)
print("\n", "Inputs:\n", inputs)
print("\n", "Targets:\n", targets)


# Testing custom_collate_fn function
print("\n*** Custom Collate Final ***")

inputs, targets = custom_collate_fn(batch)
print("\n", "Inputs:\n", inputs)
print("\n", "Targets:\n", targets)


# Let's see what this replacement by -100 accomplishes
# For illustration purposes, let's assume we have a small classification task with 2 class labels, 0 and 1, similar to chapter 6
# If we have the following logits values (outputs of the last layer of the model), we calculate the following loss

logits_1 = torch.tensor(
    [[-1.0, 1.0],  # 1st training example
     [-0.5, 1.5]]  # 2nd training example
)
targets_1 = torch.tensor([0, 1]) # Target class indices indicating the correct label for each example

loss_1 = torch.nn.functional.cross_entropy(logits_1, targets_1)
print("\n", "Loss 1:", loss_1)

# ---------------
logits_2 = torch.tensor(
    [[-1.0, 1.0],
     [-0.5, 1.5],
     [-0.5, 1.5]]  # New 3rd training example
)
targets_2 = torch.tensor([0, 1, 1])

loss_2 = torch.nn.functional.cross_entropy(logits_2, targets_2)
print("\n", "Loss 2:", loss_2)

# ---------------
# In the images directory, "ignore_index in cross-cross-entropy loss.png" explains why the loss is identical in loss_1 and loss_3
# as well as how ignore_index works internally within the torch.nn.functional.cross_entropy() function
targets_3 = torch.tensor([0, 1, -100])

loss_3 = torch.nn.functional.cross_entropy(logits_2, targets_3)
print("\n", "Loss 3:", loss_3)
print("loss_1 == loss_3:", loss_1 == loss_3)

# TODO: Exercise 7.2 Instruction and input masking
# After completing the chapter and fine-tuning the model with InstructionDataset,
# replace the instruction and input tokens with the -100 mask to use the instruction
# masking method illustrated in figure 7.13. Then evaluate whether this has a positive
# effect on model performance.
