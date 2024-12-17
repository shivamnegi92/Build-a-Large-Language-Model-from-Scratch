from data_setup.instruction_dataset import train_loader, val_loader, test_loader, train_dataset
import tiktoken


tokenizer = tiktoken.get_encoding("gpt2")


# Prerequisite for validating, set shuffle=False in train_loader in instruction_dataset.py

# The very first sequence in the first batch is the sequence of longest length, thus no padding tokens are present.
# Because of this the second sequence is tested because I want to see the padding token when I decode.
IDX = 1

print("\n----------------------------------------------------------")
print("\n******* RAW TOKEN IDs + CORRESPONDING DECODED TEXT OF SECOND TRAIN LOADER SAMPLE *******\n")

for inputs, targets in train_loader:
    # Get the second example from the batch (token IDs)
    second_example_tokens = inputs[IDX].tolist()

    print(second_example_tokens, "\n")

    print("Length of second example (train loader) tokens:", len(second_example_tokens), "\n") # 15 padding tokens (50256) get added

    # Decode token IDs back into text
    first_example_text = tokenizer.decode(second_example_tokens)

    print("Decoded Text for First Example in Batch:")
    print(first_example_text)
    break  # Stop after the first batch


# {
#     "instruction": "Edit the following sentence for grammar.",
#     "input": "He go to the park every day.",
#     "output": "He goes to the park every day."
# },


print("\n----------------------------------------------------------")
print("\n******* RAW TOKEN IDs + CORRESPONDING DECODED TEXT OF SECOND TRAIN DATASET SAMPLE *******\n")

raw_data = train_dataset[IDX]

print("Length of second example (train dataset) tokens:", len(raw_data), "\n")

print(raw_data, "\n")

print(tokenizer.decode(raw_data))

print("\n----------------------------------------------------------\n")


print("******* CHECKING FIRST BATCH FOR LONGEST SEQUENCE *******\n")
max_length = 0
max_length_index = None

# Get the first batch
for inputs, targets in train_loader:
    for i, seq in enumerate(inputs):
        seq_length = (seq != 50256).sum().item()  # Count non-padding tokens
        if seq_length > max_length:
            max_length = seq_length
            max_length_index = i

    break  # Stop after processing the first batch



# Print results
print(f"Index of longest sequence in the first batch: {max_length_index}")
print(f"Length of the longest sequence (excluding padding): {max_length}")


print("\n----------------------------------------------------------\n")
