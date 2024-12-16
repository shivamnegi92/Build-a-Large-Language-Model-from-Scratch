import json
import os


def load_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


# Alpaca-style prompt formatting
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


data = load_file("instruction-data.json")

assert len(data) == 1100, "Instruction dataset is not of the correct length, please reload the data."

# print("Number of entries:", len(data))
# print("Example entry:\n", data[50])
# print("Example entry:\n", data[999])

# Formatting input
model_input = format_input(data[50])

desired_response = f"\n\n### Response:\n{data[50]['output']}"

print(model_input + desired_response)

# Divide the dataset into a training, validation, and test set
train_portion = int(len(data) * 0.85)  # 85% for training
test_portion = int(len(data) * 0.1)    # 10% for testing
val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation

print(train_portion)
print(test_portion)
print(val_portion)

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

print("\nTraining set length:", len(train_data))
print("Validation set length:", len(val_data))
print("Test set length:", len(test_data))