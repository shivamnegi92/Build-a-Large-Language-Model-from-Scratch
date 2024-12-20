import json


def load_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data

data = load_file("data_setup/instruction-data.json")

assert len(data) == 1100, "Instruction dataset is not of the correct length, please reload the data."

# Divide the dataset into a training, validation, and test set
train_portion = int(len(data) * 0.85)  # 85% for training
test_portion = int(len(data) * 0.1)    # 10% for testing
val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]
