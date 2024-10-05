import torch
from torch.utils.data import Dataset, DataLoader


X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])

y_train = torch.tensor([0, 0, 0, 1, 1]) # class labels

X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6]
])

y_test = torch.tensor([0, 1])

class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y
    
    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y
    
    def __len__(self):
        return self.labels.shape[0]

train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

print(len(train_ds))

torch.manual_seed(123)

# train_loader = DataLoader(
#     dataset=train_ds,
#     batch_size=2,
#     shuffle=True,
#     num_workers=0
# )

train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    drop_last=True # will drop 5th sample, since its not even
)

test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx + 1}: ", x, y)
    print()