import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from tqdm import tqdm

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for data, labels in tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}', unit='batch'):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        average_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch}/{num_epochs}, Average Loss: {average_loss:.4f}')

    return model

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

def view_data_sample(loader):
    image, label = next(iter(loader))
    plt.figure(figsize=(16, 8))
    plt.axis('off')
    plt.imshow(make_grid(image, nrow=16).permute((1, 2, 0)))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def splice_batch(X, Y, num_of_labels, prints=True):
    if prints:
        print('input: ', end="")
        print("\t X shape: ", X.shape, end='\t')
        print("\t Y shape: ", Y.shape)
    X = X[Y < num_of_labels]
    Y = Y[Y < num_of_labels]
    if prints:
        print('output: ', end="")
        print("\t X shape: ", X.shape, end='\t')
        print("\t Y shape: ", Y.shape)
    return X, Y

class Net_1(nn.Module):
    def __init__(self):
        super(Net_1, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 60)
        self.fc2 = nn.Linear(60, 20)
        self.fc3 = nn.Linear(20, 3)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

# Parameters
batch_size = 128
lr = 0.001
num_epochs = 10

# Download and load the training data
trainset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=True, transform=ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=False, transform=ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

net_1 = Net_1()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net_1.parameters(), lr=lr)

# Splice the batch for classes {0, 1, 2} for both training and test sets
train_data, train_labels = splice_batch(trainset.data, trainset.targets, num_of_labels=3)
train_data = train_data.float() / 255.0  # Normalization
train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_data, test_labels = splice_batch(testset.data, testset.targets, num_of_labels=3)
test_data = test_data.float() / 255.0  # Normalization
test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Train the model
net_1 = train_model(net_1, train_loader, criterion, optimizer, num_epochs)

# Test the model
test_model(net_1, test_loader)
print(count_parameters(net_1))