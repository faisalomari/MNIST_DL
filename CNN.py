# Import pytorch and torchvision modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# Define the hyperparameters
num_classes = 10 # Number of classes to classify
batch_size = 64 # Batch size for training and testing
learning_rate = 0.01 # Learning rate for the optimizer
num_epochs = 10 # Number of epochs to train the model

# Load and normalize the train and test data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

# Define the convolutional neural network (CNN)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # First convolutional layer: 1 input channel, 16 output channels, 5x5 kernel size
        self.conv1 = nn.Conv2d(1, 16, 5)
        # Second convolutional layer: 16 input channels, 32 output channels, 5x5 kernel size
        self.conv2 = nn.Conv2d(16, 32, 5)
        # First fully connected layer: 512 input features, 256 output features
        self.fc1 = nn.Linear(512, 256)
        # Second fully connected layer: 256 input features, num_classes output features
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Apply the first convolutional layer, followed by ReLU activation and max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # Apply the second convolutional layer, followed by ReLU activation and max pooling
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # Flatten the output of the second convolutional layer
        x = x.view(-1, self.num_flat_features(x))
        # Apply the first fully connected layer, followed by ReLU activation
        x = F.relu(self.fc1(x))
        # Apply the second fully connected layer, followed by softmax activation
        x = F.softmax(self.fc2(x), dim=1)
        return x

    def num_flat_features(self, x):
        # Compute the number of features for the fully connected layer
        size = x.size()[1:] # All dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# Create an instance of the CNN model
model = CNN()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss() # Cross entropy loss for multi-class classification
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # Stochastic gradient descent optimizer

# Train the model on the training data
for epoch in range(num_epochs): # Loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # Get the inputs and labels
        inputs, labels = data
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        # Compute the loss
        loss = criterion(outputs, labels)
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        # Print statistics
        running_loss += loss.item()
        if i % 200 == 199: # Print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

print('Finished Training')

# Test the model on the test data
correct = 0
total = 0
with torch.no_grad(): # No need to track gradients for testing
    for data in testloader:
        # Get the inputs and labels
        images, labels = data
        # Forward pass
        outputs = model(images)
        # Get the predicted class
        _, predicted = torch.max(outputs.data, 1)
        # Update the statistics
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
