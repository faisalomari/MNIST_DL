import torch.nn.functional as F
import torch.nn as nn
import torch

class model_1(nn.Module):
    def __init__(self):
        super(model_1, self).__init__()
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
    
class model_2(nn.Module):
    def __init__(self):
        super(model_2, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 60)
        self.fc2 = nn.Linear(60, 20)
        self.fc3 = nn.Linear(20, 7)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
    
class model_3(nn.Module):
    def __init__(self):
        super(model_3, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 60)
        self.pool = nn.MaxPool1d(kernel_size=3)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(60, 16)
        self.dropout2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(16, 32)
        self.dropout3 = nn.Dropout(0.25)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 7)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.pool(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        return x

class model_4(nn.Module):
    def __init__(self):
        super(model_4, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.dropout2 = nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(16, 8, 3)
        self.dropout3 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(15488, 7)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = F.relu(x)
        # x = self.pool(x)
        # x = self.dropout1(x)
        x = self.conv2(x)
        x = F.relu(x)
        # x = self.pool(x)
        # x = self.dropout2(x)
        x = self.conv3(x)
        x = F.relu(x)
        # x = self.pool(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = F.softmax(x, dim=1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
class model_5(nn.Module):
    def __init__(self):
        super(model_5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=3)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(4096, 250)  # Assuming input image size is 28x28
        self.fc2 = nn.Linear(250, 10)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.sigmoid(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

#epoch 10 batch = 256, lr = 0.001
#     Test Accuracy: 87.53%
# 1045576
# class model_5(nn.Module):
#     def __init__(self):
#         super(model_5, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
#         self.pool = nn.MaxPool2d(kernel_size=3)
#         self.dropout = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(4096, 250)  # Assuming input image size is 28x28
#         self.fc2 = nn.Linear(250, 10)

#     def forward(self, x):
#         x = x.unsqueeze(1)
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)
#         x = self.dropout(x)
#         x = x.view(-1, self.num_flat_features(x))
#         x = F.sigmoid(self.fc1(x))
#         x = F.softmax(self.fc2(x), dim=1)
#         return x

#     def num_flat_features(self, x):
#         size = x.size()[1:]
#         num_features = 1
#         for s in size:
#             num_features *= s
#         return num_features