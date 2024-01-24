# Faisal Omari - 325616894 - faisalomari321@gmail.com
# Saji Assi - 314831207 - sajiassi86@gmail.com
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import numpy as np
from collections import namedtuple
from torchvision.utils import make_grid


# Sample image in dataset
def view_data_sample(loader):
  image, label = next(iter(loader))
  plt.figure(figsize=(16,8))
  plt.axis('off')
  plt.imshow(make_grid(image, nrow=16).permute((1, 2, 0)))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def splice_batch(X, Y, num_of_labels, prints=False):
  if prints:
    print('input: ',end="")
    print("\t X shape: ", X.shape, end='\t')
    print("\t Y shape: ", Y.shape)
  X = X[Y<num_of_labels]
  Y = Y[Y<num_of_labels]
  if prints:
    print('output: ',end="")
    print("\t X shape: ", X.shape, end='\t')
    print("\t Y shape: ", Y.shape)
  return X, Y


# Download and load the training data
batch_size = 16
trainset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=True, transform=ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=False, transform=ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)




