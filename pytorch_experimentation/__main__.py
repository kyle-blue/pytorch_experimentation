import torch
from torch import nn
from torchvision import datasets, transforms
import os


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


if __name__ == "__main__":
    train_dataset = datasets.CIFAR10(
        root="data", download=True, transform=transforms.ToTensor(), train=True
    )
    test_dataset = datasets.CIFAR10(
        root="data", download=True, transform=transforms.ToTensor(), train=False
    )
    print(train_dataset.data.shape)
    print(train_dataset.targets.shape)
    print(test_dataset.data.shape)
