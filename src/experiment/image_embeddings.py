import torch
import torch.nn as nn
from torchvision import models

class ImageEmbeddings(nn.Module):
    def __init__(self, embedding_size):
        super(ImageEmbeddings, self).__init__()
        self.resnet18 = models.resnet18(weights="ResNet18_Weights.DEFAULT")
        self.resnet18.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet18.fc = nn.Linear(512, embedding_size, bias=False)

    def forward(self, data):
        # d is embedding_size
        d: torch.Tensor = self.resnet18(data)
        return d