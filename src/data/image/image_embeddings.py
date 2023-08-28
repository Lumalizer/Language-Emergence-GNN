import torch
import numpy as np
import torch.nn as nn
from torchvision import models

class ImageEmbeddings(nn.Module):
    def __init__(self, embedding_size):
        super(ImageEmbeddings, self).__init__()
        self.resnet18 = models.resnet18(weights="ResNet18_Weights.DEFAULT")
        self.resnet18.fc = nn.Linear(512, embedding_size, bias=False)

    @torch.no_grad()
    def forward(self, data) -> np.ndarray:
        # d is embedding_size
        d: torch.Tensor = self.resnet18(data)
        return d.detach().numpy()