import torch
import numpy as np
import torch.nn as nn


class ImageEmbeddings(nn.Module):
    def __init__(self, embedding_size):
        super(ImageEmbeddings, self).__init__()
        self.emb = nn.Linear(160000, embedding_size, bias=False)

    @torch.no_grad()
    def forward(self, data) -> np.ndarray:
        # d is embedding_size
        d: torch.Tensor = self.emb(data)
        # d is 1d
        d = d.flatten()
        # d is 1 x embedding_size
        d = d.unsqueeze(0)
        return d.detach().numpy()
