import torch
import numpy as np
import torch.nn as nn


class MessageEmbeddings(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(MessageEmbeddings, self).__init__()
        self.emb = nn.Linear(input_size, embedding_size)

    @torch.no_grad()
    def forward(self, data) -> np.ndarray:
        # d is embedding_size
        d: torch.Tensor = self.emb(data)
        return d.detach().numpy()
