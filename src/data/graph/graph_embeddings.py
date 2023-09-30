import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv


class GraphEmbeddings(torch.nn.Module):
    def __init__(self, embedding_size):
        super(GraphEmbeddings, self).__init__()
        num_node_features = 1

        # we need to use some GNN that supports edge attributes
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/cheatsheet.html
        self.emb = GATv2Conv(num_node_features, embedding_size, edge_dim=2, heads=2, concat=True)
        self.embedding_size = embedding_size

    @torch.no_grad()
    def forward(self, data: Data) -> np.ndarray:
        node_features, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # d is node_features x embedding_size (2d)
        d = self.emb(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        d = torch.nn.Linear(d.shape[1], self.embedding_size//3)(d)
        # d is 1d
        d = d.flatten()
        # d is 1 x embedding_size
        d: torch.Tensor = d.unsqueeze(0)
        return d.detach().numpy()
