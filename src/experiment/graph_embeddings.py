import torch
import numpy as np
from torch_geometric.nn import aggr
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv

class GraphEmbeddings(torch.nn.Module):
    def __init__(self, embedding_size):
        super(GraphEmbeddings, self).__init__()
        num_node_features = 1
        self.emb1 = GATv2Conv(num_node_features, embedding_size, edge_dim=1, concat=False, heads=2)
        self.bn1 = torch.nn.BatchNorm1d(embedding_size)
        self.rel1 = torch.nn.ReLU()

        self.lstmaggrr = aggr.LSTMAggregation(embedding_size, embedding_size)

    def forward(self, d: Data) -> np.ndarray:
        node_features, edge_index, edge_attr, batch = d.x, d.edge_index, d.edge_attr, d.batch
        edge_attr: torch.Tensor = edge_attr.float().unsqueeze(dim=1)

        d = self.emb1(node_features, edge_index, edge_attr)
        d = self.bn1(d)
        d = self.rel1(d)

        d = self.lstmaggrr(d, batch)
        return d
