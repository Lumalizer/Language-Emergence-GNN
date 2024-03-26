import torch
import numpy as np
from torch_geometric.nn import aggr
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv

class GraphEmbeddings(torch.nn.Module):
    def __init__(self, embedding_size):
        super(GraphEmbeddings, self).__init__()
        num_node_features = 1
        self.emb1 = GATv2Conv(num_node_features, embedding_size*2, edge_dim=4, concat=False, heads=1)
        self.emb2 = GATv2Conv(embedding_size*2, embedding_size, edge_dim=4, concat=False, heads=2)
        self.emb3 = GATv2Conv(embedding_size, embedding_size, edge_dim=4, concat=False, heads=2)

        self.bn1 = torch.nn.BatchNorm1d(embedding_size*2)
        self.bn2 = torch.nn.BatchNorm1d(embedding_size)
        self.bn3 = torch.nn.BatchNorm1d(embedding_size)

        self.rel1 = torch.nn.ReLU()
        self.rel2 = torch.nn.ReLU()
        self.rel3 = torch.nn.ReLU()
        
        self.aggregate = aggr.GRUAggregation(embedding_size, embedding_size)

    def forward(self, d: Data) -> np.ndarray:
        node_features, edge_index, edge_attr, batch = d.x, d.edge_index, d.edge_attr, d.batch
        edge_attr: torch.Tensor = edge_attr.unsqueeze(dim=1)

        d = self.emb1(node_features, edge_index, edge_attr)
        d = self.bn1(d)
        d = self.rel1(d)

        d = self.emb2(d, edge_index, edge_attr)
        d = self.bn2(d)
        d = self.rel2(d)

        d = self.emb3(d, edge_index, edge_attr)
        d = self.bn3(d)
        d = self.rel3(d)

        d = self.aggregate(d, batch)
        return d
