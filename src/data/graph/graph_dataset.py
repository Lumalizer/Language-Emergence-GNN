import numpy as np
from torch_geometric.data import Dataset
from options import ExperimentOptions
import torch


class ShapesPosGraphDataset(Dataset):
    def __init__(self, labels: list[str], options: ExperimentOptions):
        all_labels = np.load(f'../assets/embedded_data/graph_embeddings{options.embedding_size}_labels.npy')
        mask = np.isin(all_labels, labels)
        self.data = torch.tensor(np.load(f'../assets/embedded_data/graph_embeddings{options.embedding_size}.npy')[mask])
        self.labels = labels
        super().__init__(self.data)

    def len(self):
        return len(self._indices) if self._indices else len(self.data)

    def get(self, id):
        return self.data[id]
