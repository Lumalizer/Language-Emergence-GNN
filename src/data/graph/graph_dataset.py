import numpy as np
from torch_geometric.data import Dataset
from options import ExperimentOptions


class ShapesPosGraphDataset(Dataset):
    def __init__(self, labels: list[str], options: ExperimentOptions):
        all_labels = np.load(f'../assets/embedded_data/graph_embeddings{options.embedding_size}_labels.npy')
        mask = np.isin(all_labels, labels)
        self.graphs = np.load(f'../assets/embedded_data/graph_embeddings{options.embedding_size}.npy')[mask].tolist()
        self.labels = labels
        super().__init__(self.graphs)

    def len(self):
        return len(self._indices) if self._indices else len(self.graphs)

    def get(self, id):
        return self.graphs[id]
