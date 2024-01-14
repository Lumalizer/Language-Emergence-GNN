import numpy as np
from torch_geometric.data import Dataset
from options import ExperimentOptions
from data.graph.graph_builder import GraphBuilder
import torch


class ShapesPosGraphDataset(Dataset):
    def __init__(self, labels: list[str], options: ExperimentOptions):
        self.labels = labels

        if options.use_prebuilt_embeddings:
            all_labels = np.load(f'../assets/embedded_data/graph_embeddings{options.embedding_size}_labels.npy')
            mask = np.isin(all_labels, labels)
            self.data = torch.tensor(np.load(f'../assets/embedded_data/graph_embeddings{options.embedding_size}.npy')[mask])
        else:
            builder = GraphBuilder(embedding_size=options.embedding_size)
            mask = np.isin(builder.datastrings, labels)
            self.data = builder.get_batched_data()[mask]

        super().__init__(self.data)

    def len(self):
        return len(self.data)

    def get(self, id):
        return self.data[id]
