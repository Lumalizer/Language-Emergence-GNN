from torch_geometric.data import Dataset
from options import ExperimentOptions
from data.graph_builder import GraphBuilder


class ShapesPosGraphDataset(Dataset):
    def __init__(self, labels: list[str], options: ExperimentOptions):
        self.labels = labels
        self.options = options
        builder = GraphBuilder(embedding_size=options.embedding_size)
        self.data = builder.get_batched_data(datastrings=labels).to(options.device)
        super().__init__(self.data)

    def len(self):
        return len(self.data)

    def get(self, id):
        return self.data[id]
