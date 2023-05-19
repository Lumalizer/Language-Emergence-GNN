from torch_geometric.data import Dataset


class ShapesPosGraphDataset(Dataset):
    def __init__(self, labels: list[str], get_embedded_graph_from_string: callable):
        self.graphs = [get_embedded_graph_from_string(label) for label in labels]
        self.labels = labels
        super().__init__(self.graphs)

    def len(self):
        return len(self._indices) if self._indices else len(self.graphs)

    def get(self, id):
        return self.graphs[id]
