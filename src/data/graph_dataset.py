from torch_geometric.data import Dataset
from options import ExperimentOptions
from data.graph_builder import GraphBuilder
from data.systemic_distractors import SystematicDistractors


class ShapesPosGraphDataset(Dataset):
    def __init__(self, labels: list[str], options: ExperimentOptions):
        self.labels = labels
        self.reverse_ids = {l: i for i, l in enumerate(labels)}
        self.options = options
        self.shapes = set([item for sublist in [l.split("_") for l in labels] for item in sublist])
        self.shapes.remove('0')
        self.systematic_games = SystematicDistractors(self.shapes, False)

        builder = GraphBuilder(embedding_size=options.embedding_size)
        self.data = builder.get_batched_data(datastrings=labels).to(options.device)
        super().__init__(self.data)

    def len(self):
        return len(self.data)

    def get(self, id):
        return self.data[id]
