import os
import torch
from data.datastring_builder import DatastringBuilder
from data.graph.graph_embeddings import GraphEmbeddings
from torch_geometric.data import Data, Batch
from dataclasses import dataclass
import numpy as np


@dataclass
class GraphBuilder(DatastringBuilder):
    embedding_size: int = None

    def __post_init__(self):
        super().__post_init__()
        assert self.embedding_size is not None
        # generate integer encodings for shapes and position types
        self.shapes_codes: dict[str, int] = {s: (i+1) for i, s in enumerate(sorted(self.shapes))}
        self.shapes_codes['origin'] = 0
        self.shapes_codes['0'] = -1 # empty position
        self.get_embeddings = GraphEmbeddings(self.embedding_size).forward

    def build_graph_from_string(self, graphstring: str):
        # Each node corresponds to a shape type (origin = node 0)
        # Each edge corresponds to a positional relationship between shapes and the origin

        graphstring = graphstring.replace('.png', '')
        nodes = [0] + [self.shapes_codes[shape] for i, shape in enumerate(graphstring.split('_')) if shape != '0']
        edge_attr = [(i in [0, 1], i in [0, 2]) for i, shape in enumerate(graphstring.split('_')) if shape != '0']

        x = torch.tensor(nodes, dtype=torch.float32).reshape([-1, 1])
        edge_index = torch.tensor([[1, 2], [0, 0]], dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data

    def get_batched_graphs(self):
        return Batch.from_data_list([self.build_graph_from_string(graphstring) for graphstring in self.datastrings])

    def produce_dataset(self):
        if not os.path.isdir('../assets/embedded_data'):
            os.mkdir('../assets/embedded_data')

        if not os.path.isfile(f'../assets/embedded_data/graph_embeddings{self.embedding_size}.npy'):
            data = self.get_embeddings(self.get_batched_graphs())
            data = data.reshape([-1, self.embedding_size])

            np.save(f'../assets/embedded_data/graph_embeddings{self.embedding_size}.npy', data)
            np.save(f'../assets/embedded_data/graph_embeddings{self.embedding_size}_labels.npy', np.array(self.datastrings))

    def visualize_embeddings(self, data):
        from sklearn.manifold import TSNE
        import plotly.express as px
        tsne = TSNE(n_components=2, random_state=0, n_jobs=-1)
        data = tsne.fit_transform(data)
        fig = px.scatter(x=data[:, 0], y=data[:, 1], color=self.datastrings) 
        fig.show()
