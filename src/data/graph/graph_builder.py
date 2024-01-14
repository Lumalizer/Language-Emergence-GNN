import os
import torch
from data.datastring_builder import DatastringBuilder
from data.graph.graph_embeddings import GraphEmbeddings
from torch_geometric.utils import to_networkx
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
        self.pos_codes = {'above': 0, 'below': 1, 'left': 2, 'right': 3}
        self.shapes_codes = {s: i+1 for i, s in enumerate(sorted(self.shapes))}
        self.shapes_codes['origin'] = 0
        self.shapes_codes['0'] = -1
        self.get_embeddings = GraphEmbeddings(self.embedding_size).forward

    def build_graph_from_string(self, graphstring: str):
        # Each node corresponds to a shape type (origin = 0)
        # Each edge corresponds to a positional relationship between shapes and the origin

        graphstring = graphstring.replace('.png', '')
        nodes = [self.shapes_codes['origin']]
        edges = []

        graphelements = graphstring.split('_')

        for i, shape in enumerate(graphelements):
            node_index = i + 1
            nodes.append(self.shapes_codes[shape])
            # above or below origin
            edges.append((node_index, 0, self.pos_codes['above'] if i < 2 else self.pos_codes['below']))
            # left or right of origin
            edges.append((node_index, 0, self.pos_codes['right'] if i % 2 else self.pos_codes['left']))

        x = torch.tensor(nodes, dtype=torch.float32).reshape([-1, 1])
        edge_index = torch.tensor([tuple(edge[:2]) for edge in edges], dtype=torch.int64).t().contiguous()
        edge_attr = torch.tensor([edge[2] for edge in edges])

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data

    def visualize_graph(self, data: Data):
        # visualize graph
        # make sure to show nodes, node attributes, edges, edge attributes
        import networkx as nx
        import matplotlib.pyplot as plt
        G = to_networkx(data)
        print(G.nodes, G.edges)
        print(data.x)
        print(data.edge_attr)
        print(data.edge_index)
        nx.draw(G, with_labels=True, font_weight='bold')
        plt.show()
        
    def get_batched_data(self, datastrings=None):
        if datastrings is None:
            datastrings = self.datastrings
        return Batch.from_data_list([self.build_graph_from_string(graphstring) for graphstring in datastrings])

    def produce_dataset(self):
        if not os.path.isdir('../assets/embedded_data'):
            os.mkdir('../assets/embedded_data')

        # if os.path.isfile(f'../assets/embedded_data/graph_embeddings{self.embedding_size}.npy'):
        #     return

        data = self.get_embeddings(self.get_batched_data(), detach=True)
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
