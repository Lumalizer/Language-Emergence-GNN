import os
import torch
from data.graph.graph_embeddings import NodeEmbeddings
from options import ExperimentOptions
from torch_geometric.data import Data


class GraphBuilder:
    def __init__(self, options: ExperimentOptions):
        # generate integer encodings for shapes and position types
        self.pos_codes = {'above': 0, 'below': 1, 'left': 2, 'right': 3}
        self.shapes = set("_".join([n.replace('.png', '') for n in os.listdir('assets/shapes')]).split("_"))
        self.shapes_codes = {s: i+1 for i, s in enumerate(sorted(self.shapes))}
        self.shapes_codes['origin'] = 0
        self.shapes_codes['0'] = -1

        self.get_embeddings = NodeEmbeddings(options).forward

    def get_embedded_graph_from_string(self, graphstring: str):
        data = self.build_graph_from_string(graphstring)
        return self.get_embeddings(data)

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
