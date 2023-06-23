# generating images from shape combinations, with graphs as strings encoded into the filenames
# 0_0_bike_bird means a 2x2 grid with a bike bottom-left and a bird bottom-right

from data.graph.graph_builder import GraphBuilder
from data.image.image_builder import ImageBuilder
from options import ExperimentOptions


def assure_dataset(options: ExperimentOptions):
    GraphBuilder(embedding_size=options.embedding_size).produce_dataset()
    ImageBuilder(embedding_size=options.embedding_size).produce_dataset()
