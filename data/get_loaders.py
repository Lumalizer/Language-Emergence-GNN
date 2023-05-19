from data.graph.graph_dataset import ShapesPosGraphDataset
from data.image.image_dataset import ShapesPosImgDataset
from data.single_batch import SingleBatch
from data.split_labels import split_data_labels
from options import ExperimentOptions
from torch.utils.data import DataLoader
from data.graph.graph_builder import GraphBuilder
from utils.timer import timer


def get_dataloaders(options: ExperimentOptions):
    train_labels, valid_labels = split_data_labels(options)

    if options.experiment == 'image':
        return get_image_dataloaders(options, train_labels, valid_labels)
    elif options.experiment == 'graph':
        return get_graph_dataloaders(options, train_labels, valid_labels)
    else:
        raise ValueError(f'Unknown experiment type: {options.experiment}. Possible values: image, graph')


@timer
def get_image_dataloaders(options: ExperimentOptions, train_labels, valid_labels):
    train_data = ShapesPosImgDataset(train_labels, options)
    valid_data = ShapesPosImgDataset(valid_labels, options)
    return ExtendedDataLoader(train_data, train_data.images, options), ExtendedDataLoader(valid_data, valid_data.images, options)


@timer
def get_graph_dataloaders(options: ExperimentOptions, train_labels, valid_labels):
    get_embedded_graph_from_string = GraphBuilder(options).get_embedded_graph_from_string
    train_data = ShapesPosGraphDataset(train_labels, get_embedded_graph_from_string)
    valid_data = ShapesPosGraphDataset(valid_labels, get_embedded_graph_from_string)
    return ExtendedDataLoader(train_data, train_data, options), ExtendedDataLoader(valid_data, valid_data, options)


class ExtendedDataLoader(DataLoader):
    def __init__(self, dataset: ShapesPosImgDataset | ShapesPosGraphDataset, data_target, options: ExperimentOptions):
        self.dataset: ShapesPosImgDataset | ShapesPosGraphDataset = dataset
        super().__init__(dataset, batch_size=options.batch_size, shuffle=False, drop_last=True)
        self.options = options
        # data_target makes it possible to use the same class for regular pytorch and pytorch_geomtric data
        self.data_target = data_target

    def __iter__(self):
        return SingleBatch(self, self.data_target, self.options)
