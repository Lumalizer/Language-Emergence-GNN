from data.graph.graph_dataset import ShapesPosGraphDataset
from data.image.image_dataset import ShapesPosImgDataset
from data.single_batch import SingleBatch
from data.split_labels import split_data_labels
from options import ExperimentOptions
from torch.utils.data import DataLoader
from analysis.timer import timer
from typing import Union


def get_dataloaders(options: ExperimentOptions, target_label_collector=None):
    train_labels, valid_labels = split_data_labels(options)

    if options.experiment == 'image':
        return get_image_dataloaders(options, train_labels, valid_labels, target_label_collector)
    elif options.experiment == 'graph':
        return get_graph_dataloaders(options, train_labels, valid_labels, target_label_collector)
    else:
        raise ValueError(f'Unknown experiment type: {options.experiment}. Possible values: image, graph')


@timer
def get_image_dataloaders(options: ExperimentOptions, train_labels, valid_labels, target_label_collector):
    train_data = ShapesPosImgDataset(train_labels, options)
    valid_data = ShapesPosImgDataset(valid_labels, options)
    return ExtendedDataLoader(train_data, train_data.images, options, target_label_collector), ExtendedDataLoader(valid_data, valid_data.images, options, target_label_collector)


@timer
def get_graph_dataloaders(options: ExperimentOptions, train_labels, valid_labels, target_label_collector):
    train_data = ShapesPosGraphDataset(train_labels, options)
    valid_data = ShapesPosGraphDataset(valid_labels, options)
    return ExtendedDataLoader(train_data, train_data, options, target_label_collector), ExtendedDataLoader(valid_data, valid_data, options, target_label_collector)


class ExtendedDataLoader(DataLoader):
    def __init__(self, dataset: Union[ShapesPosImgDataset, ShapesPosGraphDataset], data_target, options: ExperimentOptions, target_label_collector):
        self.dataset = dataset
        super().__init__(dataset, batch_size=options.batch_size, shuffle=False, drop_last=True)
        self.options = options
        self.target_label_collector = target_label_collector
        # data_target makes it possible to use the same class for regular pytorch and pytorch_geomtric data
        self.data_target = data_target

    def __iter__(self):
        return SingleBatch(self, self.data_target, self.options, self.target_label_collector)
