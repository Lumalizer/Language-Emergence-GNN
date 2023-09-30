from data.graph.graph_dataset import ShapesPosGraphDataset
from data.image.image_dataset import ShapesPosImgDataset
from data.single_batch import SingleBatch
from data.split_labels import split_data_labels
from options import ExperimentOptions
from torch.utils.data import DataLoader
from analysis.timer import timer
from typing import Union
import torch


@timer
def get_dataloaders(options: ExperimentOptions):
    train_labels, valid_labels = split_data_labels(options)

    if options.experiment == 'image':
        dataset_class = ShapesPosImgDataset
        data_attr = 'images'
    elif options.experiment == 'graph':
        dataset_class = ShapesPosGraphDataset
        data_attr = 'graphs'
    else:
        raise ValueError(f'Unknown experiment type: {options.experiment}. Possible values: image, graph')

    train_data = dataset_class(train_labels, options)
    valid_data = dataset_class(valid_labels, options)
    train_loader = ExtendedDataLoader(train_data, getattr(train_data, data_attr), options)
    valid_loader = ExtendedDataLoader(valid_data, getattr(valid_data, data_attr), options)

    return train_loader, valid_loader


class ExtendedDataLoader(DataLoader):
    def __init__(self, dataset: Union[ShapesPosImgDataset, ShapesPosGraphDataset], data_target: torch.Tensor, options: ExperimentOptions, collect_labels=None):
        super().__init__(dataset, batch_size=options.batch_size, shuffle=False, drop_last=True)
        self.options = options
        self.collect_labels = collect_labels
        self.data_target = data_target

    def __iter__(self):
        return SingleBatch(self, self.data_target, self.options, self.collect_labels)
