from data.graph_dataset import ShapesPosGraphDataset
from data.image_dataset import ShapesPosImgDataset
from data.single_batch import SingleBatch
from data.split_labels import split_data_labels
from options import ExperimentOptions
from torch.utils.data import DataLoader
from analysis.timer import timer
from typing import Union
import numpy as np
from collections import defaultdict


@timer
def get_dataloaders(options: ExperimentOptions):
    train_labels, valid_labels = split_data_labels(options)

    if options.experiment == 'image':
        dataset_class = ShapesPosImgDataset
    elif options.experiment == 'graph':
        dataset_class = ShapesPosGraphDataset
    else:
        raise ValueError(f'Unknown experiment type: {options.experiment}. Possible values: image, graph')

    train_loader = ExtendedDataLoader(dataset_class(train_labels, options), options)
    valid_loader = ExtendedDataLoader(dataset_class(valid_labels, options), options)

    return train_loader, valid_loader


class ExtendedDataLoader(DataLoader):
    def __init__(self, dataset: Union[ShapesPosImgDataset, ShapesPosGraphDataset], options: ExperimentOptions, collect_labels=None):
        super().__init__(dataset, batch_size=options.batch_size, shuffle=False, drop_last=True)
        self.dataset: Union[ShapesPosImgDataset, ShapesPosGraphDataset]
        self.options = options
        self.collect_labels = collect_labels
        self.shape_names_dict, self.shape_names, self.shape_combinations = self.get_organized_shapes(dataset.labels)
        self.position_names_dict, self.position_names, self.position_combinations = self.get_organize_positions(dataset.labels)

    @staticmethod
    def get_organized_shapes(labels):
        shape_names_dict = defaultdict(list)

        for i, graphstring in enumerate(labels):
            shapes = sorted([g for g in graphstring.split('_') if g != '0'])

            for shape in shapes:
                shape_names_dict[shape].append(i)

            combined_shape = shapes[0]+"_" + shapes[1]
            shape_names_dict[combined_shape].append(i)

        shape_array_dict = {k: np.array(v) for k, v in shape_names_dict.items() if len(v) > 2}
        shape_names = [k for k in shape_array_dict.keys() if "_" not in k]
        shape_combinations = [k for k in shape_array_dict.keys() if "_" in k]

        return shape_array_dict, shape_names, shape_combinations
    
    @staticmethod
    def get_organize_positions(labels):
        position_names_dict = defaultdict(list)

        for i, graphstring in enumerate(labels):
            positions_empty = [j for j, g in enumerate(graphstring.split('_')) if g == '0']

            for position in positions_empty:
                position_names_dict[str(position)].append(i)

            combined_positions = "_".join([str(n) for n in sorted(positions_empty)])
            position_names_dict[combined_positions].append(i)

        position_array_dict = {k: np.array(v) for k, v in position_names_dict.items() if len(v) > 2}
        position_names = [k for k in position_array_dict.keys() if "_" not in k]
        position_combinations = [k for k in position_array_dict.keys() if "_" in k]

        return position_array_dict, position_names, position_combinations

    def __iter__(self):
        return SingleBatch(self)
