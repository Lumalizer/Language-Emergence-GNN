from data.graph_dataset import ShapesPosGraphDataset
from data.image_dataset import ShapesPosImgDataset
from data.single_batch import SingleBatch
from data.split_labels import split_data_labels
from options import Options
from torch.utils.data import DataLoader
from typing import Union
from random import sample


def get_dataloaders(options: Options):
    train_labels, valid_labels = split_data_labels(options)

    if options.experiment == 'image':
        dataset_class = ShapesPosImgDataset
    elif options.experiment == 'graph':
        dataset_class = ShapesPosGraphDataset
    else:
        raise ValueError(f'Unknown experiment type: {options.experiment}. Possible values: image, graph')
    
    excl_train = []
    excl_test = []

    if options.systemic_distractors:
        valid_labels = train_labels
        excl_train = sample(train_labels, len(train_labels)//4)
        excl_test = [l for l in valid_labels if l not in excl_train]
    

    train_loader = ExtendedDataLoader(dataset_class(train_labels, options, excluded_graphstrings=excl_train), options)
    valid_loader = ExtendedDataLoader(dataset_class(valid_labels, options, excluded_graphstrings=excl_test), options)

    if options.systemic_distractors:
        sys1 = train_loader.dataset.systematic_games
        sys2 = valid_loader.dataset.systematic_games
        assert not(set(sys1.targets).intersection(set(sys2.targets)))

    return train_loader, valid_loader


class ExtendedDataLoader(DataLoader):
    def __init__(self, dataset: Union[ShapesPosImgDataset, ShapesPosGraphDataset], options: Options, collect_labels=None):
        super().__init__(dataset, batch_size=options.batch_size, shuffle=False, drop_last=True)
        self.dataset: Union[ShapesPosImgDataset, ShapesPosGraphDataset]
        self.options = options
        self.collect_labels = collect_labels

    def __iter__(self):
        return SingleBatch(self)
