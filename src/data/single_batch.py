import torch
import numpy as np
from typing import Union
from options import ExperimentOptions
from data.graph.graph_dataset import ShapesPosGraphDataset
from data.image.image_dataset import ShapesPosImgDataset


class SingleBatch:
    def __init__(self, dataloader: Union[ShapesPosImgDataset, ShapesPosGraphDataset], target_data: torch.Tensor, options: ExperimentOptions, collect_labels=None):
        self.options = options
        self.batch_idx = 0
        self.target_data = target_data
        self.labels = dataloader.dataset.labels
        self.collect_labels = collect_labels

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_idx >= self.options.batch_size:
            raise StopIteration()
        self.batch_idx += 1
        return self.get_batch()

    def get_batch(self):
        data_indexes_sender = torch.randint(len(self.target_data), (self.options.batch_size, self.options.game_size))
        permutes = torch.stack([torch.randperm(self.options.game_size) for _ in range(self.options.batch_size)])
        data_indexes_receiver = data_indexes_sender[torch.arange(self.options.batch_size).unsqueeze(1), permutes]

        y = permutes.argmin(dim=1)
        data_vectors_sender = self.target_data.index_select(0, data_indexes_sender.flatten()).view(self.options.batch_size, self.options.game_size, -1)
        data_vectors_receiver = self.target_data.index_select(0, data_indexes_receiver.flatten()).view(self.options.batch_size, self.options.game_size, -1)

        if self.collect_labels:
            for i in range(self.options.batch_size):
                target_label = self.labels[data_indexes_receiver[i, y[i]]]
                distractor_labels = [self.labels[l] for l in np.delete(data_indexes_receiver[i], y[i])]
                self.collect_labels(target_label, distractor_labels)

        return data_vectors_sender, y, data_vectors_receiver
