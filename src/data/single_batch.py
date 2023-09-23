import torch
import numpy as np
from options import ExperimentOptions

class SingleBatch:
    def __init__(self, dataloader, target_data, options: ExperimentOptions, collect_labels=None):
        torch.manual_seed(42)
        self.options = options
        self.batch_idx = 0
        self.data = dataloader.dataset
        self.target_data = target_data
        self.labels = dataloader.dataset.labels

        # self.position_groups = self.get_position_groups()
        self.collect_labels = collect_labels

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_idx >= self.options.batch_size:
            raise StopIteration()
        self.batch_idx += 1
        return self.get_batch()

    def get_batch(self):
        # get game_size amount of data ids for each part of the batch
        data_indexes_sender = torch.randint(len(self.data), (self.options.batch_size, self.options.game_size))
        data_vectors_sender = torch.stack([torch.stack([torch.tensor(self.target_data[data_indexes_sender[i, j]]) for j in range(self.options.game_size)]) for i in range(self.options.batch_size)], dim=0).squeeze()

        y = torch.zeros(self.options.batch_size).long()

        # select target
        data_indexes_receiver = np.zeros_like(data_indexes_sender)
        data_vectors_receiver = torch.zeros_like(data_vectors_sender)

        for j in range(self.options.batch_size):
            permute = torch.randperm(self.options.game_size)
            data_vectors_receiver[j, :, :] = data_vectors_sender[j, permute, :]
            data_indexes_receiver[j] = data_indexes_sender[j, permute]
            y[j] = permute.argmin()

            data_vectors_sender[j] = data_vectors_receiver[j, y[j]]

            if self.collect_labels:
                target_label = self.labels[data_indexes_receiver[j, y[j]]]
                distractor_labels = [self.labels[l] for l in np.delete(data_indexes_receiver[j], y[j])]
                self.collect_labels(target_label, distractor_labels)

        # vectors are batch_size x game_size x embedding_size
        return data_vectors_sender, y, data_vectors_receiver
