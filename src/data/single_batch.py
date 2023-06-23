import torch
import numpy as np
from options import ExperimentOptions
import random


class SingleBatch:
    def __init__(self, dataloader, target_data, options: ExperimentOptions, target_label_collector=None):
        self.options = options
        self.randomizer = np.random.RandomState(42)
        self.batch_idx = 0

        self.data = dataloader.dataset
        self.target_data = target_data
        self.labels = dataloader.dataset.labels

        # self.position_groups = self.get_position_groups()
        self.target_label_collector = target_label_collector

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_idx >= self.options.batch_size:
            raise StopIteration()
        self.batch_idx += 1
        return self.get_batch()

    def get_position_groups(self):
        position_groups = {}
        def empty_positions(graphstring): return "".join([str(i) for i, c in enumerate(graphstring.split('_')) if c == '0'])

        for i, label in enumerate(self.labels):
            empty = empty_positions(label)
            if empty not in position_groups:
                position_groups[empty] = []

            position_groups[empty].append(i)
        return position_groups

    def get_batch(self):
        # get game_size amount of data ids for each part of the batch
        data_indexes_sender = np.zeros((self.options.batch_size, self.options.game_size), dtype=np.int32)
        for i in range(self.options.batch_size):
            if self.options.add_shape_only_games and random.random() < 0.5:
                alignment = random.choice(list(self.position_groups.keys()))
                target_ids = self.randomizer.choice(self.position_groups[alignment], self.options.game_size, replace=False)
            else:
                target_ids = self.randomizer.choice((len(self.data)), self.options.game_size, replace=False)

            data_indexes_sender[i, :] = target_ids

        # get the data from the ids
        data_vectors_sender = []
        for i in range(self.options.batch_size):
            temp_elements = []
            for j in range(self.options.game_size):
                temp_elements.append(torch.tensor(self.target_data[data_indexes_sender[i, j]]))
            data_vectors_sender.append(torch.stack(temp_elements, dim=0))

        data_vectors_sender = torch.stack(data_vectors_sender, dim=0).contiguous().squeeze()
        y = torch.zeros(self.options.batch_size).long()

        # select target
        data_vectors_receiver = torch.zeros_like(data_vectors_sender)
        data_indexes_receiver = np.zeros_like(data_indexes_sender)
        for j in range(self.options.batch_size):
            permute = torch.randperm(self.options.game_size)
            data_vectors_receiver[j, :, :] = data_vectors_sender[j, permute, :]
            data_indexes_receiver[j] = data_indexes_sender[j, permute]
            y[j] = permute.argmin()

            target_index = data_indexes_receiver[j, y[j]]
            self.target_label_collector is not None and self.target_label_collector.append(self.labels[target_index])

        # vectors are batch_size x game_size x embedding_size
        return data_vectors_sender, y, data_vectors_receiver
