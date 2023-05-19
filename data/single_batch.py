import torch
import numpy as np
from options import ExperimentOptions
import math


class SingleBatch:
    def __init__(self, dataloader, target_data, options: ExperimentOptions):
        self.options = options
        self.randomizer = np.random.RandomState(42)
        self.batch_idx = 0

        self.data = dataloader.dataset
        self.target_data = target_data
        self.labels = dataloader.dataset.labels

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_idx >= self.options.batch_size:
            raise StopIteration()
        self.batch_idx += 1
        return self.get_batch()

    # warning: slow for now
    def get_mixed_target_ids(self):
        onethird_plus = math.ceil(self.options.game_size / 3)

        # select random targets
        keys_available = range(len(self.data))
        randoms = self.randomizer.choice(keys_available, onethird_plus, replace=False)
        selected = self.labels[randoms[0]]

        keys_available = [k for k in keys_available if k not in randoms]

        # select targets with the same shapes
        present_shapes = [s for s in selected.replace('0', '').replace('__', '_').split('_') if s]
        possible_targets = [self.labels.index(l) for l in self.labels if all([s in l for s in present_shapes]) and l != selected]
        possible_targets = [l for l in possible_targets if l in keys_available]
        same_shapes = self.randomizer.choice(possible_targets, onethird_plus, replace=False)

        keys_available = [k for k in keys_available if k not in same_shapes]

        # select targets with the same positions
        empty_positions = lambda graphstring: [i for i, c in enumerate(graphstring.split('_')) if c == '0']

        empty_selected = empty_positions(selected)
        possible_targets = [self.labels.index(l) for l in self.labels if empty_positions(l) == empty_selected and l != selected]
        possible_targets = [l for l in possible_targets if l in keys_available]
        same_positions = self.randomizer.choice(possible_targets, onethird_plus, replace=False)

        target_ids = np.concatenate((randoms, same_shapes, same_positions), dtype=np.int32)
        target_ids = self.randomizer.choice(target_ids, self.options.game_size, replace=False)
        return target_ids

    def get_batch(self):
        # get game_size amount of data ids for each part of the batch
        data_indexes_sender = np.zeros((self.options.batch_size, self.options.game_size), dtype=np.int32)
        for i in range(self.options.batch_size):
            if self.options.mixed_distractor_selection:
                target_ids = self.get_mixed_target_ids()
            else:
                target_ids = self.randomizer.choice(range(len(self.data)), self.options.game_size, replace=False)

            data_indexes_sender[i, :] = target_ids

        # get the data from the ids
        data_vectors_sender = []
        for i in range(self.options.batch_size):
            temp_elements = []
            for j in range(self.options.game_size):
                temp_elements.append(torch.tensor(self.target_data[data_indexes_sender[i, j]]))
            data_vectors_sender.append(torch.stack(temp_elements, dim=0))

        data_vectors_sender = torch.stack(data_vectors_sender, dim=2).contiguous().squeeze()
        y = torch.zeros(self.options.batch_size).long()

        # select target
        data_vectors_receiver = torch.zeros_like(data_vectors_sender)
        for j in range(self.options.batch_size):
            permute = torch.randperm(self.options.game_size)
            data_vectors_receiver[:, j, :] = data_vectors_sender[permute, j, :]
            y[j] = permute.argmin()

        return data_vectors_sender, y, data_vectors_receiver
