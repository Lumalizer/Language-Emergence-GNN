import torch
import numpy as np
import random
import egg.core.util as util
from torch_geometric.data import Batch

class SingleBatch:
    def __init__(self, dataloader: 'ExtendedDataLoader'):
        self.batch_idx = 0
        # util._set_seed(self.batch_idx)
        self.dataloader = dataloader
        self.options = dataloader.options
        self.target_data = dataloader.dataset.data
        self.labels = dataloader.dataset.labels

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_idx >= self.options.batches_per_epoch:
            raise StopIteration()

        data_vectors_sender, y, data_vectors_receiver, data_indexes_receiver, aux_input = self.get_batch()

        if self.dataloader.collect_labels:
            for i in range(self.options.batch_size):
                target_label = self.labels[data_indexes_receiver[i, y[i]]]
                distractor_labels = [self.labels[l] for l in np.delete(data_indexes_receiver[i].cpu(), y[i].cpu())]
                self.dataloader.collect_labels(target_label, distractor_labels)

        self.batch_idx += 1
        # util._set_seed(self.batch_idx)
        return data_vectors_sender, y, data_vectors_receiver, aux_input

    def get_randomized_data(self):
        data_indexes_sender = torch.randint(len(self.target_data), (self.options.batch_size, self.options.game_size))
        return data_indexes_sender.to(self.options.device)

    def get_mixed_game_ids(self, type: str):
        needed_combinations = self.options.game_size // 12 + 2

        for i in range(needed_combinations, needed_combinations + 10):
            combinations = np.random.choice(self.dataloader.shape_combinations if type == 'shape' else self.dataloader.position_combinations, i, replace=False)
            combined_array = np.concatenate([self.dataloader.shape_names_dict.get(c) if type == 'shape' else self.dataloader.position_names_dict.get(c) for c in combinations])

            if len(combined_array) >= self.options.game_size:
                combined_array = np.random.permutation(combined_array)
                return combined_array[:self.options.game_size]

        raise ValueError(f"Could not find enough {type} combinations")

    def get_randomized_data_with_combinations(self):
        r = random.choice([0, 1, 2])
        if r == 0:
            choices = torch.tensor(np.stack([self.get_mixed_game_ids(type="shape") for _ in range(self.options.batch_size)]))
        elif r == 1:
            choices = torch.tensor(np.stack([self.get_mixed_game_ids(type="positions") for _ in range(self.options.batch_size)]))
        else:
            return self.get_randomized_data()
        data_indexes_sender = choices.to(self.options.device)
        return data_indexes_sender.long()

    def get_batch(self):
        indexes_sender = self.get_randomized_data_with_combinations() if self.options.use_mixed_distractors else self.get_randomized_data()

        permutes = torch.stack([torch.randperm(self.options.game_size) for _ in range(self.options.batch_size)])
        indexes_receiver = indexes_sender[torch.arange(self.options.batch_size).unsqueeze(1), permutes]
        y = permutes.argmin(dim=1)

        if self.options.sender_target_only:
            # give sender only the target data
            for j in range(self.options.batch_size):
                indexes_sender[j, 0] = indexes_receiver[j, y[j]]

            indexes_sender = indexes_sender[:, :1]

        # return indexes_sender, y, indexes_receiver, indexes_receiver
        if self.options.experiment == 'graph':
            vectors_sender = Batch.from_data_list(self.target_data.index_select(indexes_sender))
            vectors_receiver = Batch.from_data_list(self.target_data.index_select(indexes_receiver))
        else:
            vectors_sender = self.target_data.index_select(0, indexes_sender.flatten())
            vectors_receiver = self.target_data.index_select(0, indexes_receiver.flatten())

        # return vectors_sender, y, vectors_receiver, indexes_receiver
        return None, y, None, indexes_receiver, {'vectors_sender': vectors_sender, 'y': y, 'vectors_receiver': vectors_receiver}


if __name__ == '__main__':
    from data.get_loaders import ExtendedDataLoader
