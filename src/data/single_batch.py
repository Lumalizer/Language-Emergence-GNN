import torch
import numpy as np
from options import ExperimentOptions
from joblib import Parallel, delayed


class SingleBatch:
    def __init__(self, dataloader: 'ExtendedDataLoader'):
        self.batch_idx = 0
        self.dataloader = dataloader
        self.options = dataloader.options
        self.target_data = dataloader.dataset.data
        self.labels = dataloader.dataset.labels
        self._batches = Parallel(n_jobs=4)(delayed(self.get_batch)() for _ in range(self.options.batch_size))

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_idx >= self.options.batch_size:
            raise StopIteration()

        data_vectors_sender, y, data_vectors_receiver, data_indexes_receiver = self._batches.pop()

        if self.dataloader.collect_labels:
            for i in range(self.options.batch_size):
                target_label = self.labels[data_indexes_receiver[i, y[i]]]
                distractor_labels = [self.labels[l] for l in np.delete(data_indexes_receiver[i], y[i])]
                self.dataloader.collect_labels(target_label, distractor_labels)

        self.batch_idx += 1
        return data_vectors_sender, y, data_vectors_receiver

    def get_randomized_data(self):
        data_indexes_sender = torch.randint(len(self.target_data), (self.options.batch_size, self.options.game_size))
        return data_indexes_sender

    def get_mixed_game_ids(self):
        needed_combinations = self.options.game_size // 12 + 2

        for i in range(needed_combinations, needed_combinations + 10):
            combinations = np.random.choice(self.dataloader.shape_combinations, i, replace=False)
            combined_array = np.concatenate([self.dataloader.shape_names_dict.get(c) for c in combinations])

            if len(combined_array) >= self.options.game_size:
                combined_array = np.random.permutation(combined_array)
                return combined_array[:self.options.game_size]

        raise ValueError("Could not find enough shape combinations")

    def get_randomized_data_with_shapes(self):
        choices = np.array([self.get_mixed_game_ids() for _ in range(self.options.batch_size)])
        data_indexes_sender = torch.from_numpy(choices)
        return data_indexes_sender

    def get_batch(self):
        data_indexes_sender = self.get_randomized_data_with_shapes() if self.options.use_mixed_distractors else self.get_randomized_data()

        permutes = torch.stack([torch.randperm(self.options.game_size) for _ in range(self.options.batch_size)])
        data_indexes_receiver = data_indexes_sender[torch.arange(self.options.batch_size).unsqueeze(1), permutes]
        y = permutes.argmin(dim=1)

        data_vectors_sender = self.target_data.index_select(0, data_indexes_sender.flatten()).view(self.options.batch_size, self.options.game_size, -1)
        data_vectors_receiver = self.target_data.index_select(0, data_indexes_receiver.flatten()).view(self.options.batch_size, self.options.game_size, -1)

        return data_vectors_sender, y, data_vectors_receiver, data_indexes_receiver


if __name__ == '__main__':
    from data.get_loaders import ExtendedDataLoader
