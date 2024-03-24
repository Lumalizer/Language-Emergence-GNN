import torch
import numpy as np
import random
from torch_geometric.data import Batch

class SingleBatch:
    def __init__(self, dataloader: 'ExtendedDataLoader'):
        self.batch_idx = 0
        self.dataloader = dataloader
        self.options = dataloader.options
        self.target_data = dataloader.dataset.data
        self.labels = dataloader.dataset.labels

    def __iter__(self):
        return self

    def __next__(self):
        limit = self.options.batches_per_epoch * (9*self.options.eval + 1)
        if self.batch_idx >= limit:
            raise StopIteration()

        vectors_sender, y, vectors_receiver, indexes_sender, indexes_receiver, aux_input = self._get_batch()

        if self.batch_idx == 0:
            self.options.results['labels_sender'] = []
        for i in range(self.options.batch_size):
            self.options.results['labels_sender'].append([self.labels[j] for j in indexes_sender[i].tolist()])

        if self.dataloader.collect_labels:
            for i in range(self.options.batch_size):
                target_label = self.labels[indexes_receiver[i, y[i]]]
                distractor_labels = [self.labels[l] for l in np.delete(indexes_receiver[i].cpu(), y[i].cpu())]
                self.dataloader.collect_labels(target_label, distractor_labels)

        self.batch_idx += 1
        return vectors_sender, y, vectors_receiver, aux_input

    def get_randomized_data(self):
        data_indexes_sender = torch.randint(len(self.target_data), (self.options.batch_size, self.options.game_size))
        return data_indexes_sender.to(self.options.device)
    
    def get_systematic_distractors(self):
        systematic = self.dataloader.dataset.systematic_games
        indexes = random.sample(range(len(systematic.targets)), self.options.batch_size)
        graphstrings = [[systematic.targets[i]]+random.sample(systematic.distractors[i], k=4) for i in indexes]
        ids = [[self.dataloader.dataset.reverse_ids[ele] for ele in elements] for elements in graphstrings]
        return torch.tensor(ids).long().to(self.options.device)

    def _get_batch(self):
        permutes = torch.stack([torch.randperm(self.options.game_size) for _ in range(self.options.batch_size)])

        if self.options.systemic_distractors:
            indexes_sender = self.get_systematic_distractors()
            y = permutes[:, 0]
        else:
            indexes_sender = self.get_randomized_data()
            y = permutes.argmin(dim=1)

        indexes_receiver = indexes_sender[torch.arange(self.options.batch_size).unsqueeze(1), permutes]

        if self.options.sender_target_only:
            # give sender only the target data
            for j in range(self.options.batch_size):
                for i in range(self.options.game_size):
                    indexes_sender[j, i] = indexes_receiver[j, y[j]]
                indexes_sender[j, 0] = indexes_receiver[j, y[j]]

            indexes_sender = indexes_sender[:, :1]

        if self.options.experiment == 'graph':
            data_sender = Batch.from_data_list(self.target_data.index_select(indexes_sender))
            data_receiver = Batch.from_data_list(self.target_data.index_select(indexes_receiver))
        else:
            data_sender = self.target_data.index_select(0, indexes_sender.flatten())
            data_receiver = self.target_data.index_select(0, indexes_receiver.flatten())

        # return vectors_sender, y, vectors_receiver, indexes_receiver
        return None, y, None, indexes_sender, indexes_receiver, \
            {'data_sender': data_sender, 'y': y, 'data_receiver': data_receiver}


if __name__ == '__main__':
    from data.get_loaders import ExtendedDataLoader
