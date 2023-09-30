import numpy as np
from torch.utils.data import Dataset
from options import ExperimentOptions
import torch


class ShapesPosImgDataset(Dataset):
    def __init__(self, labels: list[str], options: ExperimentOptions):
        all_labels = np.load(f'../assets/embedded_data/image_embeddings{options.embedding_size}_labels.npy')
        mask = np.isin(all_labels, labels)
        self.images = torch.tensor(np.load(f'../assets/embedded_data/image_embeddings{options.embedding_size}.npy')[mask], dtype=torch.float32)
        self.labels = labels
        super().__init__()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, id):
        return self.images[id]
