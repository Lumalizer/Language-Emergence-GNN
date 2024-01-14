import numpy as np
from torch.utils.data import Dataset
from options import ExperimentOptions
import torch
from data.image.image_loader import ImageLoader


class ShapesPosImgDataset(Dataset):
    def __init__(self, labels: list[str], options: ExperimentOptions):
        self.labels = labels

        if options.use_prebuilt_embeddings:
            all_labels = np.load(f'../assets/embedded_data/image_embeddings{options.embedding_size}_labels.npy')
            mask = np.isin(all_labels, labels)
            self.data = torch.tensor(np.load(f'../assets/embedded_data/image_embeddings{options.embedding_size}.npy')[mask], dtype=torch.float32)
        else:
            loader = ImageLoader()
            mask = np.isin(loader.datastrings, labels)
            self.data = loader.get_batched_data()[mask]
        
        super().__init__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        return self.data[id]
