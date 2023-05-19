import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from data.image.image_embeddings import ImageEmbeddings
from options import ExperimentOptions

class ShapesPosImgDataset(Dataset):
    def __init__(self, labels: list[str], options: ExperimentOptions):
        get_embeddings = ImageEmbeddings(options).forward
        process_image = lambda image: get_embeddings(torch.flatten(read_image(
            f'{options.dataset_location}/{image}.png', mode=ImageReadMode.GRAY)/255, start_dim=1).squeeze())
        
        self.images = np.stack([process_image(label) for label in labels])
        self.labels = labels
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, id):
        return self.images[id]
