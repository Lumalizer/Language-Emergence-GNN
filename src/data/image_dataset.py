from dataclasses import dataclass
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import Resize
from options import ExperimentOptions
from data.datastring_builder import DatastringBuilder
import torch

resize = Resize((140, 140), antialias=True)

@dataclass
class ImageLoader(DatastringBuilder):
    image_path: str = f"assets/output/"

    def load(self, filename, suffix=".png"):
        return resize(read_image(self.image_path+filename+suffix, mode=ImageReadMode.GRAY)/255)

    def get_batched_data(self, datastrings=None):
        if datastrings is None:
            datastrings = self.datastrings
        return torch.stack([self.load(filename) for filename in datastrings])


class ShapesPosImgDataset(Dataset):
    def __init__(self, labels: list[str], options: ExperimentOptions):
        self.labels = labels
        loader = ImageLoader()
        self.data = loader.get_batched_data(datastrings=labels).to(options.device)
        super().__init__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        return self.data[id]
