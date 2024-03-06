from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image
from options import ExperimentOptions
import torch
from data.image_builder import ImageBuilder
from torchvision.transforms import Resize

resize = Resize((140, 140), antialias=True)
ImageBuilder().assure_images()


class ShapesPosImgDataset(Dataset):
    def __init__(self, labels: list[str], options: ExperimentOptions):
        self.image_path = f"assets/output/"
        self.labels = labels
        self.data = torch.stack([self.load(filename) for filename in labels]).to(options.device)
        super().__init__()

    def load(self, filename, suffix=".png"):
        return resize(read_image(self.image_path+filename+suffix, mode=ImageReadMode.UNCHANGED).float() / 255.0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, id):
        return self.data[id]
