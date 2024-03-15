from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image
from options import Options
import torch
from data.image_builder import ImageBuilder
from torchvision.transforms import Resize
from data.systemic_distractors import SystematicDistractors

resize = Resize((140, 140), antialias=True)
ImageBuilder().assure_images()


class ShapesPosImgDataset(Dataset):
    def __init__(self, labels: list[str], options: Options, excluded_graphstrings=[]):
        self.image_path = f"assets/output/"
        self.labels = labels
        self.reverse_ids = {l: i for i, l in enumerate(labels)}
        self.data = torch.stack([self.load(filename) for filename in labels]).to(options.device)
        self.options = options
        
        self.shapes = set([item for sublist in [l.split("_") for l in labels] for item in sublist])
        self.shapes.remove('0')
        self.systematic_games = SystematicDistractors(self.shapes, False, excluded_graphstrings=excluded_graphstrings)

        super().__init__()

    def load(self, filename, suffix=".png"):
        return resize(read_image(self.image_path+filename+suffix, mode=ImageReadMode.UNCHANGED).float() / 255.0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, id):
        return self.data[id]
