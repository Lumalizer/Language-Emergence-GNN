from torchvision.io import read_image, ImageReadMode
import torch
from data.datastring_builder import DatastringBuilder
from dataclasses import dataclass

@dataclass
class ImageLoader(DatastringBuilder):
    image_path: str = f"../assets/output/"

    def load(self, filename, suffix=".png"):
        return read_image(self.image_path+filename+suffix, mode=ImageReadMode.GRAY)/255
    
    def get_batched_data(self, datastrings=None):
        if datastrings is None:
            datastrings = self.datastrings
        return torch.stack([self.load(filename) for filename in datastrings])
