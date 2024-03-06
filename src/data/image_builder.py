import os
from PIL import Image
from dataclasses import dataclass
from data.datastring_builder import DatastringBuilder


@dataclass
class ImagePlacement:
    image: str
    position: tuple[int, int]


@dataclass
class ImageBuilder(DatastringBuilder):
    embedding_size: int = None

    @staticmethod
    def create_pasted_image(filename: str, *images: list[ImagePlacement]):
        new_image = Image.open('assets/white.png').convert('L')
        for image in images:
            new_image.paste(Image.open(image.image), image.position)

        new_image.save(filename)
        new_image.close()

    def assure_images(self):
        for possibility in self.get_grid_possibilities():
            filename = '_'.join([str(p) for p in possibility])
            pastes = []

            if os.path.isfile(f"assets/output/{filename}.png"):
                continue

            for i, shape in enumerate(possibility):
                if not shape:
                    continue

                pastes.append(ImagePlacement(f"assets/shapes/{shape}.png", ((i % 2 == 1) * 200, (i > 1) * 200)))
            self.create_pasted_image(f"assets/output/{filename}.png", *pastes)
