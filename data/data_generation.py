from itertools import combinations, permutations, chain
from dataclasses import dataclass, field
from PIL import Image
import os

# generating images from shape combinations, with graphs as strings encoded into the filenames
# 0_0_bike_bird means a 2x2 grid with a bike bottom-left and a bird bottom-right


def assure_dataset():
    ShapesGridBuilder().produce_dataset()


@dataclass
class ImagePlacement:
    image: str
    position: tuple[int, int]


@dataclass
class ShapesGridBuilder:
    size: int = 2
    shapes_per_grid: int = 2
    shapes: list[str] = field(
        default_factory=lambda: [s.replace(".png", "") for s in os.listdir('assets/shapes') if s.endswith('.png')])

    def __post_init__(self):
        assert (self.size**2 >= self.shapes_per_grid)
        if not os.path.isdir('assets/output'):
            os.mkdir('assets/output')

    @property
    def shape_combinations(self):
        return tuple(combinations(self.shapes, self.shapes_per_grid))

    def get_grid_items(self, *shapes):
        zeroes = tuple(0 for _ in range(self.size**2 - len(shapes)))
        return zeroes + shapes

    def get_grid_possibilities(self):
        return set(chain.from_iterable(permutations(
            self.get_grid_items(*shapes), self.size**2) for shapes in self.shape_combinations))

    @staticmethod
    def create_pasted_image(filename: str, *images: list[ImagePlacement]):
        new_image = Image.open('assets/white.png').convert('L')
        for image in images:
            new_image.paste(Image.open(image.image), image.position)

        new_image.save(filename)
        new_image.close()

    def produce_dataset(self):
        for possibility in self.get_grid_possibilities():
            filename = '_'.join([str(p) for p in possibility])
            pastes = []

            if os.path.isfile(f"assets/output/{filename}.png"):
                continue

            for i, element in enumerate(possibility):
                if not element:
                    continue

                pastes.append(ImagePlacement(f"assets/shapes/{element}.png", ((i % 2 == 1) * 200, (i > 1) * 200)))
            self.create_pasted_image(f"assets/output/{filename}.png", *pastes)
