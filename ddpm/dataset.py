from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from glob import glob
import torch
import torch.nn as nn
from tqdm import tqdm


class AerialDataset(Dataset):
    def __init__(
            self,
            folder: str,
            image_size: int,
            augment_horizontal_flip: bool = False,
    ) -> None:
        self.paths = [path for path in glob(f"{folder}/*.png")]
        self.image_size = image_size

        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, ix) -> torch.Tensor:
        path = self.paths[ix]
        img = Image.open(path).convert("RGB")
        return self.transform(img)


class PokemonDataset(Dataset):
    def __init__(
            self,
            augment_horizontal_flip: bool = False,
    ) -> None:
        folder = f"../data/pokemon/gen3"
        image_size = 32
        paths = [path for path in glob(f"{folder}/*.png")]
        self.image_size = image_size

        transform = transforms.Compose(
            [
                transforms.Pad((image_size * 2, image_size * 2), padding_mode='constant', fill=(255, 255, 255)),
                transforms.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
                transforms.CenterCrop(image_size * 2),
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ]
        )

        self.images = []

        for path in tqdm(paths, "Loading images"):
            # Load image with RGBA channels
            img = Image.open(path).convert("RGBA")

            # Create a white background image
            white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))

            # Paste the image onto the white background
            img = Image.alpha_composite(white_bg, img)

            # Convert the result to RGB
            img = img.convert("RGB")

            # Apply transformations
            img = transform(img)

            self.images.append(img)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, ix) -> torch.Tensor:
        return self.images[ix]


class FFHQDataset(Dataset):
    def __init__(
            self,
            folder: str = "../data/ffhq",
            scale: float = 1.0,
            image_count_limit: int = 70000
    ) -> None:
        image_size = int(64 * scale)
        paths = [path for path in glob(f"{folder}/**/*.png", recursive=True)[:image_count_limit]]
        self.image_size = image_size

        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ]
        )

        self.images = []

        for path in tqdm(paths, "Loading images"):
            # Load image with RGB channels
            img = Image.open(path).convert("RGB")

            # Apply transformations
            img = transform(img)

            self.images.append(img)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, ix) -> torch.Tensor:
        return self.images[ix]
