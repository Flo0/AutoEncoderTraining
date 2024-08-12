import os
from torch.utils.data import Dataset
from torchvision.io import read_image


class AutoEncoderImageDataset(Dataset):
    def __init__(self, image_dir: str, load_into_memory: bool):
        self.image_dir = image_dir
        self.images = []
        self.load_into_memory = load_into_memory
        for root, dirs, files in os.walk(self.image_dir):
            for file in files:
                if load_into_memory:
                    self.images.append(read_image(os.path.join(root, file)))
                else:
                    self.images.append(file)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.load_into_memory:
            return image
        return read_image(image)
