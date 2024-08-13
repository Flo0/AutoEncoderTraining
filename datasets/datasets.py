import os
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode


class AutoEncoderImageDataset(Dataset):
    def __init__(self, image_dir: str, load_into_memory: bool, train=True, cut=0.1, transform=None):
        self.image_dir = image_dir
        self.images = []
        self.load_into_memory = load_into_memory
        self.transform = transform
        for root, dirs, files in os.walk(self.image_dir):
            for file in files:
                if load_into_memory:
                    image = read_image(os.path.join(root, file), mode=ImageReadMode.RGB)
                    if transform is not None:
                        image = transform(image)
                    self.images.append(image)
                else:
                    self.images.append(file)
        # On training data, cut the last part from the list
        if not train:
            self.images = self.images[:int(len(self.images) * cut)]
        else:
            self.images = self.images[int(len(self.images) * cut):]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if not self.load_into_memory:
            image = read_image(image)
            if self.transform is not None:
                image = self.transform(image)
        return image
