import os
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import re
import numpy as np
from tqdm import tqdm


def numerical_sort(value):
    numbers = re.findall(r'\d+', value)
    return int(numbers[0]) if numbers else float('inf')


class AutoEncoderNPZDataset(Dataset):
    def __init__(self, npz_dir: str, transform=None):
        self.npz_dir = npz_dir
        self.npz_files = []
        self.transform = transform

        for root, dirs, files in os.walk(self.npz_dir):
            for file in files:
                if file.endswith('.npz'):
                    self.npz_files.append(os.path.join(root, file))

        if len(self.npz_files) == 0:
            raise ValueError("No NPZ files found in the directory")

        batches = []
        for file in tqdm(self.npz_files, "Loading NPZ files"):
            loaded_np = np.load(file, allow_pickle=True)
            flattened_images = loaded_np['data']
            images = flattened_images.reshape(-1, 3, 32, 32)
            images = images.transpose(0, -2, -1, 1)
            batches.append(images)

        self.data = np.concatenate(batches, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data


class DenoiserNPZDataset(Dataset):
    def __init__(self, npz_dir: str, noise_transform, transform=None, load_noise=False):
        self.npz_dir = npz_dir
        self.npz_files = []
        self.transform = transform
        self.noise_transform = noise_transform
        self.load_noise = load_noise

        for root, dirs, files in os.walk(self.npz_dir):
            for file in files:
                if file.endswith('.npz'):
                    self.npz_files.append(os.path.join(root, file))

        if len(self.npz_files) == 0:
            raise ValueError("No NPZ files found in the directory")

        batches = []
        for file in tqdm(self.npz_files, "Loading NPZ files"):
            loaded_np = np.load(file, allow_pickle=True)
            flattened_images = loaded_np['data']
            images = flattened_images.reshape(-1, 3, 32, 32)
            images = images.transpose(0, -2, -1, 1)
            batches.append(images)

        self.data = np.concatenate(batches, axis=0)
        if self.load_noise:
            self.noise = np.array([noise_transform(data) for data in tqdm(self.data, "Generating noise")])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        noised_image = self.noise[idx] if self.load_noise else self.noise_transform(image)
        if self.transform is not None:
            image = self.transform(image)
            noised_image = self.transform(noised_image)
        return image, noised_image


class AutoEncoderImageDataset(Dataset):
    def __init__(self, image_dir: str, load_into_memory: bool, train=True, cut=0.1, transform=None):
        self.image_dir = image_dir
        self.images = []
        self.load_into_memory = load_into_memory
        self.transform = transform
        for root, dirs, files in os.walk(self.image_dir):
            files.sort(key=numerical_sort)
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


class DenoiserImageDataset(Dataset):
    def __init__(self, image_dir: str, noise_transform, train=True, cut=0.1, transform=None):
        self.image_dir = image_dir
        self.images = []
        self.transform = transform
        for root, dirs, files in os.walk(self.image_dir):
            files.sort(key=numerical_sort)
            for file in files:
                image = read_image(os.path.join(root, file), mode=ImageReadMode.RGB)
                if transform is not None:
                    image = transform(image)
                self.images.append((image, noise_transform(image)))

        # On training data, cut the last part from the list
        if not train:
            self.images = self.images[:int(len(self.images) * cut)]
        else:
            self.images = self.images[int(len(self.images) * cut):]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_pair = self.images[idx]
        return image_pair
