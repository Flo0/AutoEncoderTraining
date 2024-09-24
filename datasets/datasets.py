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


class VAEImageDataset(Dataset):
    def __init__(self, image_dir: str, transform=None):
        self.image_dir = image_dir
        self.images = []
        self.transform = transform
        for root, dirs, files in os.walk(self.image_dir):
            files.sort(key=numerical_sort)
            for file in tqdm(files, "Loading images"):
                image = read_image(os.path.join(root, file), mode=ImageReadMode.RGB)
                if transform is not None:
                    image = transform(image)
                self.images.append(image)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        return image


class VAENPZDataset(Dataset):
    def __init__(self, npz_dir: str, transform=None, n_biggest_classes=None, dimensionality=32):
        self.npz_dir = npz_dir
        npz_files = []
        self.transform = transform

        for root, dirs, files in os.walk(npz_dir):
            for file in files:
                if file.endswith('.npz'):
                    npz_files.append(os.path.join(root, file))

        if len(npz_files) == 0:
            raise ValueError("No NPZ files found in the directory")

        elements = []
        for file in tqdm(npz_files, "Loading NPZ files"):
            loaded_np = np.load(file, allow_pickle=True)

            if 'labels' not in loaded_np:
                raise ValueError("No labels found in the NPZ file")

            flattened_images = loaded_np['data']
            images = flattened_images.reshape(-1, 3, dimensionality, dimensionality)
            images = images.transpose(0, -2, -1, 1)

            for label, data in zip(loaded_np['labels'], images):
                elements.append((label, data))

        if n_biggest_classes is not None:
            # Count the occurrences of each class
            label_counts = {}
            for label, _ in tqdm(elements, "Counting labels"):
                if label not in label_counts:
                    label_counts[label] = 0
                label_counts[label] += 1

            # Sort the classes by the number of occurrences
            sorted_labels = sorted(label_counts.items(), key=lambda item: item[1])

            # Take the n_biggest_classes most common classes
            biggest_classes = sorted_labels[:n_biggest_classes]

            biggest_class_label_set = set([label for label, _ in biggest_classes])

            # Filter the elements to only include the biggest classes
            elements = [
                (label, data) for label, data in tqdm(elements, "Filtering elements") if label in biggest_class_label_set
            ]

        self.elements = elements

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, idx):
        labels, data = self.elements[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data
