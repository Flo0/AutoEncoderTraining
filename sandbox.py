from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from datasets.sampling import matplotlib_imshow_grid

from auto_encoders.custom.primitive import PrimitiveAutoEncoder
from datasets.datasets import AutoEncoderImageDataset

resize = (64, 64)

preprocessing = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(resize),  # Resize the image
    transforms.ToTensor()  # Convert to a tensor with shape (C, H, W) and range [0, 1]
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to range [-1, 1]
])

training_data = AutoEncoderImageDataset(image_dir="./data/pokemon/gen3-processed-rgb", load_into_memory=True, train=True, transform=preprocessing)

print('Training set has {} instances'.format(len(training_data)))

train_data_loader = DataLoader(training_data, batch_size=32, shuffle=True)

model = PrimitiveAutoEncoder(in_out_channels=(3, 3), scaling=0.5)

images = next(iter(train_data_loader))

print(images.shape)

output_images = model(images)

print(output_images.shape)
