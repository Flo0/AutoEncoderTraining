import torchvision
from torch import classes
from torchvision.transforms import transforms

from datasets import AutoEncoderImageDataset
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize the image
    transforms.ToTensor(),          # Convert to a tensor with shape (C, H, W) and range [0, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to range [-1, 1]
])

training_data = AutoEncoderImageDataset(image_dir="./data/pokemon/gen3-processed", load_into_memory=True, train=True)
test_data = AutoEncoderImageDataset(image_dir="./data/pokemon/gen3-processed", load_into_memory=True, train=False)

train_data_loader = DataLoader(training_data, batch_size=32, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=32, shuffle=True)

print('Training set has {} instances'.format(len(training_data)))
print('Validation set has {} instances'.format(len(test_data)))

import matplotlib.pyplot as plt
import numpy as np


# Helper function for inline image display
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


dataiter = iter(train_data_loader)
images = next(dataiter)

# Create a grid from the images and show them
img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel=True)
