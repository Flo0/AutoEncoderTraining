import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torch.utils.data import DataLoader


# Helper function for inline image display
def matplotlib_imshow(img, one_channel=False):
    # Convert to float for operations and normalize
    if one_channel:
        img = img.float()
        img = img.mean(dim=0)
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg)
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def matplotlib_imshow_grid(images, one_channel=False):
    return matplotlib_imshow(torchvision.utils.make_grid(images), one_channel=one_channel)


def sample_dataset(dataset, n_samples):
    data_loader = DataLoader(dataset, batch_size=n_samples, shuffle=True)
    dataiter = iter(data_loader)
    images = next(dataiter)
    # Create a grid from the images and show them
    matplotlib_imshow_grid(images, one_channel=False)
