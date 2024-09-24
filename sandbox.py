from datasets.datasets import VAENPZDataset
from torchvision.transforms import v2
import matplotlib.pyplot as plt

batch_size = 64
learning_rate = 1e-3
weight_decay = 1e-2
num_epochs = 10
latent_dim = 5
hidden_dim = 512

valid_dataset = VAENPZDataset(npz_dir='./data/external/image_net/valid', n_biggest_classes=latent_dim, transform=None)

# Display all images in the dataset

images = len(valid_dataset)
cols = 10
rows = images // cols + 1

fig, axs = plt.subplots(rows, cols, figsize=(20, 20))

for i in range(images):
    ax = axs[i // cols, i % cols]
    ax.imshow(valid_dataset[i][1])
    ax.axis('off')

plt.show()