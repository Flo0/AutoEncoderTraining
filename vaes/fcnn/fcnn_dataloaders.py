import torch
from torchvision import transforms

from datasets.datasets import AutoEncoderNPZDataset

batch_size = 128

preprocessing = transforms.Compose([
    transforms.ToTensor()
])

train_data = AutoEncoderNPZDataset("./data/external/image_net/train", preprocessing)
test_data = AutoEncoderNPZDataset("./data/external/image_net/valid", preprocessing)

# Create data loaders
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False,
)
