import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms

from auto_encoders.unet.unet import BaseUNet
from datasets.datasets import AutoEncoderNPZDataset, DenoiserNPZDataset
from util import image_utils
from training import loss_functions

data_str = "./data/external/image_net/valid"

preprocessing = transforms.Compose([
    transforms.ToTensor(),
    transforms.ToPILImage()
])

dataset = DenoiserNPZDataset(data_str, noise_transform=lambda x: x)

original_image = dataset[3][0]

noised_images = {}

for deviation in [5, 10, 15, 20, 25]:
    gaussian_image = image_utils.add_gaussian_noise(original_image, 0, deviation)
    gaussian_image = preprocessing(gaussian_image)
    noised_images[deviation] = gaussian_image

model = BaseUNet()
state_dict = torch.load("./output/denoise_models/Denoising_Unet_MSE/model_20240827_070155_2")
model.load_state_dict(state_dict)

model.to("cuda:1")

denoised_images = {}

for deviation, noised_image in noised_images.items():
    noised_image = transforms.ToTensor()(noised_image).to("cuda:1")
    noised_image = noised_image.unsqueeze(0)
    denoised_image = model(noised_image)
    denoised_images[deviation] = denoised_image

# Plot the original image and the noised images
fig, axes = plt.subplots(2, 6, figsize=(12, 4))

axes[0][0].imshow(preprocessing(original_image))
axes[0][0].set_title("Original")
axes[0][0].axis('off')

axes[1][0].imshow(preprocessing(original_image))
axes[1][0].set_title("Denoised")
axes[1][0].axis('off')

loss_func = loss_functions.MSELoss()

losses = {}

for i, (deviation, noised_image) in enumerate(noised_images.items()):
    noised_image = transforms.ToTensor()(noised_image).to("cuda:1")
    noised_image = noised_image.unsqueeze(0)
    denoised_image = denoised_images[deviation]
    loss = loss_func(denoised_image, noised_image).item()
    losses[deviation] = loss

for i, (deviation, noised_image) in enumerate(noised_images.items()):
    noise_variance = 1.0 / 255.0 * deviation
    axes[0][i + 1].imshow(noised_image)
    axes[0][i + 1].set_title("Noise Variance: {:1.2f}".format(noise_variance))
    axes[0][i + 1].axis('off')

for i, (deviation, denoised_image) in enumerate(denoised_images.items()):
    axes[1][i + 1].imshow(transforms.ToPILImage()(denoised_image.cpu().squeeze(0)))
    axes[1][i + 1].set_title("MSE Loss: {:1.4f}".format(losses[deviation]))
    axes[1][i + 1].axis('off')

plt.tight_layout()
plt.savefig('noised_images.png')
