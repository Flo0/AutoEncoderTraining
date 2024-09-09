from matplotlib import pyplot as plt
from torchvision import transforms

from datasets.datasets import AutoEncoderNPZDataset
from util import image_utils
from training import loss_functions

data_str = "./data/external/image_net/valid"

preprocessing = transforms.Compose([
    transforms.ToTensor(),
    transforms.ToPILImage()
])

dataset = AutoEncoderNPZDataset(data_str)

original_image = dataset[5]
gaussian_image = image_utils.add_gaussian_noise(original_image, 0, 8)
salt_pepper_image = image_utils.add_salt_pepper_noise(original_image, 0.01, 0.01)
horizontal_shift_image = image_utils.shift_image(original_image, 2, 0)
vertical_shift_image = image_utils.shift_image(original_image, 0, 2)
gaussian_blur_image = image_utils.gaussian_blur(original_image, 3)

plot_data = {
    "Original\n": original_image,
    "Gaussian\nNoise": gaussian_image,
    "Salt & Pepper\n": salt_pepper_image,
    "Horizontal Shift (2px)\n": horizontal_shift_image,
    "Vertical Shift (2px)\n": vertical_shift_image,
    "Gaussian Blur\n (3x3)": gaussian_blur_image
}

n_cols = len(plot_data)
n_rows = 1

fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 8))

axes = axes.flatten()

for i, (label, img) in enumerate(plot_data.items()):
    axes[i].imshow(preprocessing(img))
    axes[i].set_title(label)
    axes[i].axis('off')

loss_func = {
    "MSE": lambda x, y: loss_functions.MSELoss()(x, y) * 10.0,
    "L1": loss_functions.L1Loss(),
    "FFL (1.0)": lambda x, y: loss_functions.FFLLoss()(x, y) * 10.0,
    "SSIM (3x3)": lambda x, y: loss_functions.SSIMLoss(window_size=3)(x, y) / 10.0
}

# Convert images to PyTorch tensors
plot_data_dis = {label: transforms.ToTensor()(img) for label, img in plot_data.items()}

plot_data_dis = {label: img.unsqueeze(0) for label, img in plot_data_dis.items()}


# Compute losses for each transformation
loss_values = {
    label: {loss_name: loss_func[loss_name](img, plot_data_dis["Original\n"]).item() for loss_name in loss_func}
    for label, img in plot_data_dis.items()
}

# Plot images with loss values displayed below
fig, axes = plt.subplots(2, n_cols, figsize=(16, 8))

# Plot images
for i, (label, img) in enumerate(plot_data.items()):
    axes[0, i].imshow(preprocessing(img))
    axes[0, i].set_title(label)
    axes[0, i].axis('off')

# Display loss values
for i, (label, _) in enumerate(plot_data.items()):
    loss_text = "\n".join([f"{loss_name}: {loss_values[label][loss_name]:.4f}" for loss_name in loss_func])
    axes[1, i].text(0.5, 0.5, loss_text, fontsize=12, ha='center', va='center')
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()