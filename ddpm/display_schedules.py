import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


# Schedulers as provided by you
def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def sigmoid_beta_schedule(timesteps: int, start: int = -3, end: int = 3, tau: float = 1) -> torch.Tensor:
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = ((-((t * (end - start) + start) / tau)).sigmoid() - v_start) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # Normalize to ensure first value is 1
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


# Plotting each beta schedule in its own plot
timesteps = 1000
linear_betas = linear_beta_schedule(timesteps)
cosine_betas = cosine_beta_schedule(timesteps)
sigmoid_betas = sigmoid_beta_schedule(timesteps)


# Create a function to plot the schedules and their logarithms
def plot_schedule_with_log(betas: torch.Tensor, scheduler_name: str):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the beta schedule
    axs[0].plot(betas.numpy(), label=f"{scheduler_name} Beta Schedule", color='blue')
    axs[0].set_title(f"{scheduler_name} Beta Schedule")
    axs[0].set_xlabel("Timesteps")
    axs[0].set_ylabel("Beta values")
    axs[0].legend()

    # Plot the log of the beta schedule
    axs[1].plot(torch.log(betas).numpy(), label=f"{scheduler_name} Beta Schedule (Log)", color='orange')
    axs[1].set_title(f"{scheduler_name} Beta Schedule (Log)")
    axs[1].set_xlabel("Timesteps")
    axs[1].set_ylabel("Log(Beta values)")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(f"{scheduler_name}_schedules.png")
    plt.show()


# Plotting each schedule and its logarithm
plot_schedule_with_log(linear_betas, "Linear")
plot_schedule_with_log(cosine_betas, "Cosine")
plot_schedule_with_log(sigmoid_betas, "Sigmoid")


# Function to add noise to an image with adjustable noise intensity
def add_noise(image: torch.Tensor, betas: torch.Tensor, t: int, noise_scale: float = 0.5) -> torch.Tensor:
    alpha_t = torch.cumprod(1 - betas[:t], dim=0)[-1]
    noise = torch.randn_like(image)  # Gaussian noise
    noisy_image = torch.sqrt(alpha_t) * image + noise_scale * torch.sqrt(1 - alpha_t) * noise
    return noisy_image  # Remove clamping here


# Load and prepare the image
def load_image(image_path: str) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB").resize((64, 64))
    img = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
    img = torch.tensor(img).permute(2, 0, 1)  # Convert to (C, H, W) format
    return img


# Modified function to display series of images with noise at different timesteps
def display_noisy_images(image: torch.Tensor, betas: torch.Tensor, scheduler_name: str, noise_scale: float = 0.5):
    steps = torch.linspace(0, len(betas) - 1, 16, dtype=torch.int32).tolist()  # 16 timesteps
    fig, axs = plt.subplots(2, 8, figsize=(24, 6))
    fig.suptitle(f"Noisy Images over Time - {scheduler_name} Scheduler", fontsize=16)

    for i, t in enumerate(steps):
        noisy_img = add_noise(image, betas, t + 1, noise_scale=noise_scale)

        # Apply normalization to ensure the image is displayable
        noisy_img = (noisy_img - noisy_img.min()) / (noisy_img.max() - noisy_img.min())

        noisy_img_np = noisy_img.permute(1, 2, 0).numpy()  # Convert back to (H, W, C)
        ax = axs[i // 8, i % 8]  # Arrange images in 2 rows of 8
        ax.imshow(noisy_img_np)
        ax.set_title(f"Timestep {t}")
        ax.axis("off")

    plt.savefig(f"{scheduler_name}_noisy_images.png")
    plt.show()


# Load the image
image_path = '../data/img.png'  # Provide your image path here
image = load_image(image_path)

# Show 16 noisy images for each scheduler with adjusted noise intensity
noise_scale = 0.3  # Adjust this value to control the noise intensity
display_noisy_images(image, linear_betas, "Linear", noise_scale=noise_scale)
display_noisy_images(image, cosine_betas, "Cosine", noise_scale=noise_scale)
display_noisy_images(image, sigmoid_betas, "Sigmoid", noise_scale=noise_scale)
