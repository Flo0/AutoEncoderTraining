import numpy as np
import matplotlib.pyplot as plt
import cv2


# Linear noise schedule (your existing implementation)
def create_linear_noise_schedule(N, start=0.0001, end=0.1):
    return np.linspace(start, end, N + 1)


# Constant noise schedule
def create_constant_noise_schedule(N, scale=0.025):
    return np.full(N + 1, scale)


# Cosine noise schedule (commonly used in DDPMs)
def create_cosine_noise_schedule(N, stretch=1, scale=0.1):
    timesteps = np.linspace(0, N, N + 1)
    # Slow down the noise increase by raising timesteps to a power
    alpha_bar = np.cos((timesteps / N) ** stretch * np.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]  # Normalize alpha_bar to start from 1
    beta = 1 - alpha_bar  # Noise variance
    return beta * scale


# Quadratic noise schedule
def create_quadratic_noise_schedule(N, scale=0.5):
    return ((np.linspace(0, 1, N + 1)) ** 2) * scale


# Exponential noise schedule
def create_exponential_noise_schedule(N, gamma=1.5, scale=0.1):
    return scale * (1 - np.exp(-gamma * np.linspace(0, 1, N + 1)))


# Sigmoid noise schedule
def create_sigmoid_noise_schedule(N, steepness=10):
    return (1 / (1 + np.exp(-steepness * (np.linspace(0, 1, N + 1) - 0.5))))


# Function to simulate q(x_t | x_{t-1}) ~ N(x_t; sqrt(1 - beta_t) * x_{t-1}, beta_t * I)
def add_noise_markov(image, noise_schedule, N):
    noisy_images = [image.copy()]  # Start with the original image
    for t in range(1, N + 1):
        beta_t = noise_schedule[t]
        # Scale the previous image by sqrt(1 - beta_t)
        noisy_image = np.sqrt(1 - beta_t) * noisy_images[-1]
        # Add Gaussian noise with variance beta_t
        noise = np.random.normal(0, np.sqrt(beta_t) * 255, image.shape)  # Adjust noise intensity
        noisy_image = np.clip(noisy_image + noise, 0, 255).astype(np.uint8)
        noisy_images.append(noisy_image)
    return noisy_images


# Load a sample image (grayscale or RGB)
image_path = './data/img.png'  # Replace with the path to your image
image = cv2.imread(image_path)  # Use IMREAD_COLOR to load RGB images
image = cv2.resize(image, (128, 128))  # Resize for visualization
image = image.astype(np.float32)

# Set parameters
N = 200  # Number of timesteps
S = 10  # Stride: display every S-th image

# You can try different schedules here
schedules = {
    "Linear": create_linear_noise_schedule(N),
    "Constant": create_constant_noise_schedule(N),
    "Cosine": create_cosine_noise_schedule(N),
    "Quadratic": create_quadratic_noise_schedule(N),
    "Exponential": create_exponential_noise_schedule(N),
    "Sigmoid": create_sigmoid_noise_schedule(N)
}

# Plot for each noise schedule
for schedule_name, noise_schedule in schedules.items():
    # Add noise progressively using a Markov process for the current schedule
    noisy_images = add_noise_markov(image, noise_schedule, N)

    # Plotting the noise schedule results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the noise schedule (standard and log scale)
    axes[0].plot(np.arange(N + 1), noise_schedule)
    axes[0].set_title(f'{schedule_name} Noise Schedule (Standard Scale)')
    axes[0].set_xlabel('Timestep')
    axes[0].set_ylabel('Noise Level')

    axes[1].plot(np.arange(N + 1), np.log(noise_schedule + 1e-10))  # Log scale to avoid log(0)
    axes[1].set_title(f'{schedule_name} Noise Schedule (Log Scale)')
    axes[1].set_xlabel('Timestep')
    axes[1].set_ylabel('Log Noise Level')

    plt.tight_layout()

    plt.savefig(f'{schedule_name}_noise_schedule_plots.png')

    # Create a new figure for displaying the noisy images
    fig, axes = plt.subplots(1, len(range(0, N + 1, S)), figsize=(15, 2.5))

    # Minimize white borders and margins between subplots
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.05)

    # Display only every S-th noisy image, keeping aspect ratio consistent
    for idx, i in enumerate(range(0, N + 1, S)):
        if len(image.shape) == 3:  # RGB image
            axes[idx].imshow(cv2.cvtColor(noisy_images[i], cv2.COLOR_BGR2RGB))
        else:  # Grayscale image
            axes[idx].imshow(noisy_images[i], cmap='gray')
        axes[idx].set_title(f'{i}', fontsize=8)
        axes[idx].axis('off')

        # Maintain equal aspect ratio without stretching
        axes[idx].set_aspect('equal')

    # Set image on index 0 to the original image with range 0 - 1.0
    # Also change it to RGB from BGR
    axes[0].imshow(cv2.cvtColor(image / 255, cv2.COLOR_BGR2RGB))

    plt.savefig(f'{schedule_name}_noise_schedule.png')
    plt.show()
