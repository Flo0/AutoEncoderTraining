import cv2
import numpy as np


# Function to add Gaussian noise to an image
def add_gaussian_noise(image, noise_level):
    """
    Adds Gaussian noise to the image.
    :param image: Input image
    :param noise_level: Standard deviation of the noise
    :return: Noisy image
    """
    noise = np.random.normal(0, noise_level, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255)  # Clip to valid image range
    return noisy_image.astype(np.uint8)


def save_image(image, title="Image"):
    # Save the image directly in BGR format
    cv2.imwrite(f"{title}.jpg", image)


# Load an image using OpenCV
image_path = './data/img.png'
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Ensure image is in float32 format for noise addition
image = image.astype(np.float32)

noise_steps = 20
max_noise_level = 150
incremental_noise = max_noise_level / noise_steps  # Incremental noise level

# Save the original image
save_image(image, title="image_0")

# Recursively add noise to the image
for i in range(1, noise_steps + 1):
    noise_level = incremental_noise * i
    # Add noise to the progressively noisier image
    image = add_gaussian_noise(image, noise_level)
    save_image(image, title=f"image_{i}")

print("Recursive noise addition completed.")
