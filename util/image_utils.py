import cv2
import numpy as np
import torch
import PIL.Image as Image
from matplotlib import pyplot as plt
from numpy.fft import fft2, fftshift


def get_frequency_components(image):
    # Perform 2D FFT
    f_transform = fft2(image)
    f_transform_shifted = fftshift(f_transform)

    # Compute magnitude spectrum
    magnitude_spectrum = np.abs(f_transform_shifted)

    # Normalize the magnitude spectrum for display
    magnitude_spectrum = np.log(magnitude_spectrum + 1)  # Using log to compress the range

    # Normalize to range [0, 255] for better visualization
    magnitude_spectrum = (magnitude_spectrum - magnitude_spectrum.min()) / (magnitude_spectrum.max() - magnitude_spectrum.min())
    # magnitude_spectrum *= 255

    return magnitude_spectrum


def display_frequency_components_torch(image):
    # Ensure image is a float tensor
    if image.dtype != torch.float:
        image = image.to(torch.float)

    # Perform 2D FFT
    f_transform = torch.fft.fft2(image)
    f_transform_shifted = torch.fft.fftshift(f_transform)

    # Compute magnitude spectrum
    magnitude_spectrum = torch.abs(f_transform_shifted)

    # Normalize the magnitude spectrum for display
    magnitude_spectrum = torch.log(magnitude_spectrum + 1)  # Log scaling

    # Normalize to range [0, 255] for better visualization
    magnitude_spectrum = (magnitude_spectrum - magnitude_spectrum.min()) / (magnitude_spectrum.max() - magnitude_spectrum.min())
    magnitude_spectrum *= 255

    # Convert tensor to NumPy array for visualization
    magnitude_spectrum_np = magnitude_spectrum.cpu().numpy()

    # Display the magnitude spectrum
    plt.imshow(magnitude_spectrum_np, cmap='gray')
    plt.title('Frequency Domain Representation')
    plt.colorbar()
    plt.show()


def shift_image(img, dx_pixels, dy_pixels):
    if isinstance(img, np.ndarray):
        horiz_shifted = np.roll(img, dx_pixels, axis=1)
        vert_shifted = np.roll(horiz_shifted, dy_pixels, axis=0)
        return vert_shifted
    elif torch.is_tensor(img):
        img = img.cpu().numpy()
        horiz_shifted = np.roll(img, dx_pixels, axis=1)
        vert_shifted = np.roll(horiz_shifted, dy_pixels, axis=0)
        return torch.from_numpy(vert_shifted)
    elif isinstance(img, Image.Image):
        img = np.array(img)
        horiz_shifted = np.roll(img, dx_pixels, axis=1)
        vert_shifted = np.roll(horiz_shifted, dy_pixels, axis=0)
        return vert_shifted
    else:
        raise ValueError("Invalid image type. Must be a numpy array, PyTorch tensor, or PIL image.")


def add_gaussian_noise(image, mean=0.0, std_dev=1.0):
    if isinstance(image, np.ndarray):
        noise = np.random.normal(mean, std_dev, image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    elif torch.is_tensor(image):
        noise = torch.normal(mean, std_dev, size=image.size())
        noisy_image = image + noise
        return torch.clamp(noisy_image, 0, 1)  # Assuming image is in range [0, 1]

    elif isinstance(image, Image.Image):
        image = np.array(image)
        noise = np.random.normal(mean, std_dev, image.shape)
        noisy_image = image + noise
        return Image.fromarray(np.clip(noisy_image, 0, 255).astype(np.uint8))

    else:
        raise ValueError("Invalid image type. Must be a numpy array, PyTorch tensor, or PIL image. But got: {}".format(type(image)))


def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    if isinstance(image, np.ndarray):
        noisy_image = np.copy(image)
        total_pixels = image.size

        # Salt noise (white pixels)
        num_salt = int(total_pixels * salt_prob)
        salt_coords = [np.random.randint(0, i, num_salt) for i in image.shape]
        noisy_image[tuple(salt_coords)] = 255

        # Pepper noise (black pixels)
        num_pepper = int(total_pixels * pepper_prob)
        pepper_coords = [np.random.randint(0, i, num_pepper) for i in image.shape]
        noisy_image[tuple(pepper_coords)] = 0

        return noisy_image

    elif torch.is_tensor(image):
        noisy_image = image.clone()
        total_pixels = image.numel()

        # Salt noise (white pixels)
        num_salt = int(total_pixels * salt_prob)
        salt_coords = [torch.randint(0, i, (num_salt,)) for i in image.shape]
        noisy_image[salt_coords] = 1.0  # Assuming the image is normalized [0, 1]

        # Pepper noise (black pixels)
        num_pepper = int(total_pixels * pepper_prob)
        pepper_coords = [torch.randint(0, i, (num_pepper,)) for i in image.shape]
        noisy_image[pepper_coords] = 0.0

        return noisy_image

    elif isinstance(image, Image.Image):
        image = np.array(image)
        noisy_image = np.copy(image)
        total_pixels = image.size

        # Salt noise (white pixels)
        num_salt = int(total_pixels * salt_prob)
        salt_coords = [np.random.randint(0, i, num_salt) for i in image.shape]
        noisy_image[tuple(salt_coords)] = 255

        # Pepper noise (black pixels)
        num_pepper = int(total_pixels * pepper_prob)
        pepper_coords = [np.random.randint(0, i, num_pepper) for i in image.shape]
        noisy_image[tuple(pepper_coords)] = 0

        return Image.fromarray(noisy_image)

    else:
        raise ValueError("Invalid image type. Must be a numpy array, PyTorch tensor, or PIL image.")


def gaussian_blur(image, kernel_size=3):
    if isinstance(image, np.ndarray):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    elif torch.is_tensor(image):
        image = image.cpu().numpy()
        blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return torch.from_numpy(blurred_image)

    elif isinstance(image, Image.Image):
        image = np.array(image)
        blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return Image.fromarray(blurred_image)

    else:
        raise ValueError("Invalid image type. Must be a numpy array, PyTorch tensor, or PIL image.")
