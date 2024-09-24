import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from vaes.explorative.ex_data_prep import test_loader, train_loader
from vaes.explorative.ex_vae import VAE
from sklearn.decomposition import PCA
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_latent_space(model, dataloader):
    """
    Plots the 2D latent space of the VAE using only the mean (mu).
    Args:
        model (VAE): The trained VAE model.
        dataloader (DataLoader): DataLoader for the dataset.
        num_batches (int): Number of batches to plot.
    """
    model.eval()  # Set to evaluation mode
    latents = []
    labels = []

    with torch.no_grad():
        for (data, target) in tqdm(dataloader):
            data = data.to(device)

            # Encode the data to get the distribution (mu and logvar)
            dist = model.encode(data)

            # Get the mean (mu) directly
            mu = dist.mean

            # Store the latent vectors and labels
            latents.append(mu.cpu())
            labels.append(target)

    # Concatenate all latents and labels into single tensors
    latents = torch.cat(latents)
    labels = torch.cat(labels)

    # Plot the latent space
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='tab10', s=5)
    plt.colorbar(scatter, label='Digit Label')
    plt.title(f'2D Latent Space of VAE (Mean, No Sampling)')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.savefig('../../vae_latent_space.png')
    plt.show()


def plot_latent_space_sampled(model, dataloader, num_batches=100, num_samples=10):
    """
    Plots the 2D latent space of the VAE using multi-sampling for a better estimate of the latent space.

    Args:
        model (VAE): The trained VAE model.
        dataloader (DataLoader): DataLoader for the dataset.
        num_batches (int): Number of batches to plot.
        num_samples (int): Number of samples to take from the latent distribution to get a better estimate.
    """
    model.eval()  # Set to evaluation mode
    latents = []
    labels = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            data = data.to(device)

            # Encode the data to get the distribution (mu and logvar)
            dist = model.encode(data)

            # Multi-sample from the latent distribution
            z_samples = []
            for _ in range(num_samples):
                z_samples.append(model.reparameterize(dist).cpu())
            z_samples = torch.stack(z_samples)

            # Average the latent samples to get a smoother representation
            z_mean = z_samples.mean(dim=0)

            # Store the latent vectors and labels
            latents.append(z_mean)
            labels.append(target)

    # Concatenate all latents and labels into single tensors
    latents = torch.cat(latents)
    labels = torch.cat(labels)

    # Plot the latent space
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='tab10', s=5)
    plt.colorbar(scatter, label='Digit Label')
    plt.title(f'2D Latent Space of VAE (Multi-Sampling, {num_samples} Samples)')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.savefig('../../vae_latent_space_sampled.png')
    plt.show()


def plot_latent_space_pca(model, dataloader, num_batches=100):
    """
    Plots the 2D latent space of the VAE using PCA for dimensionality reduction.

    Args:
        model (VAE): The trained VAE model.
        dataloader (DataLoader): DataLoader for the dataset.
        num_batches (int): Number of batches to plot.
    """
    model.eval()
    latents = []
    labels = []

    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(dataloader), "PCA", num_batches):
            if batch_idx >= num_batches:
                break
            data = data.to(device)

            # Get the mean (mu) from the latent distribution
            dist = model.encode(data)
            mu = dist.mean  # Use the mean for visualization

            # Store the latent vectors and labels
            latents.append(mu.cpu())
            labels.append(target)

    # Concatenate all latents and labels into single tensors
    latents = torch.cat(latents)
    labels = torch.cat(labels)

    # Apply PCA to reduce dimensionality to 2D
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents)

    # Plot the latent space
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap='tab10', s=5)
    plt.colorbar(scatter, label='Digit Label')
    plt.title('2D Latent Space of VAE using PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig('../../vae_latent_space_pca.png')
    plt.show()


def plot_latent_space_with_interpolation(model, dataloader, x1, y1, x2, y2, K=10):
    """
    Plots the 2D latent space of the VAE using only the mean (mu) and interpolates between two points.

    Args:
        model (VAE): The trained VAE model.
        dataloader (DataLoader): DataLoader for the dataset.
        x1, y1: Coordinates of the first point in latent space.
        x2, y2: Coordinates of the second point in latent space.
        K (int): Number of interpolation steps.
    """
    model.eval()  # Set to evaluation mode
    latents = []
    labels = []

    with torch.no_grad():
        for (data, target) in tqdm(dataloader):
            data = data.to(device)

            # Encode the data to get the distribution (mu and logvar)
            dist = model.encode(data)

            # Get the mean (mu) directly
            mu = dist.mean

            # Store the latent vectors and labels
            latents.append(mu.cpu())
            labels.append(target)

    # Concatenate all latents and labels into single tensors
    latents = torch.cat(latents)
    labels = torch.cat(labels)

    # Plot the latent space
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='tab10', s=5)
    plt.colorbar(scatter, label='Digit Label')

    # Linear interpolation
    interp_x = np.linspace(x1, x2, K)
    interp_y = np.linspace(y1, y2, K)

    # Plot the interpolation path with a dashed line and lower opacity
    plt.plot(interp_x, interp_y, 'r--', linewidth=2, alpha=0.7)  # Red dashed line

    # Highlight the start and end points of the interpolation
    plt.scatter([x1, x2], [y1, y2], c='red', s=50, edgecolor='black', label='Interpolation Points')

    plt.title(f'2D Latent Space of VAE with Linear Interpolation')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')

    # Save the figure
    plt.savefig('../../vae_latent_space_with_interpolation.png')

    # Show the plot
    plt.legend()
    plt.show()



for beta in [0.2, 0.5, 1, 2, 5]:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = VAE(input_dim=784, hidden_dim=512, latent_dim=2, beta=beta).to(device)

    model.load_state_dict(torch.load("../../vae_mnist.pth"))

    # Example: Call the function and pass in two points from the latent space
    plot_latent_space_with_interpolation(model, train_loader, x1=-0.9, y1=-1.8, x2=-2.1, y2=0.1, K=20)

    # Use the trained model and dataloader
    # plot_latent_space(model, train_loader)

# Use the trained model and dataloader
# plot_latent_space(model_raw, train_loader)
