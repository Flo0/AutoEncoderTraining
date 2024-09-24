from vaes.explorative.ex_data_prep import train_loader, test_loader
from vaes.explorative.ex_vae import VAE

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_latent_space_pca_all(models, dataloader, metas, num_batches=100):
    """
    Plots the 2D latent space of the VAE using PCA for different beta values side by side.

    Args:
        models (list): List of trained VAE models.
        dataloader (DataLoader): DataLoader for the dataset.
        metas (list): List of meta values corresponding to the models.
        num_batches (int): Number of batches to plot.
    """

    # Create a side-by-side plot for each beta
    fig, axs = plt.subplots(4, 4, figsize=(14, 14), sharex=True, sharey=True)

    for idx, model in enumerate(models):
        model.eval()
        latents = []
        labels = []

        with torch.no_grad():
            for batch_idx, (data, target) in tqdm(enumerate(dataloader), f"PCA ({metas[idx]})", total=num_batches):
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

        x_index = idx // 4
        y_index = idx % 4

        # Plot the PCA result for each beta on a different subplot
        scatter = axs[x_index][y_index].scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap='tab10', s=5)
        axs[x_index][y_index].set_title(f'PCA ({metas[idx]})')
        axs[x_index][y_index].set_xlabel('PC 1')
        axs[x_index][y_index].set_ylabel('PC 2')

    # Add colorbar to the right-most plot
    # fig.colorbar(scatter, ax=axs, label='Digit Label', orientation='vertical')

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_latent_space_pca_consistent(models, dataloader, betas, num_batches=100):
    """
    Plots the 2D latent space of the VAE using PCA for different beta values side by side,
    ensuring all plots share the same PCA transformation.

    Args:
        models (list): List of trained VAE models.
        dataloader (DataLoader): DataLoader for the dataset.
        betas (list): List of beta values corresponding to the models.
        num_batches (int): Number of batches to plot.
    """
    latents_all = []  # Collect latent vectors from all models
    labels_all = []  # Collect labels for consistency

    # First pass: Collect all latent vectors from all models
    for model in models:
        model.eval()
        latents = []
        labels = []

        with torch.no_grad():
            for batch_idx, (data, target) in tqdm(enumerate(dataloader), "Collecting latents", total=num_batches):
                if batch_idx >= num_batches:
                    break
                data = data.to(device)

                # Get the mean (mu) from the latent distribution
                dist = model.encode(data)
                mu = dist.mean  # Use the mean for visualization

                # Store the latent vectors and labels
                latents.append(mu.cpu())
                labels.append(target)

        # Concatenate latent vectors and labels for each model
        latents_all.append(torch.cat(latents))
        labels_all.append(torch.cat(labels))

    # Concatenate all latents across models for consistent PCA
    latents_all_concat = torch.cat(latents_all).numpy()
    labels_all_concat = torch.cat(labels_all)

    # Apply PCA on the concatenated latent space across all models
    pca = PCA(n_components=2)
    latents_all_pca = pca.fit_transform(latents_all_concat)

    # Split back the PCA-transformed latents for each model
    latents_all_split = np.split(latents_all_pca, len(models))

    # Create side-by-side plots for each beta
    fig, axs = plt.subplots(4, int(len(models) / 4), figsize=(5 * len(models), 5))

    for idx, (latents_2d, labels, beta) in enumerate(zip(latents_all_split, labels_all, betas)):
        axs[idx / 4][idx % 4].scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels.numpy(), cmap='tab10', s=5)
        axs[idx / 4][idx % 4].set_title(f'PCA (Beta={beta})')
        axs[idx / 4][idx % 4].set_xlabel('Principal Component 1')
        axs[idx / 4][idx % 4].set_ylabel('Principal Component 2')

    # Add colorbar to the right-most plot
    # fig.colorbar(scatter, ax=axs, label='Digit Label', orientation='vertical')

    # Adjust layout
    plt.tight_layout()
    plt.show()


# Load the models corresponding to different beta values
models = []

ldims_in = [2, 3, 4, 16]
betas_in = [0.1, 2, 10, 50]

for beta in betas_in:
    for ldim in ldims_in:
        model = VAE(input_dim=784, hidden_dim=512, latent_dim=ldim, beta=beta).to(device)
        model.load_state_dict(torch.load(f"../../vae_mnist_ldim_{ldim}_beta_{beta}.pth"))
        models.append(model)

metas = []

for beta in betas_in:
    for ldim in ldims_in:
        metas.append(f"beta={beta}, ldim={ldim}")

# Call the function to generate side-by-side PCA plots for different beta values
plot_latent_space_pca_all(models, train_loader, metas, num_batches=100)
