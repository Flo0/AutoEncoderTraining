from vaes.explorative.ex_data_prep import train_loader, test_loader
from vaes.explorative.ex_vae import VAE

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_latent_space_pca_all_3d(models, dataloader, metas, num_batches=100):
    """
    Plots the 3D latent space of the VAE using PCA for different beta values side by side.

    Args:
        models (list): List of trained VAE models.
        dataloader (DataLoader): DataLoader for the dataset.
        metas (list): List of meta values corresponding to the models.
        num_batches (int): Number of batches to plot.
    """

    # Create a figure with 3D subplots for each model
    fig = plt.figure(figsize=(14, 14))
    plot_idx = 0  # Counter for successful plots

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

        # Check if the latent space dimension is >= 3
        if latents.shape[1] < 3:
            print(f"Skipping model {metas[idx]}: latent dimension {latents.shape[1]} is less than 3.")
            continue

        # Apply PCA to reduce dimensionality to 3D
        pca = PCA(n_components=3)
        latents_3d = pca.fit_transform(latents)

        # Create a 3D subplot for each model
        ax = fig.add_subplot(3, 3, plot_idx + 1, projection='3d')
        scatter = ax.scatter(latents_3d[:, 0], latents_3d[:, 1], latents_3d[:, 2], c=labels, cmap='tab10', s=5)
        ax.set_title(f'PCA 3D ({metas[idx]})')
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')

        plot_idx += 1  # Increment plot counter

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.savefig('pca_3d.png')
    plt.show()


# Load the models corresponding to different beta values
models = []

ldims_in = [3, 4, 16]
betas_in = [0.1, 2, 10]

for beta in betas_in:
    for ldim in ldims_in:
        model = VAE(input_dim=784, hidden_dim=512, latent_dim=ldim, beta=beta).to(device)
        model.load_state_dict(torch.load(f"../../vae_mnist_ldim_{ldim}_beta_{beta}.pth"))
        models.append(model)

metas = []

for beta in betas_in:
    for ldim in ldims_in:
        metas.append(f"beta={beta}, ldim={ldim}")

# Call the function to generate side-by-side 3D PCA plots for different beta values
plot_latent_space_pca_all_3d(models, train_loader, metas, num_batches=100)
