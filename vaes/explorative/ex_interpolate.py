import torch
import numpy as np
from tqdm import tqdm
from vaes.explorative.ex_vae import VAE
from vaes.explorative.ex_data_prep import train_loader
import matplotlib.pyplot as plt
from matplotlib import gridspec

def find_closest_latent_vectors(model, dataloader, mu1, mu2):
    """
    Finds the closest latent vectors to the specified points (mu1, mu2) in the latent space.

    Args:
        model (VAE): The trained VAE model.
        dataloader (DataLoader): DataLoader for the dataset.
        mu1 (torch.Tensor): The target latent vector for the first point.
        mu2 (torch.Tensor): The target latent vector for the second point.

    Returns:
        torch.Tensor: The latent vector closest to mu1.
        torch.Tensor: The latent vector closest to mu2.
    """
    model.eval()
    closest_z1 = None
    closest_z2 = None
    min_dist1 = float('inf')
    min_dist2 = float('inf')

    with torch.no_grad():
        for (data, _) in tqdm(dataloader):
            data = data.to(device)
            # Encode the data to get the latent distribution
            dist = model.encode(data)
            mu = dist.mean  # Extract the means of the latent distribution

            # Calculate the distances to the target points mu1 and mu2
            dist1 = torch.norm(mu - mu1, dim=1)
            dist2 = torch.norm(mu - mu2, dim=1)

            # Find the closest latent vector to mu1
            min_dist1_idx = torch.argmin(dist1)
            if dist1[min_dist1_idx] < min_dist1:
                min_dist1 = dist1[min_dist1_idx]
                closest_z1 = mu[min_dist1_idx]

            # Find the closest latent vector to mu2
            min_dist2_idx = torch.argmin(dist2)
            if dist2[min_dist2_idx] < min_dist2:
                min_dist2 = dist2[min_dist2_idx]
                closest_z2 = mu[min_dist2_idx]

    return closest_z1, closest_z2


def interpolate_and_decode(model, z1, z2, N=10):
    """
    Linearly interpolate between two latent vectors and pass them through the decoder.

    Args:
        model (VAE): The trained VAE model.
        z1 (torch.Tensor): The first latent vector.
        z2 (torch.Tensor): The second latent vector.
        N (int): The number of interpolation steps (including the start and end points).

    Returns:
        torch.Tensor: Decoded outputs from the interpolated latent vectors.
    """
    # Generate N interpolation steps
    z_interp = []
    for alpha in np.linspace(0, 1, N):
        z_interpolated = (1 - alpha) * z1 + alpha * z2  # Linear interpolation
        z_interp.append(z_interpolated)

    # Stack the interpolated vectors into a tensor
    z_interp = torch.stack(z_interp).to(z1.device)

    # Pass the interpolated vectors through the decoder
    decoded_outputs = model.decode(z_interp)

    return decoded_outputs


def find_and_interpolate_between_latents(model, dataloader, mu1, mu2, N=10):
    """
    Finds the closest latent vectors to two specified points (mu1, mu2),
    interpolates between them, and decodes the interpolated vectors.

    Args:
        model (VAE): The trained VAE model.
        dataloader (DataLoader): DataLoader for the dataset.
        mu1 (torch.Tensor): The target latent vector for the first point.
        mu2 (torch.Tensor): The target latent vector for the second point.
        N (int): The number of interpolation steps.

    Returns:
        torch.Tensor: Decoded outputs from the interpolated latent vectors.
    """
    # Find the closest latent vectors to the defined mu1 and mu2
    z1, z2 = find_closest_latent_vectors(model, dataloader, mu1, mu2)

    # Perform interpolation and decode the interpolated latent vectors
    decoded_images = interpolate_and_decode(model, z1, z2, N)

    return decoded_images


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_raw = VAE(input_dim=784, hidden_dim=512, latent_dim=2).to(device)

model_raw.load_state_dict(torch.load("../../vae_mnist.pth"))

# Example usage:
# Define target latent vectors in the latent space
mu1 = torch.tensor([-0.9, -1.8]).to(device)
mu2 = torch.tensor([-2.1, 0.1]).to(device)

# Perform interpolation and decode
decoded_images = find_and_interpolate_between_latents(model_raw, train_loader, mu1, mu2, N=20)

# Reshape the decoded images for visualization
decoded_images = decoded_images.view(-1, 1, 28, 28)


num_images = 20
fig = plt.figure(figsize=(num_images, 2))
gs = gridspec.GridSpec(1, num_images, wspace=8 / 100)

for i in range(num_images):
    ax = plt.subplot(gs[0, i])

    ax.imshow(decoded_images[i].detach().cpu().numpy().squeeze(), cmap='gray')

    ax.axis('off')
    ax.set_title(f'{i + 1}', fontsize=18, pad=4)

plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.08, hspace=0)

# Show the plot
plt.savefig('../../vae_interpolation.png')
plt.show()
