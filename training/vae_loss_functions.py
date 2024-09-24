import torch
from torch import nn
import torch.nn.functional as F


# Loss function for VAE: Reconstruction loss + KL divergence
def vae_loss(reconstructed_x, original_x, mu, logvar, reconstruction_loss_fn):
    recon_loss = reconstruction_loss_fn(reconstructed_x, original_x)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_divergence


# Monte carlo KL divergence
def kl_divergence(z, mu, std):
    # 1. define the first two probabilities (in this case Normal for both)
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)

    # 2. get the probabilities from the equation
    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)

    # kl
    kl = (log_qzx - log_pz)

    # sum over last dim to go from single dim distribution to multi-dim
    kl = kl.sum(-1)
    return kl


def vae_loss(base_sigma, original_x, reconstructed_x, latent_mu, latent_sigma, recon_error_fn=F.mse_loss):
    """
    Compute the loss for Variational Autoencoder, using minibatch processing.

    Parameters:
    - data_dimension: Dimension of the input data
    - base_sigma: Standard deviation of reconstruction (sigma_x)
    - original_x: The original input batch
    - reconstructed_x: The reconstructed input batch
    - latent_mu: Mean of the latent variable distribution
    - latent_sigma: Standard deviation of the latent variable distribution
    - recon_loss_fn: The reconstruction loss function (default: Mean Squared Error)

    Returns:
    - Total loss (regularized reconstruction + KL divergence)
    """

    # Compute the reconstruction error (||x - f(z)||^2)
    recon_error = recon_error_fn(reconstructed_x, original_x)

    # Regularize the reconstruction loss by base_sigma (σ_x^2)
    regularized_recon_loss = (original_x.size(0) / 2) * torch.log(2 * torch.pi * base_sigma ** 2) - recon_error / (2 * base_sigma ** 2)

    # KL divergence loss
    kl_divergence = 0.5 * torch.sum(1 + torch.log(latent_sigma ** 2) - latent_mu ** 2 - latent_sigma ** 2)

    # Scale KL divergence by minibatch size
    kl_divergence /= original_x.size(0)

    # Total loss: regularized reconstruction loss + KL divergence
    total_loss = regularized_recon_loss + kl_divergence
    return total_loss
