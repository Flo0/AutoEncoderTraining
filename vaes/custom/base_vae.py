from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEResult:
    def __init__(self, z_distribution, z_sample, x_reconstructed, loss_reconstructed=None, loss_kl=None):
        self.z_distribution = z_distribution
        self.z_sample = z_sample
        self.x_reconstructed = x_reconstructed

        self.loss_reconstructed = loss_reconstructed
        self.loss_kl = loss_kl
        self.total_loss = loss_reconstructed + loss_kl if loss_reconstructed is not None and loss_kl is not None else None


class VariationalAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, sample_sigma=1.0):
        super(VariationalAutoEncoder, self).__init__()
        self.sample_sigma = sample_sigma  # Prior variance of latent space (sigma_x^2)
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        """
        Encodes the input into mean and log variance of the latent space distribution.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            mu (torch.Tensor): Mean of the latent distribution.
            log_sigma2 (torch.Tensor): Log variance of the latent distribution (log(sigma_j^2)).
        """
        mu, log_sigma2 = self.encoder(x)  # Encoder returns mu and log(sigma_j^2)
        return mu, log_sigma2

    def reparameterize(self, mu, log_sigma2):
        """
        Applies the reparameterization trick to sample from N(mu, sigma^2).

        Args:
            mu (torch.Tensor): Mean of the distribution.
            log_sigma2 (torch.Tensor): Log variance of the distribution.

        Returns:
            torch.Tensor: Reparameterized latent variable z.
        """
        sigma = torch.exp(0.5 * log_sigma2)
        epsilon = torch.randn_like(sigma)
        return mu + sigma * epsilon

    def decode(self, latent_z):
        """
        Decodes the latent variable z into the original input space.

        Args:
            latent_z (torch.Tensor): Latent variable sampled from the latent distribution.

        Returns:
            torch.Tensor: Reconstructed input.
        """
        return self.decoder(latent_z)

    def forward(self, input_x, compute_loss=True, recon_error_fn=F.mse_loss):
        """
        Forward pass through the VAE model. If compute_loss is True, it returns the loss.

        Args:
            input_x (torch.Tensor): The input data.
            compute_loss (bool): Whether to compute and return the VAE loss (default: True).
            recon_error_fn (callable): The reconstruction loss function (default: MSE).

        Returns:
            Either reconstructed data, or the total loss (if compute_loss is True).
        """
        # Encode input to get mu and log variance
        mu, log_sigma2 = self.encode(input_x)

        # Reparameterize to get latent variable z
        latent_z = self.reparameterize(mu, log_sigma2)

        recon_loss, kl_loss = None, None

        # Decode to reconstruct the input
        reconstructed_x = self.decode(latent_z)

        if compute_loss:
            # Dimensionality of each input sample (total number of pixels/features)
            data_dim = input_x[0].numel()  # This is the correct input dimensionality

            # Reconstruction error term (||x - f(z)||^2)
            recon_error = recon_error_fn(reconstructed_x, input_x, reduction='sum')

            # Regularization term: D/2 * log(2 * pi * sigma_x^2)
            regularization_term = (data_dim / 2) * torch.log(torch.tensor(2 * torch.pi * self.sample_sigma ** 2))

            # Final reconstruction loss: regularization - recon_error / (2 * sigma_x^2)
            recon_loss = regularization_term - recon_error / (2 * self.sample_sigma ** 2)

            # KL divergence term: 1/2 * sum(1 + log(sigma_j^2) - mu_j^2 - sigma_j^2)
            kl_loss = -0.5 * torch.sum(1 + log_sigma2 - mu.pow(2) - log_sigma2.exp())

        return VAEResult(
            z_distribution=(mu, log_sigma2),
            z_sample=latent_z,
            x_reconstructed=reconstructed_x,
            loss_reconstructed=recon_loss,
            loss_kl=kl_loss
        )
