from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VAEResult:
    """
    Dataclass for VAE output.

    Attributes:
        z_distribution (torch.distributions.Distribution): The distribution of the latent variable z.
        z_sample (torch.Tensor): The sampled value of the latent variable z.
        x_reconstructed (torch.Tensor): The reconstructed output from the VAE.
        loss (torch.Tensor): The overall loss of the VAE.
        loss_reconstructed (torch.Tensor): The reconstruction loss component of the VAE loss.
        loss_kl (torch.Tensor): The KL divergence component of the VAE loss.
    """
    z_distribution: torch.distributions.Distribution
    z_sample: torch.Tensor
    x_reconstructed: torch.Tensor

    loss: torch.Tensor
    loss_reconstructed: torch.Tensor
    loss_kl: torch.Tensor


class VariationalAutoEncoder(nn.Module):
    """
    Variational Autoencoder (VAE) class.

    Args:
        encoder (nn.Module): The encoder network.
        decoder (nn.Module): The decoder network.
        base_accumulator (nn.Module): The accumulator function for the base distribution (default: nn.Softplus()).
    """

    def __init__(self, encoder, decoder, base_accumulator=nn.Softplus()):
        super(VariationalAutoEncoder, self).__init__()

        self.encoder = encoder
        self.base_accumulator = base_accumulator
        self.decoder = decoder

    def encode(self, x, eps: float = 1e-8):
        """
        Encodes the input data into the latent space.

        Args:
            x (torch.Tensor): Input data.
            eps (float): Small value to avoid numerical instability.

        Returns:
            torch.distributions.MultivariateNormal: Normal distribution of the encoded data.
        """

        encoded_x = self.encoder(x)
        mu, sigma = torch.chunk(encoded_x, 2, dim=-1)
        scale = self.base_accumulator(sigma) + eps
        scale_tril = torch.diag_embed(scale)

        return torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)

    def reparameterize(self, distribution):
        """
        Reparameterizes the encoded data to sample from the latent space.

        Args:
            distribution (torch.distributions.MultivariateNormal): Normal distribution of the encoded data.
        Returns:
            torch.Tensor: Sampled data from the latent space.
        """
        return distribution.rsample()

    def decode(self, z):
        """
        Decodes the data from the latent space to the original input space.

        Args:
            z (torch.Tensor): Data in the latent space.

        Returns:
            torch.Tensor: Reconstructed data in the original input space.
        """
        return self.decoder(z)

    def forward(self, x, compute_loss: bool = True):
        """
        Performs a forward pass of the VAE.

        Args:
            x (torch.Tensor): Input data.
            compute_loss (bool): Whether to compute the loss or not.

        Returns:
            VAEResult: VAE output dataclass.
        """
        encoded_distribution = self.encode(x)
        z = self.reparameterize(encoded_distribution)

        recon_x = self.decode(z)

        if not compute_loss:
            return VAEResult(
                z_distribution=encoded_distribution,
                z_sample=z,
                x_reconstructed=recon_x,
                loss=None,
                loss_reconstructed=None,
                loss_kl=None,
            )

        # compute loss terms
        loss_recon = F.mse_loss(recon_x, x, reduction="mean")
        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros_like(z, device=z.device),
            scale_tril=torch.eye(z.shape[-1], device=z.device).unsqueeze(0).expand(z.shape[0], -1, -1),
        )
        loss_kl = torch.distributions.kl.kl_divergence(encoded_distribution, std_normal).mean()

        loss = loss_recon + loss_kl

        return VAEResult(
            z_distribution=encoded_distribution,
            z_sample=z,
            x_reconstructed=recon_x,
            loss=loss,
            loss_reconstructed=loss_recon,
            loss_kl=loss_kl,
        )
