import torch
import torch.nn as nn
import torch.nn.functional as F

from base_vae import *


class FCNNEncoder(nn.Module):
    def __init__(self, input_channels, latent_channels):
        """
        Fully Convolutional Neural Network Encoder for VAE.

        Args:
            input_channels (int): Number of input image channels (e.g., 1 for grayscale, 3 for RGB).
            latent_channels (int): Number of channels in the latent space.
        """
        super(FCNNEncoder, self).__init__()

        # Convolutional layers to compress the input
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1)  # 32x32 -> 16x16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 16x16 -> 8x8
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 8x8 -> 4x4

        # Final convolution to output the latent space directly as a feature map
        self.conv4_mu = nn.Conv2d(128, latent_channels, kernel_size=4, stride=1, padding=0)  # 4x4 -> 1x1 (latent space)
        self.conv4_logvar = nn.Conv2d(128, latent_channels, kernel_size=4, stride=1, padding=0)  # 4x4 -> 1x1

    def forward(self, x):
        # Apply convolutions with ReLU activations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Output the latent space without flattening
        mu = self.conv4_mu(x)
        log_sigma2 = self.conv4_logvar(x)

        return mu, log_sigma2


class FCNNDecoder(nn.Module):
    def __init__(self, latent_channels, output_channels):
        """
        Fully Convolutional Neural Network Decoder for VAE.

        Args:
            latent_channels (int): Number of channels in the latent space.
            output_channels (int): Number of output image channels (e.g., 1 for grayscale, 3 for RGB).
        """
        super(FCNNDecoder, self).__init__()

        # Transposed convolutional layers to upsample the latent space
        self.deconv1 = nn.ConvTranspose2d(latent_channels, 128, kernel_size=4, stride=1, padding=0)  # 1x1 -> 4x4
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 4x4 -> 8x8
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 8x8 -> 16x16
        self.deconv4 = nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1)  # 16x16 -> 32x32

    def forward(self, z):
        # Apply transposed convolutions with ReLU activations
        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        z = F.relu(self.deconv3(z))

        # Final layer uses a sigmoid to output values between 0 and 1 (for images)
        z = torch.sigmoid(self.deconv4(z))

        return z


class FCNNVAE(VariationalAutoEncoder):
    def __init__(self, input_channels, latent_channels):
        """
        Fully Convolutional Neural Network VAE.

        Args:
            input_channels (int): Number of input image channels (e.g., 1 for grayscale, 3 for RGB).
            latent_channels (int): Number of channels in the latent space.
        """
        encoder = FCNNEncoder(input_channels, latent_channels)
        decoder = FCNNDecoder(latent_channels, input_channels)
        super(FCNNVAE, self).__init__(encoder, decoder)
