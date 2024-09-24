import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """ A helper module that performs two convolutional layers with ReLU activation. """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class UNetVAETheory(nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim):
        super(UNetVAETheory, self).__init__()

        # U-Net Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # Latent space: Use 1x1 conv layers for mu and logvar to keep it fully convolutional
        self.conv_mu = nn.Conv2d(1024, latent_dim, kernel_size=1)
        self.conv_logvar = nn.Conv2d(1024, latent_dim, kernel_size=1)

        # Decoder input (to match the latent dimension)
        self.conv_decoder_input = nn.Conv2d(latent_dim, 1024, kernel_size=1)

        # U-Net Decoder
        self.dec4 = ConvBlock(1024 + 512, 512)
        self.dec3 = ConvBlock(512 + 256, 256)
        self.dec2 = ConvBlock(256 + 128, 128)
        self.dec1 = ConvBlock(128 + 64, 64)

        # Final output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        # Maxpooling and upsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upconv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def encode(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        # Compute mu and logvar using 1x1 conv layers
        mu = self.conv_mu(bottleneck)
        logvar = self.conv_logvar(bottleneck)

        return mu, logvar, enc1, enc2, enc3, enc4

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, enc1, enc2, enc3, enc4):
        # Decoder path with skip connections
        z = self.conv_decoder_input(z)  # Convert latent space back to bottleneck shape
        dec4 = self.dec4(torch.cat([self.upconv(z), enc4], dim=1))
        dec3 = self.dec3(torch.cat([self.upconv(dec4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.upconv(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.upconv(dec2), enc1], dim=1))

        return self.final_conv(dec1)

    def forward(self, x):
        # Encode
        mu, logvar, enc1, enc2, enc3, enc4 = self.encode(x)

        # Sample using the reparameterization trick
        z = self.reparameterize(mu, logvar)

        # Decode
        reconstruction = self.decode(z, enc1, enc2, enc3, enc4)
        return reconstruction, mu, logvar


# Loss Function
def vae_loss(recon_x, x, mu, logvar, sigma_x=1.0):
    """VAE loss combining Reconstruction Loss (via MSE) and KL Divergence."""
    # Reconstruction loss (Monte-Carlo Sampling expectation approximation)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / (2 * sigma_x ** 2)

    # KL Divergence term: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_divergence


# Example Usage:
# Create the fully convolutional UNetVAE model
model = UNetVAETheory(in_channels=1, out_channels=1, latent_dim=16)

# Dummy input for a grayscale image of size (1, 64, 64)
input_image = torch.randn(1, 1, 64, 64)

# Forward pass through the model
reconstructed_image, mu, logvar = model(input_image)

# Calculate VAE loss
loss = vae_loss(reconstructed_image, input_image, mu, logvar)
print("VAE Loss:", loss.item())
