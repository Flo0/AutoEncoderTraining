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


class UNetVAE(nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim):
        super(UNetVAE, self).__init__()

        # U-Net Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # Variational Latent Space
        self.fc_mu = nn.Linear(1024 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(1024 * 4 * 4, latent_dim)

        # Decoder input for sampling
        self.fc_decoder = nn.Linear(latent_dim, 1024 * 4 * 4)

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

        # Flatten and produce mu and logvar for the latent space
        bottleneck_flat = bottleneck.view(bottleneck.size(0), -1)
        mu = self.fc_mu(bottleneck_flat)
        logvar = self.fc_logvar(bottleneck_flat)

        return mu, logvar, enc1, enc2, enc3, enc4

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, enc1, enc2, enc3, enc4):
        # Project back to the bottleneck shape
        z = self.fc_decoder(z)
        z = z.view(z.size(0), 1024, 4, 4)

        # Decoder path with skip connections
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


# Example Usage:
# Create the UNetVAE model
model = UNetVAE(in_channels=1, out_channels=1, latent_dim=16)

# Dummy input for a grayscale image of size (1, 64, 64)
input_image = torch.randn(1, 1, 64, 64)

# Forward pass through the model
reconstructed_image, mu, logvar = model(input_image)

print(reconstructed_image.shape, mu.shape, logvar.shape)