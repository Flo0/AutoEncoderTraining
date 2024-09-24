import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x_pooled = self.pool(x)
        return x, x_pooled


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(mid_channels + mid_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class VariationalUNET(nn.Module):
    def __init__(self, input_channels, output_channels, features=[64, 128, 256, 512]):
        super(VariationalUNET, self).__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.features = features

        # Encoder part
        for feature in features:
            self.encoders.append(EncoderBlock(input_channels, feature))
            input_channels = feature

        # Bottleneck part (probabilistic)
        self.fc_mu = nn.Linear(features[-1] * 16 * 16, 100)  # Adapt the size according to your bottleneck feature map size
        self.fc_var = nn.Linear(features[-1] * 16 * 16, 100)

        # Decoder part
        for feature in reversed(features):
            self.decoders.append(DecoderBlock(feature * 2, feature, feature))

        self.final_conv = nn.Conv2d(features[0], output_channels, kernel_size=1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        skip_connections = []

        for encoder in self.encoders:
            x, x_pooled = encoder(x)
            skip_connections.append(x)
            x = x_pooled

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        x = self.reparameterize(mu, log_var)
        x = x.view(-1, self.features[-1], 16, 16)  # Adjust shape accordingly

        skip_connections = skip_connections[::-1]

        for idx, decoder in enumerate(self.decoders):
            x = decoder(x, skip_connections[idx])

        return self.final_conv(x), mu, log_var


# Example of how to use this module
input_channels = 3  # For RGB images
output_channels = 3  # For RGB output
model = VariationalUNET(input_channels, output_channels)