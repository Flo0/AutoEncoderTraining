import torch
import torch.nn as nn
import torchvision


class UBlock(nn.Module):
    def __init__(self, input_channel_count, output_channel_count):
        super().__init__()
        self.conv_in = nn.Conv2d(input_channel_count, output_channel_count, 3)
        self.relu = nn.ReLU()
        self.conv_out = nn.Conv2d(output_channel_count, output_channel_count, 3)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.relu(x)
        x = self.conv_out(x)
        return x


class UEncoder(nn.Module):
    def __init__(self, channels=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        encoder_parts = []
        for channel_index in range(len(channels) - 1):
            block = UBlock(channels[channel_index], channels[channel_index + 1])
            encoder_parts.append(block)

        self.encoder_parts = nn.ModuleList(encoder_parts)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        features = []
        for block in self.encoder_parts:
            x = block(x)
            features.append(x)
            x = self.pool(x)
        return features


class UDecoder(nn.Module):
    def __init__(self, channels=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.channels = channels
        up_convolutions = []
        decoder_parts = []

        for channel_index in range(len(channels) - 1):
            transposed_conv = nn.ConvTranspose2d(channels[channel_index], channels[channel_index + 1], 2, 2)
            up_convolutions.append(transposed_conv)

            block = UBlock(channels[channel_index], channels[channel_index + 1])
            decoder_parts.append(block)

        self.up_convolutions = nn.ModuleList(up_convolutions)
        self.decoder_parts = nn.ModuleList(decoder_parts)

    def forward(self, x, encoder_features):
        for i in range(len(self.channels) - 1):
            x = self.up_convolutions[i](x)
            cropped_features = self.crop(encoder_features[i], x)
            x = torch.cat([x, cropped_features], dim=1)
            x = self.decoder_parts[i](x)
        return x

    def crop(self, encoder_features, x):
        _, _, height, width = x.shape
        encoder_features = torchvision.transforms.CenterCrop([height, width])(encoder_features)
        return encoder_features
