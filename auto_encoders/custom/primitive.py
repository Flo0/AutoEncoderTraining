import torch.nn as nn
import torch.nn.functional as NNFunc


class PrimitiveAutoEncoder(nn.Module):
    def __init__(self, in_out_channels=(3, 3), scaling=1.0, keep_dim=True):
        super().__init__()
        self.keep_dim = keep_dim
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_out_channels[0], int(64 * scaling), kernel_size=3),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(int(64 * scaling), int(128 * scaling), kernel_size=3),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(int(128 * scaling), int(256 * scaling), kernel_size=3),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(int(256 * scaling), int(128 * scaling), kernel_size=2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(int(128 * scaling), int(64 * scaling), kernel_size=2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(int(64 * scaling), in_out_channels[1], kernel_size=2, stride=2),
            nn.Sigmoid()
        )

        # Combine the encoder and decoder
        self.codec = nn.Sequential(self.encoder, self.decoder)

    def forward(self, x):
        decoded_features = self.codec(x)
        if self.keep_dim:
            decoded_features = NNFunc.interpolate(decoded_features, x.shape[-2:])
        return decoded_features
