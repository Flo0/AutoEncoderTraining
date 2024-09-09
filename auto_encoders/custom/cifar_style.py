from torch import nn


class ConvAutoEncoder(nn.Module):
    def __init__(self, keep_dim=True, scale=1.0):
        super(ConvAutoEncoder, self).__init__()
        self.keep_dim = keep_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, int(scale * 16), kernel_size=3, stride=2, padding=1),  # [B, 16, 16, 16]
            nn.ReLU(True),
            nn.Conv2d(int(scale * 16), int(scale * 32), kernel_size=3, stride=2, padding=1),  # [B, 32, 8, 8]
            nn.ReLU(True),
            nn.Conv2d(int(scale * 32), int(scale * 64), kernel_size=7),  # [B, 64, 1, 1]
            nn.ReLU(True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(int(scale * 64), int(scale * 32), kernel_size=7),  # [B, 32, 8, 8]
            nn.ReLU(True),
            nn.ConvTranspose2d(int(scale * 32), int(scale * 16), kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 16, 16, 16]
            nn.ReLU(True),
            nn.ConvTranspose2d(int(scale * 16), 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 3, 32, 32]
            nn.Sigmoid()  # To ensure the output is between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        if self.keep_dim:
            x = nn.functional.interpolate(x, size=x.shape[-2:])

        return x
