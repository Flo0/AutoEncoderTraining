import torch.nn.functional as NNFunc
from unet_parts import *


class UNet(nn.Module):
    def __init__(self, enc_chs=(3, 64, 128, 256, 512, 1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1, retain_dim=False):
        super().__init__()
        self.encoder = UEncoder(enc_chs)
        self.decoder = UDecoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim = retain_dim

    def forward(self, x):
        encoded_features = self.encoder(x)
        out = self.decoder(encoded_features[::-1][0], encoded_features[::-1][1:])
        out = self.head(out)
        if self.retain_dim:
            out = NNFunc.interpolate(out, x.shape[-2:])
        return out
