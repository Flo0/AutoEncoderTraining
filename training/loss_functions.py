import torch
import torch.nn as nn
from focal_frequency_loss import FocalFrequencyLoss as FFL
from pytorch_msssim import SSIM, MS_SSIM
from skimage.metrics import mean_squared_error as ski_mse
from skimage.metrics import structural_similarity as ski_ssim
from skimage.metrics import variation_of_information as ski_vi


class SKIMSELoss(nn.Module):
    def __init__(self):
        super(SKIMSELoss, self).__init__()

    def forward(self, x, y):
        return ski_mse(x, y)

    @staticmethod
    def get_short_name():
        return "SKI_MSE"

    @staticmethod
    def get_full_name():
        return "Scikit-Image Mean Squared Error"


class SKISSIMLoss(nn.Module):
    def __init__(self):
        super(SKISSIMLoss, self).__init__()

    def forward(self, x, y):
        return 1 - ski_ssim(x, y, multichannel=True)

    @staticmethod
    def get_short_name():
        return "SKI_SSIM"

    @staticmethod
    def get_full_name():
        return "Scikit-Image Structural Similarity Index"


class SKIVILoss(nn.Module):
    def __init__(self):
        super(SKIVILoss, self).__init__()

    def forward(self, x, y):
        return ski_vi(x, y)

    @staticmethod
    def get_short_name():
        return "SKI_VI"

    @staticmethod
    def get_full_name():
        return "Scikit-Image Variation of Information"


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, x, y):
        return torch.mean(torch.abs(x - y))

    @staticmethod
    def get_short_name():
        return "L1"

    @staticmethod
    def get_full_name():
        return "L1 Loss"


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, x, y):
        return torch.mean((x - y) ** 2)

    @staticmethod
    def get_short_name():
        return "MSE"

    @staticmethod
    def get_full_name():
        return "Mean Squared Error"


class FFLLoss(nn.Module):
    def __init__(self, loss_weight=1.0, alpha=1.0):
        super(FFLLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.ffl = FFL(loss_weight=loss_weight, alpha=alpha)

    def forward(self, x, y):
        return self.ffl(x, y)

    @staticmethod
    def get_short_name():
        return "FFL"

    @staticmethod
    def get_full_name():
        return "Focal Frequency Loss"


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

        self.ssim = SSIM(win_size=window_size, size_average=size_average, data_range=1.0)

    def forward(self, x, y):
        return 1 - self.ssim(x, y)

    @staticmethod
    def get_short_name():
        return "SSIM"

    @staticmethod
    def get_full_name():
        return "Structural Similarity Index"


class PyFFLLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        """
        alpha: Exponent controlling the strength of the FFL in higher frequencies.
        beta: Exponent controlling the contribution of the FFL to the overall loss.
        """
        super(PyFFLLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, y):
        # Compute Fourier transform of the input and target images
        input_fft = torch.fft.fftn(x, dim=(-2, -1))
        target_fft = torch.fft.fftn(y, dim=(-2, -1))

        # Compute the magnitude of the Fourier coefficients
        input_fft_mag = torch.abs(input_fft)
        target_fft_mag = torch.abs(target_fft)

        # Compute the difference in the Fourier magnitudes
        diff = input_fft_mag - target_fft_mag

        # FFL: weighted by frequency (f^alpha) and summed
        ffl_loss = torch.mean((torch.abs(diff) ** self.alpha) * (input_fft_mag ** self.beta))

        return ffl_loss

    @staticmethod
    def get_short_name():
        return "PY_FFL"

    @staticmethod
    def get_full_name():
        return "PyTorch Focal Frequency Loss"
