from training.auto_encoder_training import *
from auto_encoders.custom.cifar_style import ConvAutoEncoder
from training.loss_functions import *

LEARN_RATE = 0.001
FFL_LOSS_WEIGHT = 1.0

loss_functions = {
    "FFL": FFLLoss(),
    "MSE": MSELoss(),
    "L1": L1Loss(),
    "SSIM_9": SSIMLoss(window_size=9)
}

from util.model_utils import count_parameters

count_parameters(ConvAutoEncoder(keep_dim=True, scale=2.0))

for scale in [1.0, 2.0]:
    for key, loss in loss_functions.items():
        # Model is Primitive AutoEncoder
        model = ConvAutoEncoder(keep_dim=True, scale=scale)

        # Optimizer is Adam
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)

        train_auto_encoder_image_net(
            model=model,
            optimizer=optimizer,
            prefix="BASIC_" + str(scale) + key,
            resize=(32, 32),
            loss_function=loss,
            epochs=16,
            batch_size=5120
        )
