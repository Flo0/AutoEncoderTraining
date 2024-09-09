import training.denoising_training as denoising_training
from auto_encoders.unet.unet import *
from training.loss_functions import *
from util.model_utils import count_parameters

LEARN_RATE = 0.001
FFL_LOSS_WEIGHT = 1.0

denoising_training.SAVE_MODEL = True
denoising_training.BATCH_REPORT_INTERVAL = 40
denoising_training.CUDA_DEVICE = "cuda:0"

loss_functions = {
    "MSE": MSELoss()
}

for key, loss in loss_functions.items():
    model = BaseUNet()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
    # Train the model
    denoising_training.train_denoise_image_net(
        model=model,
        optimizer=optimizer,
        prefix="Denoising_Unet_" + key,
        resize=(32, 32),
        loss_function=loss,
        epochs=6,
        batch_size=1024,
        noise_str=20
    )
