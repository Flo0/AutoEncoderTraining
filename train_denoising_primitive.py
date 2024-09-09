import training.denoising_training as denoising_training
from auto_encoders.custom.cifar_style import ConvAutoEncoder
from training.loss_functions import *

LEARN_RATE = 0.001
FFL_LOSS_WEIGHT = 1.0

denoising_training.SAVE_MODEL = True
denoising_training.BATCH_REPORT_INTERVAL = 40
denoising_training.CUDA_DEVICE = "cuda:1"

loss_functions = {
    "MSE": MSELoss(),
    "FFL": FFLLoss()
}

for key, loss in loss_functions.items():
    # Model is Primitive AutoEncoder
    model = ConvAutoEncoder(keep_dim=False, scale=2.0)

    # Optimizer is Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)

    # Train the model
    denoising_training.train_denoise_image_net(
        model=model,
        optimizer=optimizer,
        prefix="Denoising_BASIC_" + str(2.0) + key,
        resize=(32, 32),
        loss_function=loss,
        epochs=16,
        batch_size=5120,
        noise_str=20
    )
