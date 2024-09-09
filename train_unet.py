from training.auto_encoder_training import *
from auto_encoders.unet.unet import *
from training.loss_functions import *

LEARN_RATE = 0.001
FFL_LOSS_WEIGHT = 1.0

# Model is Primitive AutoEncoder
model = BaseUNet()

# Optimizer is Adam
optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)

# Loss function is Focal Frequency Loss
loss = FFLLoss()

# Train the model
train_auto_encoder_image_net(model=model, optimizer=optimizer, prefix="UNet_1.0", resize=(32, 32), loss_function=loss, epochs=5)
