import torch
from focal_frequency_loss import FocalFrequencyLoss as FFL

from auto_encoder_training import train_auto_encoder_pokemon
from auto_encoders.custom.primitive import PrimitiveAutoEncoder

LEARN_RATE = 0.001
FFL_LOSS_WEIGHT = 1.0

# Model is Primitive AutoEncoder
model = PrimitiveAutoEncoder(in_out_channels=(3, 3), scaling=0.5, keep_dim=True)

# Optimizer is Adam
optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)

# Loss function is Focal Frequency Loss
ffl_loss = FFL(loss_weight=FFL_LOSS_WEIGHT, alpha=1.0)

# Train the model
train_auto_encoder_pokemon(model=model, optimizer=optimizer, loss_function=ffl_loss, epochs=2)
