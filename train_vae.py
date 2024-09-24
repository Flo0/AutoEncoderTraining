import torch
import training.fixed_vae_training as training

from vaes.hunter.basic_fixeddim_vae import BasicHunterVAE

batch_size = 64
learning_rate = 1e-3
weight_decay = 1e-2
num_epochs = 50
latent_dim = 1000
hidden_dim = 512

training.CUDA_DEVICE = "cuda:0"
training.BATCH_REPORT_INTERVAL = 20
training.SAVE_MODEL = True

model = BasicHunterVAE(input_dim=32 * 32 * 3, hidden_dim=hidden_dim, latent_dim=latent_dim)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# def train_auto_encoder_pokemon(model, optimizer, prefix, resize=(32, 32), epochs=10, batch_size=32):
training.train_auto_encoder_image_net(model, optimizer, "fixed_vae", epochs=num_epochs, batch_size=batch_size, n_classes=latent_dim)
