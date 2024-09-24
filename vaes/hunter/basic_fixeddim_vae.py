import torch.nn as nn
from vaes.hunter.base_vae import VariationalAutoEncoder


class HunterEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(HunterEncoder, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 8, 2 * latent_dim),  # 2 for mean and variance.
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


class HunterDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(HunterDecoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 8),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 8, hidden_dim // 4),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),  # Swish activation function
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.decoder(z)


class BasicHunterVAE(VariationalAutoEncoder):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        output_dim = input_dim
        super(BasicHunterVAE, self).__init__(HunterEncoder(input_dim, hidden_dim, latent_dim), HunterDecoder(latent_dim, hidden_dim, output_dim))
