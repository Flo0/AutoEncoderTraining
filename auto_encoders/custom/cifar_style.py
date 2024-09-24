import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        """
        Fully Connected Neural Network Encoder for VAE.

        Args:
            input_dim (int): Dimension of the input data.
            hidden_dim (int): Dimension of the hidden layer.
            latent_dim (int): Dimension of the latent space (z).
        """
        super(FCNNEncoder, self).__init__()
        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # Output mean (mu)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # Output log(sigma^2)

    def forward(self, x):
        # Forward pass through first hidden layer with activation
        h = F.relu(self.fc1(x))
        # Output mean and log variance for latent space
        mu = self.fc_mu(h)
        log_sigma2 = self.fc_logvar(h)
        return mu, log_sigma2


class FCNNDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        """
        Fully Connected Neural Network Decoder for VAE.

        Args:
            latent_dim (int): Dimension of the latent space (z).
            hidden_dim (int): Dimension of the hidden layer.
            output_dim (int): Dimension of the output data (reconstructed input).
        """
        super(FCNNDecoder, self).__init__()
        # Fully connected layers
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        # Forward pass through hidden layer with activation
        h = F.relu(self.fc1(z))
        # Output reconstructed data
        x_reconstructed = torch.sigmoid(self.fc2(h))  # Sigmoid for normalized output
        return x_reconstructed
