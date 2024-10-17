import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader

batch_size = 128
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Lambda(lambda x: x.view(-1) - 0.5),
])

# Download and load the training data
train_data = datasets.MNIST(
    '../../data/external/mnist',
    download=True,
    train=True,
    transform=transform,
)
# Download and load the test data
test_data = datasets.MNIST(
    '../../data/external/mnist',
    download=True,
    train=False,
    transform=transform,
)

# Create data loaders
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False,
)


import torch
import torch.optim as optim
from torch.nn import functional as F

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, log_sigma2):
    # Reconstruction loss (binary cross-entropy)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL divergence term
    kl_loss = -0.5 * torch.sum(1 + log_sigma2 - mu.pow(2) - log_sigma2.exp())

    return recon_loss + kl_loss

# Training function for one epoch
def train(model, train_loader, optimizer, device):
    model.train()  # Set model to training mode
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)

        optimizer.zero_grad()  # Zero the gradients

        # Forward pass through the model
        mu, log_sigma2 = model.encode(data)
        z = model.reparameterize(mu, log_sigma2)  # Sample from latent space
        recon_batch = model.decode(z)

        # Compute loss
        loss = loss_function(recon_batch, data, mu, log_sigma2)
        loss.backward()  # Backpropagate

        train_loss += loss.item()
        optimizer.step()  # Update model parameters

        if batch_idx % 100 == 0:
            print(f'Train Epoch: [{batch_idx}/{len(train_loader)}]\tLoss: {loss.item() / len(data):.6f}')

    print(f'====> Average loss: {train_loss / len(train_loader.dataset):.4f}')

# Testing function to evaluate the model on test data
def test(model, test_loader, device):
    model.eval()  # Set model to evaluation mode
    test_loss = 0

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)

            # Forward pass through the model
            mu, log_sigma2 = model.encode(data)
            z = model.reparameterize(mu, log_sigma2)  # Sample from latent space
            recon_batch = model.decode(z)

            # Compute loss
            test_loss += loss_function(recon_batch, data, mu, log_sigma2).item()

    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')

# Initialize the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model, optimizer, and training parameters
latent_channels = 16  # Change this to any number of latent channels you want
model = FCNNVAE(input_channels=1, latent_channels=latent_channels).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Number of epochs to train the model
epochs = 10

# Training loop
for epoch in range(1, epochs + 1):
    print(f"Epoch {epoch}")
    train(model, train_loader, optimizer, device)
    test(model, test_loader, device)

# Save the trained model
torch.save(model.state_dict(), 'vae_mnist.pth')
