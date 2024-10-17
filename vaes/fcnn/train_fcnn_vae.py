import torch
from torchvision import transforms
import torch.optim as optim
from torch.nn import functional as F
from fcnn_vae import VanillaVAE as VAE
from datasets.datasets import AutoEncoderNPZDataset
from ddpm.dataset import FFHQDataset
from torchvision.utils import save_image
import os

batch_size = 64
sample_dir = './vae_samples'
os.makedirs(sample_dir, exist_ok=True)  # Directory to save sample images

#preprocessing = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Resize((64, 64))  # Resizing to 64x64 for input consistency
#])

# train_data = AutoEncoderNPZDataset("../../data/external/image_net/train", preprocessing)
# test_data = AutoEncoderNPZDataset("../../data/external/image_net/valid", preprocessing)

train_data = FFHQDataset(image_count_limit=70000, folder="../../data/ffhq")
test_data = FFHQDataset(image_count_limit=1000, folder="../../data/ffhq")

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


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, log_sigma2):
    # Reconstruction loss (mean squared error)
    recon_loss = F.mse_loss(recon_x, x)

    # KL divergence term
    kl_loss = -0.5 * torch.sum(1 + log_sigma2 - mu.pow(2) - log_sigma2.exp())

    return recon_loss + kl_loss


# Training function for one epoch
def train(model, train_loader, optimizer, device):
    model.train()  # Set model to training mode
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
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
        for data in test_loader:
            data = data.to(device)

            # Forward pass through the model
            mu, log_sigma2 = model.encode(data)
            z = model.reparameterize(mu, log_sigma2)  # Sample from latent space
            recon_batch = model.decode(z)

            # Compute loss
            test_loss += loss_function(recon_batch, data, mu, log_sigma2).item()

    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')


# Function to sample and save images from the VAE
def sample_images(model, epoch, device, latent_dim, suffix, num_samples=1):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Sample from a normal distribution in the latent space
        z = torch.randn(num_samples, latent_dim).to(device)

        # Decode the sampled latent vector to generate images
        generated_images = model.decode(z)

        # Rescale the images from [-1, 1] to [0, 1] (for saving purposes)
        generated_images = (generated_images + 1) / 2

        # Save images to the specified directory
        save_image(generated_images, os.path.join(sample_dir, f'sample_epoch_{epoch}{suffix}.png'), nrow=8)
        print(f'Sampled images saved for epoch {epoch}.')


# Initialize the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model, optimizer, and training parameters
latent_dim = 8  # Number of latent dimensions
model = VAE(in_channels=3, latent_dim=latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Number of epochs to train the model
epochs = 100

# Training loop
for epoch in range(1, epochs + 1):
    print(f"Epoch {epoch}")
    train(model, train_loader, optimizer, device)
    test(model, test_loader, device)

    # Sample and save images after each epoch
    sample_images(model, epoch, device, latent_dim, "_A")
    sample_images(model, epoch, device, latent_dim, "_B")
    sample_images(model, epoch, device, latent_dim, "_C")

# Save the trained model
torch.save(model.state_dict(), 'vae_model.pth')
