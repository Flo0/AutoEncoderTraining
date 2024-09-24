import random
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from vaes.explorative.ex_vae import VAE
from vaes.explorative.ex_data_prep import train_loader, test_loader
from tqdm import tqdm

learning_rate = 1e-3
weight_decay = 1e-2
num_epochs = 65
hidden_dim = 512
batch_size = 128


def train(model, dataloader, optimizer, prev_updates, writer=None):
    """
    Trains the model on the given data.

    Args:
        model (nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): The data loader.
        loss_fn: The loss function.
        optimizer: The optimizer.
    """
    model.train()  # Set the model to training mode

    for batch_idx, (data, target) in enumerate(tqdm(dataloader)):
        n_upd = prev_updates + batch_idx

        data = data.to(device)

        optimizer.zero_grad()  # Zero the gradients

        output = model(data)  # Forward pass
        loss = output.loss

        loss.backward()

        if n_upd % 100 == 0:
            # Calculate and log gradient norms
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)

            # print(f'Step {n_upd:,} (N samples: {n_upd * batch_size:,}), Loss: {loss.item():.4f} (Recon: {output.loss_recon.item():.4f}, KL: {output.loss_kl.item():.4f}) Grad: {total_norm:.4f}')

            if writer is not None:
                global_step = n_upd
                writer.add_scalar('Loss/Train', loss.item(), global_step)
                writer.add_scalar('Loss/Train/BCE', output.loss_recon.item(), global_step)
                writer.add_scalar('Loss/Train/KLD', output.loss_kl.item(), global_step)
                writer.add_scalar('GradNorm/Train', total_norm, global_step)

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()  # Update the model parameters

    return prev_updates + len(dataloader)


def test(model, dataloader, cur_step, writer=None):
    """
    Tests the model on the given data.

    Args:
        model (nn.Module): The model to test.
        dataloader (torch.utils.data.DataLoader): The data loader.
        cur_step (int): The current step.
        writer: The TensorBoard writer.
    """
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    test_recon_loss = 0
    test_kl_loss = 0

    with torch.no_grad():
        for data, target in tqdm(dataloader, desc='Testing'):
            data = data.to(device)
            data = data.view(data.size(0), -1)  # Flatten the data

            output = model(data, compute_loss=True)  # Forward pass

            test_loss += output.loss.item()
            test_recon_loss += output.loss_recon.item()
            test_kl_loss += output.loss_kl.item()

    test_loss /= len(dataloader)
    test_recon_loss /= len(dataloader)
    test_kl_loss /= len(dataloader)
    print(f'====> Test set loss: {test_loss:.4f} (BCE: {test_recon_loss:.4f}, KLD: {test_kl_loss:.4f})')

    if writer is not None:
        writer.add_scalar('Loss/Test', test_loss, global_step=cur_step)
        writer.add_scalar('Loss/Test/BCE', output.loss_recon.item(), global_step=cur_step)
        writer.add_scalar('Loss/Test/KLD', output.loss_kl.item(), global_step=cur_step)

        # Log reconstructions
        writer.add_images('Test/Reconstructions', output.x_recon.view(-1, 1, 28, 28), global_step=cur_step)
        writer.add_images('Test/Originals', data.view(-1, 1, 28, 28), global_step=cur_step)


def set_seed(seed):
    # Set the Python random seed
    random.seed(seed)

    # Set the NumPy random seed
    np.random.seed(seed)

    # Set the PyTorch random seed for both CPU and CUDA
    torch.manual_seed(seed)

    # If using GPUs, make the CUDA behavior deterministic
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU environments

    # Ensures that CUDA operations are deterministic
    torch.backends.cudnn.deterministic = True

    # Disable the benchmark mode to ensure reproducibility (may slow down performance)
    torch.backends.cudnn.benchmark = False


set_seed(1000)

for beta in [0.1, 2, 10, 50]:
    for ldim in [2, 3, 4, 16]:
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        model = VAE(input_dim=784, hidden_dim=hidden_dim, latent_dim=ldim, beta=beta).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        writer = SummaryWriter(f'../../runs/mnist/vae_ldim_{ldim}_beta_{beta}_{datetime.now().strftime("%Y%m%d-%H%M%S")}')

        prev_updates = 0
        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs} (ldim={ldim}) (beta={beta})')
            prev_updates = train(model, train_loader, optimizer, prev_updates, writer=writer)
            test(model, test_loader, prev_updates, writer=writer)

        writer.close()

        torch.save(model.state_dict(), f'../../vae_mnist_ldim_{ldim}_beta_{beta}.pth')
