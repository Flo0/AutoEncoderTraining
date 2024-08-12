from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from unet import UNet
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
import matplotlib.pyplot as plt

writer = SummaryWriter()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize for grayscale images
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = UNet().cuda()  # or .to(device) if using GPU

criterion = nn.MSELoss()  # or other loss suitable for your task
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10

for epoch in range(num_epochs):
    for data in train_loader:
        img, _ = data
        img = img.cuda()  # or img.to(device) if using GPU

        # Forward pass
        output = model(img)
        loss = criterion(output, img)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    for data in train_loader:
        img, _ = data
        img = img.cuda()  # or img.to(device)
        output = model(img)
        break

# Compare original and reconstructed images
img = img.cpu().numpy()
output = output.cpu().numpy()

# Plot original and reconstructed images
fig, axes = plt.subplots(1, 2)
axes[0].imshow(img[0][0], cmap='gray')
axes[0].set_title("Original")
axes[1].imshow(output[0][0], cmap='gray')
axes[1].set_title("Reconstructed")
plt.show()
