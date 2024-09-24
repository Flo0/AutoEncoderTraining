import os
from datetime import datetime

from PIL import Image
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from tqdm import tqdm

from datasets.datasets import VAENPZDataset, VAEImageDataset
from torch.utils.data import DataLoader
from vaes.hunter.base_vae import VAEResult

import torchvision.transforms.v2 as v2

BATCH_REPORT_INTERVAL = 20
CUDA_DEVICE = "cuda:0"
SAVE_MODEL = True


def train_variational_auto_encoder(model, optimizer, train_loader, val_loader, prefix, epochs=10, epoch_sample=True, samples=None):
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    device = torch.device(CUDA_DEVICE if torch.cuda.is_available() else "cpu")
    print('Using device:', device)
    model.to(device)

    if epoch_sample and samples is None:
        raise ValueError('Samples must be provided if epoch_sample is True')

    def train_one_epoch(epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.
        last_index = len(train_loader) - 1
        batch_counter = 0

        for train_index, data in tqdm(enumerate(train_loader), "Batches", total=len(train_loader)):
            input_images = data.to(device)
            optimizer.zero_grad()
            vae_result: VAEResult = model(input_images)
            loss = vae_result.loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_counter += 1
            if batch_counter == BATCH_REPORT_INTERVAL or train_index == last_index:
                last_loss = running_loss / batch_counter
                tb_x = epoch_index * len(train_loader) + train_index + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.
                batch_counter = 0

        return last_loss

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/{}_variational_auto_encoder_{}'.format(prefix, timestamp))
    epoch_number = 0

    best_vloss = 1_000_000.

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer)

        if epoch_sample:
            sample_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            sample_folder = './output/samples/{}'.format(prefix)
            os.makedirs(sample_folder, exist_ok=True)

            save_grid = True if epoch == epochs - 1 else False

            image_path = sample_folder + '/sample_{}_E{}.png'.format(sample_timestamp, (epoch + 1))
            grid, output_images = create_and_save_samples(model, samples, device, image_path, to_disk=save_grid)

            # Change to add a channel dimension for RGB images
            output_images = output_images.view(-1, 3, 32, 32)

            writer.add_images('Sample Images', output_images, epoch_number + 1)
            writer.flush()

        running_vloss = 0.0
        model.eval()

        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                validation_images = vdata.to(device)
                val_output: VAEResult = model(validation_images)
                vloss = val_output.loss
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch_number + 1)

        writer.flush()

        if SAVE_MODEL and avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_folder = './output/models/{}'.format(prefix)
            os.makedirs(model_folder, exist_ok=True)
            model_path = './output/models/{}/model_{}_{}'.format(prefix, timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1


# Redefine the preprocessing for RGB
def train_auto_encoder_pokemon(model, optimizer, prefix, resize=(32, 32), epochs=10, batch_size=32):
    preprocessing = v2.Compose([
        v2.ToImage(),  # Converts the input to an image format
        v2.Resize(resize),  # Resizes the image to 32x32 if needed
        v2.ConvertImageDtype(torch.float32),  # Converts to float32
        v2.Lambda(lambda x: x.reshape(-1))  # Flattens the RGB image using reshape
    ])

    training_data = VAEImageDataset(image_dir="./data/pokemon/gen3-processed-rgb", transform=preprocessing)
    test_data = VAEImageDataset(image_dir="./data/pokemon/gen3-processed-rgb", transform=preprocessing)

    sample_data = VAEImageDataset(image_dir="./data/pokemon/gen3-processed-rgb", transform=preprocessing)

    samples = torch.stack([sample_data[i] for i in range(9)], dim=0)

    print("Shape of samples: ", samples[0].shape)

    print('Training set has {} instances'.format(len(training_data)))
    print('Validation set has {} instances'.format(len(test_data)))

    train_data_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_variational_auto_encoder(model, optimizer, train_data_loader, test_data_loader, prefix, epochs=epochs,
                                          samples=samples)


def train_auto_encoder_image_net(model, optimizer, prefix, resize=(32, 32), epochs=10, batch_size=32, n_classes=None):
    preprocessing = v2.Compose([
        v2.ToImage(),  # Converts the input to an image format
        v2.Resize(resize),  # Resizes the image to 32x32 if needed
        v2.ConvertImageDtype(torch.float32),  # Converts to float32
        v2.Lambda(lambda x: x.reshape(-1))  # Flattens the RGB image using reshape
    ])

    train_data = VAENPZDataset("./data/external/image_net/train", preprocessing, n_biggest_classes=n_classes)
    val_data = VAENPZDataset("./data/external/image_net/valid", preprocessing, n_biggest_classes=n_classes)

    sample_data = VAENPZDataset("./data/external/image_net/valid", preprocessing, n_biggest_classes=n_classes)

    samples = torch.stack([sample_data[i] for i in range(9)], dim=0)

    print("Shape of samples: ", samples.shape)

    print('Training set has {} instances'.format(len(train_data)))
    print('Validation set has {} instances'.format(len(val_data)))

    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    return train_variational_auto_encoder(model, optimizer, train_data_loader, test_data_loader, prefix, epochs=epochs,
                                          samples=samples)


def create_and_save_samples(model, samples, device, output_path, to_disk=True):
    input_images = samples.to(device)
    with torch.no_grad():
        vae_output: VAEResult = model(input_images)

    input_images = input_images.cpu()
    output_images = vae_output.x_reconstructed.cpu()

    # Reshape from flattened vectors to 32x32x3 RGB images
    input_images = input_images.view(-1, 3, 32, 32)  # Reshape to (9, 3, 32, 32)
    output_images = output_images.view(-1, 3, 32, 32)  # Reshape to (9, 3, 32, 32)

    # Create a grid with PIL, considering RGB images
    n_images = len(input_images)
    pairs_per_row = 3
    row_count = n_images // pairs_per_row
    grid_width = pairs_per_row
    grid_height = row_count
    single_image_width, single_image_height = 32, 32  # Since the images are 32x32
    padding_pixels = 4

    grid_image = Image.new('RGB', (
        (single_image_width * 2) * grid_width + padding_pixels * (grid_width + 1),
        single_image_height * row_count + padding_pixels * (row_count + 1)
    ), (128, 128, 128))

    pil_transform = transforms.ToPILImage()

    for i in range(n_images):
        row = i // pairs_per_row
        col = i % pairs_per_row
        start_x = col * (single_image_width * 2) + padding_pixels * (col + 1)
        start_y = row * single_image_height + padding_pixels * (row + 1)

        # Convert RGB images to PIL
        pil_input_image = pil_transform(input_images[i])
        pil_output_image = pil_transform(output_images[i])

        # Original image
        grid_image.paste(pil_input_image, (start_x, start_y))
        # Reconstructed image
        grid_image.paste(pil_output_image, (start_x + single_image_width, start_y))

    if to_disk:
        grid_image.save(output_path)
    return grid_image, output_images
