import os
from datetime import datetime
from multiprocessing import Process

from PIL import Image, ImageFont, ImageDraw
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from tqdm import tqdm
from util import image_utils

from datasets.datasets import AutoEncoderImageDataset, AutoEncoderNPZDataset
from torch.utils.data import DataLoader

BATCH_REPORT_INTERVAL = 20
CUDA_DEVICE = "cuda:0"
SAVE_MODEL = True


def train_auto_encoder(model, optimizer, loss_fn, train_loader, val_loader, prefix, epochs=10, epoch_sample=True, samples=None):
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    device = torch.device(CUDA_DEVICE if torch.cuda.is_available() else "cpu")
    print('Using device:', device)
    # model = nn.DataParallel(model)
    model.to(device)

    if epoch_sample and samples is None:
        raise ValueError('Samples must be provided if epoch_sample is True')

    def train_one_epoch(epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.
        last_index = len(train_loader) - 1
        batch_counter = 0

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for train_index, data in tqdm(enumerate(train_loader), "Batches", total=len(train_loader)):
            # Every data instance is an input + label pair
            input_images = data.to(device)

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            output_train_images = model(input_images)

            # Compute the loss and its gradients
            loss = loss_fn(input_images, output_train_images)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            batch_counter += 1
            if batch_counter == BATCH_REPORT_INTERVAL or train_index == last_index:
                last_loss = running_loss / batch_counter  # loss per batch
                tb_x = epoch_index * len(train_loader) + train_index + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.
                batch_counter = 0

        return last_loss

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/{}_auto_encoder_{}'.format(prefix, timestamp))
    epoch_number = 0

    best_vloss = 1_000_000.

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer)

        if epoch_sample:
            sample_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            sample_folder = './output/samples/{}'.format(prefix)
            os.makedirs(sample_folder, exist_ok=True)

            save_grid = True if epoch == epochs - 1 else False

            image_path = sample_folder + '/sample_{}_E{}.png'.format(sample_timestamp, (epoch + 1))
            grid, output_images = create_and_save_samples(model, samples, device, image_path, to_disk=save_grid)

            writer.add_images('Sample Images', output_images, epoch_number + 1)
            writer.flush()

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                validation_images = vdata.to(device)
                output_val_images = model(validation_images)
                vloss = loss_fn(validation_images, output_val_images)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch_number + 1)

        writer.flush()

        # Track the best performance, and save the model's state
        if SAVE_MODEL and avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_folder = './output/models/{}'.format(prefix)
            os.makedirs(model_folder, exist_ok=True)
            model_path = './output/models/{}/model_{}_{}'.format(prefix, timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1


def train_auto_encoder_pokemon(model, optimizer, loss_function, prefix, resize=(32, 32), epochs=10, batch_size=32):
    preprocessing = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(resize),  # Resize the image
        transforms.ToTensor()  # Convert to a tensor with shape (C, H, W) and range [0, 1]
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to range [-1, 1]
    ])

    training_data = AutoEncoderImageDataset(image_dir="./data/pokemon/gen3-processed-rgb", load_into_memory=True, train=True, transform=preprocessing)
    test_data = AutoEncoderImageDataset(image_dir="./data/pokemon/gen3-processed-rgb", load_into_memory=True, train=False, transform=preprocessing)

    sample_data = AutoEncoderImageDataset(image_dir="./data/pokemon/gen3-processed-rgb", load_into_memory=True, train=True, transform=preprocessing,
                                          cut=0.0)

    samples = torch.stack([sample_data[i] for i in range(9)], dim=0)

    print('Training set has {} instances'.format(len(training_data)))
    print('Validation set has {} instances'.format(len(test_data)))

    train_data_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_auto_encoder(model, optimizer, loss_function, train_data_loader, test_data_loader, prefix, epochs=epochs, samples=samples)


def train_auto_encoder_image_net(model, optimizer, loss_function, prefix, resize=(32, 32), epochs=10, batch_size=32):
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(resize)
    ])

    train_data = AutoEncoderNPZDataset("./data/external/image_net/train", preprocessing)
    val_data = AutoEncoderNPZDataset("./data/external/image_net/valid", preprocessing)

    sample_data = AutoEncoderNPZDataset("./data/external/image_net/valid", preprocessing)

    samples = torch.stack([sample_data[i] for i in range(9)], dim=0)

    print('Training set has {} instances'.format(len(train_data)))
    print('Validation set has {} instances'.format(len(val_data)))

    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    return train_auto_encoder(model, optimizer, loss_function, train_data_loader, test_data_loader, prefix, epochs=epochs, samples=samples)


def create_and_save_samples(model, samples, device, output_path, to_disk=True):
    input_images = samples.to(device)
    with torch.no_grad():
        output_images = model(input_images)

    input_images = input_images.cpu()
    output_images = output_images.cpu()

    # Create a grid with PIL
    n_images = len(input_images)

    pairs_per_row = 3
    row_count = n_images // pairs_per_row

    grid_width = pairs_per_row
    grid_height = row_count

    # Calculate the grid size
    channels, single_image_width, single_image_height = output_images[0].size()
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

        pil_input_image = pil_transform(input_images[i])
        pil_output_image = pil_transform(output_images[i])
        # Original image
        grid_image.paste(pil_input_image, (start_x, start_y))
        # Reconstructed image
        grid_image.paste(pil_output_image, (start_x + single_image_width, start_y))

    if to_disk:
        grid_image.save(output_path)
    return grid_image, output_images
