import os
from datetime import datetime

from PIL import Image
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

from datasets.datasets import AutoEncoderImageDataset
from torch.utils.data import DataLoader

BATCH_REPORT_INTERVAL = 10


def train_auto_encoder(model, optimizer, loss_fn, train_loader, val_loader, prefix, epochs=10, epoch_sample=True):
    def train_one_epoch(epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.
        last_images = None
        last_index = len(train_loader) - 1

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for train_index, data in enumerate(train_loader):
            # Every data instance is an input + label pair
            input_images = data

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            output_train_images = model(input_images)

            if train_index == last_index:
                last_images = (input_images, output_train_images)

            # Compute the loss and its gradients
            loss = loss_fn(input_images, output_train_images)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if train_index != 0 and train_index % BATCH_REPORT_INTERVAL == 0:
                last_loss = running_loss / BATCH_REPORT_INTERVAL  # loss per batch
                tb_x = epoch_index * len(train_loader) + train_index + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss, last_images

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/{}_auto_encoder_{}'.format(prefix, timestamp))
    epoch_number = 0

    best_vloss = 1_000_000.

    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss, last_images = train_one_epoch(epoch_number, writer)

        if epoch_sample:
            sample_folder = './output/samples/{}'.format(prefix)
            os.makedirs(sample_folder, exist_ok=True)

            input_images = last_images[0]
            output_images = last_images[1]

            # Save the last batch of images
            input_image = input_images[0]
            output_image = output_images[0]

            input_image = transforms.ToPILImage()(input_image)
            output_image = transforms.ToPILImage()(output_image)

            # Horizontal stack the images
            stacked_image = Image.new('RGB', (input_image.width + output_image.width, input_image.height))
            stacked_image.paste(input_image, (0, 0))
            stacked_image.paste(output_image, (input_image.width, 0))

            stacked_image.save(sample_folder + '/sample_{}.png'.format(timestamp))

            writer.add_images('Sample Images', output_images, epoch_number + 1)
            writer.flush()

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                validation_images = vdata
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
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_folder = './output/models/{}'.format(prefix)
            os.makedirs(model_folder, exist_ok=True)
            model_path = './output/models/{}/model_{}_{}'.format(prefix, timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1


def train_auto_encoder_pokemon(model, optimizer, loss_function, prefix, resize=(32, 32), epochs=10):
    preprocessing = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(resize),  # Resize the image
        transforms.ToTensor()  # Convert to a tensor with shape (C, H, W) and range [0, 1]
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to range [-1, 1]
    ])

    training_data = AutoEncoderImageDataset(image_dir="./data/pokemon/gen3-processed-rgb", load_into_memory=True, train=True, transform=preprocessing)
    test_data = AutoEncoderImageDataset(image_dir="./data/pokemon/gen3-processed-rgb", load_into_memory=True, train=False, transform=preprocessing)

    print('Training set has {} instances'.format(len(training_data)))
    print('Validation set has {} instances'.format(len(test_data)))

    train_data_loader = DataLoader(training_data, batch_size=32, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=32, shuffle=True)

    return train_auto_encoder(model, optimizer, loss_function, train_data_loader, test_data_loader, prefix, epochs=epochs)
