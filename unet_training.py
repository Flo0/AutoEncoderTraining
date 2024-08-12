from datasets import AutoEncoderImageDataset

dataset = AutoEncoderImageDataset(image_dir="./data/pokemon/gen3-processed", load_into_memory=True)
