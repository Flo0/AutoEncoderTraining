import torch
from torch.utils.data import Dataset

from ddpm.dataset import PokemonDataset, FFHQDataset
from diffusion import DiffusionModel
from trainer import Trainer
from utils import parser
from argparse import ArgumentParser, Namespace
import os
import json
from mapping import MODEL_NAME_MAPPING


def main(config_file_path: str, dataset: Dataset, model_milestone: int = None, from_old_model: bool = False) -> None:
    torch.cuda.empty_cache()

    if not os.path.isfile(config_file_path):
        raise FileNotFoundError(config_file_path)

    with open(config_file_path, "r") as f:
        config_file = json.load(f)

    unet_config = config_file.get("unet_config")
    trainer_config = config_file.get("trainer_config")
    diffusion_config = config_file.get("diffusion_config")

    unet_ = MODEL_NAME_MAPPING.get(unet_config.get("model_mapping"))

    model = unet_(
        dim=unet_config.get("input"),
        channels=unet_config.get("channels"),
        dim_mults=tuple(unet_config.get("dim_mults")),
    ).to("cuda")

    diffusion_model = DiffusionModel(
        model,
        image_size=diffusion_config.get("image_size"),
        beta_scheduler=diffusion_config.get("betas_scheduler"),
        timesteps=diffusion_config.get("timesteps"),
    )

    trainer = Trainer(
        diffusion_model=diffusion_model,
        dataset=dataset,
        results_folder=f'../ddpm_results/{config_file.get("model_name")}',
        train_batch_size=trainer_config.get("train_batch_size"),
        train_lr=trainer_config.get("train_lr"),
        train_num_steps=trainer_config.get("train_num_steps"),
        save_and_sample_every=trainer_config.get("save_and_sample_every"),
        num_samples=trainer_config.get("num_samples"),
    )

    if milestone := model_milestone:
        trainer.load(milestone, is_old_model=from_old_model)

    trainer.train()


if __name__ == "__main__":
    main("./train_model_config.json", FFHQDataset())
