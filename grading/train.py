import os
import random
import numpy as np
import torch
import wandb
import argparse
from datetime import datetime

from config_reader import load_config, save_config, config_to_yaml_string, SimpleConfig
from data import create_dataloaders
from network import create_model
from loss import create_criterion
from trainer import Trainer, create_optimizer, create_scheduler


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_output_dir(cfg):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{cfg.experiment_name}_{timestamp}"

    output_dir = os.path.join(cfg.output_base_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    config_path = os.path.join(output_dir, "config.yaml")
    save_config(cfg, config_path)

    return output_dir


def setup_wandb(cfg, output_dir):
    if cfg.logging.use_wandb:
        wandb.init(
            project=cfg.logging.project_name,
            name=cfg.experiment_name,
            config=cfg.to_dict(),
            dir=output_dir,
            reinit=True
        )


def main(config_path: str = "config/base.yaml") -> None:
    cfg = load_config(config_path)

    print("Configuration:")
    print(config_to_yaml_string(cfg))

    set_seed(cfg.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = setup_output_dir(cfg)
    cfg.output_dir = output_dir

    setup_wandb(cfg, output_dir)

    train_loader, val_loader = create_dataloaders(cfg)
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")

    model = create_model(cfg)
    model.to(device)
    print(f"Model: {cfg.model.name}")

    criterion = create_criterion(cfg)
    optimizer = create_optimizer(model, cfg)
    scheduler = create_scheduler(optimizer, cfg)

    trainer = Trainer(model, criterion, optimizer, scheduler, device, cfg)

    best_acc = trainer.train(train_loader, val_loader)

    print(f"Training completed. Best validation accuracy: {best_acc:.4f}")

    if cfg.logging.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train retinal disease grading model")
    parser.add_argument("--config", type=str, default="config/base.yaml",
                        help="Path to configuration file")
    args = parser.parse_args()

    main(args.config)