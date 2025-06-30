import os
import random
import numpy as np
import torch
import argparse
from datetime import datetime

from config_reader import load_config, save_config, config_to_yaml_string
from data import create_dataloaders
from network import create_model
from loss import create_criterion
from trainer import Trainer, create_optimizer, create_scheduler, visualize_predictions


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
    os.makedirs(output_dir, exist_ok = True)

    config_path = os.path.join(output_dir, "config.yaml")
    save_config(cfg, config_path)

    return output_dir


def main(config_path: str = "config/base.yaml") -> None:
    cfg = load_config(config_path)

    print("Configuration:")
    print(config_to_yaml_string(cfg))

    set_seed(cfg.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = setup_output_dir(cfg)
    cfg.output_dir = output_dir

    seg_train_loader, seg_test_loader, cls_train_loader, cls_test_loader = create_dataloaders(cfg)

    print(f"Segmentation - Train: {len(seg_train_loader.dataset)}, Test: {len(seg_test_loader.dataset)}")
    print(f"Classification - Train: {len(cls_train_loader.dataset)}, Test: {len(cls_test_loader.dataset)}")

    model = create_model(cfg)
    model.to(device)

    print("Multi-task Model components:")
    print(f"- Encoder: {type(model.encoder).__name__}")
    print(f"- Decoder: {type(model.decoder).__name__}")
    print(f"- Segmentation Head: {type(model.segmentation_head).__name__}")
    print(f"- Classification Head: {type(model.classification_head).__name__}")

    seg_criterion, cls_criterion = create_criterion(cfg)
    optimizer = create_optimizer(model, cfg)
    scheduler = create_scheduler(optimizer, cfg)

    trainer = Trainer(model, seg_criterion, cls_criterion, optimizer, scheduler, device, cfg)

    best_combined_score = trainer.train(seg_train_loader, seg_test_loader, cls_train_loader, cls_test_loader)

    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))
    visualize_predictions(model, seg_test_loader, cls_test_loader, device, output_dir)

    print(f"Training completed. Best combined score: {best_combined_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multi-task model")
    parser.add_argument("--config", type = str, default = "config/base.yaml",
                       help = "Path to configuration file")
    args = parser.parse_args()

    main(args.config)