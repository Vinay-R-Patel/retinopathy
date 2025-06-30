import argparse
import os
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from config_reader import load_config
from data import MultiClassSegmentationDataset, RetinoGradingDataset, get_transforms
from network import create_model
from loss import create_loss_functions
from trainer import train_one_epoch, validate, visualize_predictions, plot_training_curves, enhanced_log_to_wandb, save_detailed_report

import wandb
WANDB_AVAILABLE = True



def main():
    parser = argparse.ArgumentParser(description='🎯 Train MixOE Multi-task Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    config = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")

    os.makedirs(config.output.save_dir, exist_ok=True)

    if WANDB_AVAILABLE and getattr(config, 'wandb', None) and config.wandb.enabled:
        wandb.init(
            project=config.wandb.project,
            entity=getattr(config.wandb, 'entity', None),
            name=getattr(config.wandb, 'name', None),
            tags=getattr(config.wandb, 'tags', []),
            notes=getattr(config.wandb, 'notes', None),
            config=dict(config)
        )
        print("📊 WandB initialized successfully")

    train_transform, val_transform = get_transforms(config)

    seg_train_dataset = MultiClassSegmentationDataset(
        config.data.segmentation.train_images_dir,
        config.data.segmentation.train_hard_exudates_dir,
        config.data.segmentation.train_haemorrhages_dir,
        train_transform
    )
    seg_test_dataset = MultiClassSegmentationDataset(
        config.data.segmentation.test_images_dir,
        config.data.segmentation.test_hard_exudates_dir,
        config.data.segmentation.test_haemorrhages_dir,
        val_transform
    )

    cls_train_dataset = RetinoGradingDataset(
        config.data.classification.train_images_dir,
        config.data.classification.train_labels_csv,
        train_transform
    )
    cls_test_dataset = RetinoGradingDataset(
        config.data.classification.test_images_dir,
        config.data.classification.test_labels_csv,
        val_transform
    )

    print(f"📊 Dataset sizes:")
    print(f"  🎯 Segmentation - Train: {len(seg_train_dataset)}, Test: {len(seg_test_dataset)}")
    print(f"  🎪 Classification - Train: {len(cls_train_dataset)}, Test: {len(cls_test_dataset)}")

    seg_train_loader = DataLoader(
        seg_train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    seg_test_loader = DataLoader(
        seg_test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    cls_train_loader = DataLoader(
        cls_train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    cls_test_loader = DataLoader(
        cls_test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )

    model = create_model(config).to(device)

    print("🏗️ MixOE Multi-task Model Architecture:")
    print(f"  • Backbone: {getattr(config.model, 'backbone_name', 'N/A')}")
    print(f"  • Number of experts: {getattr(config.model, 'n_experts', 8)}")
    print(f"  • Segmentation classes: {config.model.seg_classes}")
    print(f"  • Classification classes: {config.model.cls_classes}")
    print(f"  • Image size: {config.preprocessing.image_size}")
    print(f"  • Advanced fundus preprocessing: {getattr(config.preprocessing, 'advanced_fundus', False)}")

    criterion = create_loss_functions(config)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config.optimizer.lr_scheduler.mode,
        patience=config.optimizer.lr_scheduler.patience,
        factor=config.optimizer.lr_scheduler.factor,
        min_lr=float(getattr(config.optimizer.lr_scheduler, 'min_lr', 1e-7))
    )

    best_metrics = {
        "val_seg_dice": (0, 0),
        "val_cls_accuracy": (0, 0),
        "val_cls_macro_f1": (0, 0),
        "val_cls_cohen_kappa": (0, 0),
        "combined_score": (0, 0)
    }
    train_history = []
    val_history = []

    print(f"\n🚀 Starting training for {config.training.num_epochs} epochs...")
    print("=" * 80)

    for epoch in range(1, config.training.num_epochs + 1):
        print(f"\n📅 Epoch {epoch}/{config.training.num_epochs}")
        print("-" * 50)

        train_metrics = train_one_epoch(
            model, seg_train_loader, cls_train_loader, criterion, optimizer,
            device, epoch, config, scheduler
        )

        val_metrics, val_additional_data = validate(model, seg_test_loader, cls_test_loader, criterion, device, config)

        combined_score = val_metrics.get("val_seg_dice", 0) + val_metrics.get("val_cls_accuracy", 0)
        val_metrics["combined_score"] = combined_score

        scheduler.step(combined_score)

        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"  🚂 Train - Loss: {train_metrics.get('total_loss', 0):.4f}, "
              f"Seg Dice: {train_metrics.get('seg_dice', 0):.4f}, "
              f"Cls Acc: {train_metrics.get('cls_accuracy', 0):.4f}")
        print(f"  🔍 Val   - Seg Dice: {val_metrics.get('val_seg_dice', 0):.4f}, "
              f"Cls Acc: {val_metrics.get('val_cls_accuracy', 0):.4f}, "
              f"F1: {val_metrics.get('val_cls_macro_f1', 0):.4f}")
        print(f"  ⚖️  Task weights - Seg: {train_metrics.get('seg_weight', 0):.4f}, "
              f"Cls: {train_metrics.get('cls_weight', 0):.4f}")
        print(f"  🎯 Combined Score: {combined_score:.4f}")

        current_metrics = {
            "val_seg_dice": val_metrics.get("val_seg_dice", 0),
            "val_cls_accuracy": val_metrics.get("val_cls_accuracy", 0),
            "val_cls_macro_f1": val_metrics.get("val_cls_macro_f1", 0),
            "val_cls_cohen_kappa": val_metrics.get("val_cls_cohen_kappa", 0),
            "combined_score": combined_score
        }

        models_saved = []
        for metric_name, current_value in current_metrics.items():
            best_value, best_epoch = best_metrics[metric_name]
            if current_value > best_value:
                best_metrics[metric_name] = (current_value, epoch)
                model_path = os.path.join(config.output.save_dir, f"best_{metric_name}_model.pth")
                torch.save(model.state_dict(), model_path)
                models_saved.append(f"{metric_name}: {current_value:.4f}")

        if models_saved:
            print(f"  💾 New best models saved for: {', '.join(models_saved)}")

        if WANDB_AVAILABLE and getattr(config, 'wandb', None) and config.wandb.enabled:
            enhanced_log_to_wandb(train_metrics, epoch, "train")
            enhanced_log_to_wandb(val_metrics, epoch, "val", val_additional_data)

        train_history.append(train_metrics)
        val_history.append(val_metrics)

        if epoch % 10 == 0 or epoch == config.training.num_epochs:
            if config.output.visualize_predictions:
                viz_paths = visualize_predictions(model, seg_test_loader, cls_test_loader, device, config, epoch, config.output.save_dir)
                print(f"  🖼️  Visualizations saved: {viz_paths}")

    best_combined_score, best_epoch = best_metrics["combined_score"]
    print(f"\n🏆 Loading best model (combined score: {best_combined_score:.4f} from epoch {best_epoch})")
    model_path = os.path.join(config.output.save_dir, "best_combined_score_model.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))

    if config.output.plot_training_curves:
        plot_path = plot_training_curves(train_history, val_history, config, config.output.save_dir)
        print(f"📈 Training curves saved: {plot_path}")

    if config.output.visualize_predictions:
        final_viz_paths = visualize_predictions(model, seg_test_loader, cls_test_loader, device, config, "final", config.output.save_dir)
        print(f"🖼️ Final visualizations saved: {final_viz_paths}")

    if getattr(config.output, 'save_detailed_report', True):
        report_path = save_detailed_report(train_history, val_history, config, config.output.save_dir, best_metrics)
        print(f"📋 Detailed report saved: {report_path}")

    print(f"\n🎉 Training completed successfully!")
    print(f"🏆 Best metrics achieved:")
    for metric_name, (value, epoch) in best_metrics.items():
        print(f"  • {metric_name}: {value:.4f} (Epoch {epoch})")
    print(f"💾 Models saved in: {config.output.save_dir}")

    if WANDB_AVAILABLE and getattr(config, 'wandb', None) and config.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()