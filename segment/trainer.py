import os
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

from loss import multiclass_dice_score, calculate_class_wise_metrics

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Logging will be disabled.")


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    epoch_metrics = {}

    for images, masks in tqdm(dataloader, desc="Training"):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        batch_metrics = calculate_class_wise_metrics(outputs, masks)

        for key, value in batch_metrics.items():
            if isinstance(value, (int, float)):
                if key not in epoch_metrics:
                    epoch_metrics[key] = 0
                epoch_metrics[key] += value

    avg_loss = total_loss / len(dataloader)
    for key in epoch_metrics:
        epoch_metrics[key] /= len(dataloader)

    return avg_loss, epoch_metrics


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    epoch_metrics = {}

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()

            batch_metrics = calculate_class_wise_metrics(outputs, masks)

            for key, value in batch_metrics.items():
                if isinstance(value, (int, float)):
                    if key not in epoch_metrics:
                        epoch_metrics[key] = 0
                    epoch_metrics[key] += value

    avg_loss = total_loss / len(dataloader)
    for key in epoch_metrics:
        epoch_metrics[key] /= len(dataloader)

    return avg_loss, epoch_metrics


def visualize_predictions(model, dataloader, device, output_dir, num_samples=4):
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

    class_colors = {
        0: [0, 0, 0],
        1: [255, 0, 0],
        2: [0, 255, 0]
    }

    with torch.no_grad():
        for i, (images, masks) in enumerate(dataloader):
            if i >= num_samples:
                break

            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            pred_mask = torch.softmax(outputs[0], dim=0).cpu().numpy()
            pred_class = np.argmax(pred_mask, axis=0)

            pred_colored = np.zeros((pred_class.shape[0], pred_class.shape[1], 3), dtype=np.uint8)
            for class_id, color in class_colors.items():
                pred_colored[pred_class == class_id] = color

            true_mask = masks[0].cpu().numpy()
            true_colored = np.zeros((true_mask.shape[0], true_mask.shape[1], 3), dtype=np.uint8)
            for class_id, color in class_colors.items():
                true_colored[true_mask == class_id] = color

            image = images[0].cpu().numpy().transpose(1, 2, 0)
            image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            image = np.clip(image, 0, 1)

            axes[i, 0].imshow(image)
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(true_colored)
            axes[i, 1].set_title('True Mask (Red: HE, Green: EX)')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred_colored)
            axes[i, 2].set_title('Predicted Mask (Red: HE, Green: EX)')
            axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_optimizer(model, cfg):
    if cfg.optimizer.name == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay
        )
    return optimizer


def create_scheduler(optimizer, cfg):
    if cfg.scheduler.name == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=cfg.scheduler.patience,
            factor=cfg.scheduler.factor
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    return scheduler


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device, cfg):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.cfg = cfg
        self.train_losses = []
        self.val_losses = []
        self.train_metrics_history = []
        self.val_metrics_history = []

        self.use_wandb = cfg.logging.use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=cfg.logging.project_name,
                name=f"{cfg.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "model": cfg.model.name,
                    "encoder_weights": cfg.model.encoder_weights,
                    "num_classes": cfg.model.num_classes,
                    "image_size": cfg.image_size,
                    "num_epochs": cfg.training.num_epochs,
                    "batch_size": cfg.training.batch_size,
                    "learning_rate": cfg.optimizer.lr,
                    "weight_decay": cfg.optimizer.weight_decay,
                    "optimizer": cfg.optimizer.name,
                    "scheduler": cfg.scheduler.name,
                    "loss": cfg.loss.name,
                    "augmentation": cfg.augmentation.to_dict()
                }
            )

            wandb.watch(self.model, log="all", log_freq=100)

    def train(self, train_loader, val_loader):
        best_dice = 0

        for epoch in range(self.cfg.training.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.cfg.training.num_epochs}")

            train_loss, train_metrics = train_epoch(
                self.model, train_loader, self.criterion, self.optimizer, self.device
            )
            val_loss, val_metrics = validate_epoch(
                self.model, val_loader, self.criterion, self.device
            )

            self.train_losses.append(train_loss)
            self.train_metrics_history.append(train_metrics)
            self.val_losses.append(val_loss)
            self.val_metrics_history.append(val_metrics)

            current_lr = self.optimizer.param_groups[0]['lr']

            self.scheduler.step(val_loss)

            print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_metrics['dice_mean']:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_metrics['dice_mean']:.4f}")
            print(f"Val Dice (FG): {val_metrics['dice_mean_fg']:.4f}, Val IoU (FG): {val_metrics['iou_mean_fg']:.4f}")
            print(f"Learning Rate: {current_lr:.2e}")

            print("Class-wise Validation Metrics:")
            for class_name in ['background', 'hard_exudates', 'haemorrhages']:
                dice = val_metrics[f'dice_{class_name}']
                iou = val_metrics[f'iou_{class_name}']
                precision = val_metrics[f'precision_{class_name}']
                recall = val_metrics[f'recall_{class_name}']
                print(f"  {class_name}: Dice={dice:.4f}, IoU={iou:.4f}, Prec={precision:.4f}, Rec={recall:.4f}")

            if self.use_wandb:
                wandb_log = {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "learning_rate": current_lr,
                    "best_dice": best_dice
                }

                for key, value in train_metrics.items():
                    wandb_log[f"train/{key}"] = value

                for key, value in val_metrics.items():
                    wandb_log[f"val/{key}"] = value

                wandb.log(wandb_log)

            current_dice = val_metrics['dice_mean']
            if current_dice > best_dice:
                best_dice = current_dice
                model_path = os.path.join(self.cfg.output_dir, 'best_model.pth')
                torch.save(self.model.state_dict(), model_path)
                print(f"New best model saved with Dice: {best_dice:.4f}")

            if (epoch + 1) % 10 == 0:
                self.log_predictions(val_loader, epoch + 1)

        self.save_training_plots()

        if self.use_wandb:
            self.log_training_curves()
            wandb.finish()

        return best_dice

    def log_predictions(self, val_loader, epoch):
        """Log prediction visualizations to wandb"""
        if not self.use_wandb:
            return

        self.model.eval()
        images_to_log = []

        class_colors = {
            0: [0, 0, 0],
            1: [255, 0, 0],
            2: [0, 255, 0]
        }

        with torch.no_grad():
            for i, (images, masks) in enumerate(val_loader):
                if i >= 4:
                    break

                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)

                pred_mask = torch.softmax(outputs[0], dim=0).cpu().numpy()
                pred_class = np.argmax(pred_mask, axis=0)

                pred_colored = np.zeros((pred_class.shape[0], pred_class.shape[1], 3), dtype=np.uint8)
                for class_id, color in class_colors.items():
                    pred_colored[pred_class == class_id] = color

                true_mask = masks[0].cpu().numpy()
                true_colored = np.zeros((true_mask.shape[0], true_mask.shape[1], 3), dtype=np.uint8)
                for class_id, color in class_colors.items():
                    true_colored[true_mask == class_id] = color

                image = images[0].cpu().numpy().transpose(1, 2, 0)
                image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                image = np.clip(image, 0, 1)

                wandb_image = wandb.Image(
                    image,
                    masks={
                        "ground_truth": {
                            "mask_data": true_mask,
                            "class_labels": {0: "background", 1: "hard_exudates", 2: "haemorrhages"}
                        },
                        "prediction": {
                            "mask_data": pred_class,
                            "class_labels": {0: "background", 1: "hard_exudates", 2: "haemorrhages"}
                        }
                    },
                    caption=f"Sample {i+1} - Epoch {epoch}"
                )
                images_to_log.append(wandb_image)

        wandb.log({"predictions": images_to_log}, step=epoch)

    def log_training_curves(self):
        """Log training curves to wandb"""
        if not self.use_wandb:
            return

        train_dices = [metrics['dice_mean'] for metrics in self.train_metrics_history]
        val_dices = [metrics['dice_mean'] for metrics in self.val_metrics_history]
        val_dices_fg = [metrics['dice_mean_fg'] for metrics in self.val_metrics_history]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()

        axes[0, 1].plot(train_dices, label='Train Dice')
        axes[0, 1].plot(val_dices, label='Val Dice')
        axes[0, 1].plot(val_dices_fg, label='Val Dice (FG)')
        axes[0, 1].set_title('Dice Scores')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].legend()

        class_names = ['background', 'hard_exudates', 'haemorrhages']
        colors = ['black', 'red', 'green']
        for i, (class_name, color) in enumerate(zip(class_names, colors)):
            val_dice_class = [metrics[f'dice_{class_name}'] for metrics in self.val_metrics_history]
            axes[0, 2].plot(val_dice_class, label=f'{class_name}', color=color)
        axes[0, 2].set_title('Class-wise Dice Scores')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Dice Score')
        axes[0, 2].legend()

        for i, (class_name, color) in enumerate(zip(class_names, colors)):
            val_iou_class = [metrics[f'iou_{class_name}'] for metrics in self.val_metrics_history]
            axes[1, 0].plot(val_iou_class, label=f'{class_name}', color=color)
        axes[1, 0].set_title('Class-wise IoU Scores')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IoU Score')
        axes[1, 0].legend()

        for i, (class_name, color) in enumerate(zip(class_names, colors)):
            val_prec_class = [metrics[f'precision_{class_name}'] for metrics in self.val_metrics_history]
            axes[1, 1].plot(val_prec_class, label=f'{class_name}', color=color)
        axes[1, 1].set_title('Class-wise Precision')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].legend()

        for i, (class_name, color) in enumerate(zip(class_names, colors)):
            val_rec_class = [metrics[f'recall_{class_name}'] for metrics in self.val_metrics_history]
            axes[1, 2].plot(val_rec_class, label=f'{class_name}', color=color)
        axes[1, 2].set_title('Class-wise Recall')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Recall')
        axes[1, 2].legend()

        plt.tight_layout()

        wandb.log({"training_curves": wandb.Image(plt)})
        plt.close()

    def save_training_plots(self):
        train_dices = [metrics['dice_mean'] for metrics in self.train_metrics_history]
        val_dices = [metrics['dice_mean'] for metrics in self.val_metrics_history]
        val_dices_fg = [metrics['dice_mean_fg'] for metrics in self.val_metrics_history]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()

        axes[0, 1].plot(train_dices, label='Train Dice')
        axes[0, 1].plot(val_dices, label='Val Dice')
        axes[0, 1].plot(val_dices_fg, label='Val Dice (FG)')
        axes[0, 1].set_title('Dice Scores')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].legend()

        class_names = ['background', 'hard_exudates', 'haemorrhages']
        colors = ['black', 'red', 'green']
        for i, (class_name, color) in enumerate(zip(class_names, colors)):
            val_dice_class = [metrics[f'dice_{class_name}'] for metrics in self.val_metrics_history]
            axes[0, 2].plot(val_dice_class, label=f'{class_name}', color=color)
        axes[0, 2].set_title('Class-wise Dice Scores')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Dice Score')
        axes[0, 2].legend()

        for i, (class_name, color) in enumerate(zip(class_names, colors)):
            val_iou_class = [metrics[f'iou_{class_name}'] for metrics in self.val_metrics_history]
            axes[1, 0].plot(val_iou_class, label=f'{class_name}', color=color)
        axes[1, 0].set_title('Class-wise IoU Scores')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IoU Score')
        axes[1, 0].legend()

        for i, (class_name, color) in enumerate(zip(class_names, colors)):
            val_prec_class = [metrics[f'precision_{class_name}'] for metrics in self.val_metrics_history]
            axes[1, 1].plot(val_prec_class, label=f'{class_name}', color=color)
        axes[1, 1].set_title('Class-wise Precision')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].legend()

        for i, (class_name, color) in enumerate(zip(class_names, colors)):
            val_rec_class = [metrics[f'recall_{class_name}'] for metrics in self.val_metrics_history]
            axes[1, 2].plot(val_rec_class, label=f'{class_name}', color=color)
        axes[1, 2].set_title('Class-wise Recall')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Recall')
        axes[1, 2].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()