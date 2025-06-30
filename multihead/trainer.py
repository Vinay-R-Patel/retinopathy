import os
import json
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from sklearn.metrics import (
    accuracy_score as sklearn_accuracy,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    balanced_accuracy_score, cohen_kappa_score,
    matthews_corrcoef, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import seaborn as sns
import wandb

from loss import multiclass_dice_score, accuracy_score


def calculate_comprehensive_metrics(all_preds, all_targets, num_classes, task_name="classification"):
    """Calculate comprehensive metrics for both classification and segmentation tasks"""
    metrics = {}

    if task_name == "classification":
        metrics['accuracy'] = sklearn_accuracy(all_targets, all_preds)
        metrics['balanced_accuracy'] = balanced_accuracy_score(all_targets, all_preds)

        metrics['precision_macro'] = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(all_targets, all_preds, average='macro', zero_division=0)

        metrics['precision_weighted'] = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

        precision_per_class = precision_score(all_targets, all_preds, average=None, zero_division=0)
        recall_per_class = recall_score(all_targets, all_preds, average=None, zero_division=0)
        f1_per_class = f1_score(all_targets, all_preds, average=None, zero_division=0)

        class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
        for i in range(num_classes):
            if i < len(class_names):
                metrics[f'precision_{class_names[i]}'] = precision_per_class[i]
                metrics[f'recall_{class_names[i]}'] = recall_per_class[i]
                metrics[f'f1_{class_names[i]}'] = f1_per_class[i]

        metrics['cohen_kappa'] = cohen_kappa_score(all_targets, all_preds)
        metrics['matthews_corr'] = matthews_corrcoef(all_targets, all_preds)

        cm = confusion_matrix(all_targets, all_preds)
        metrics['confusion_matrix'] = cm.tolist()

    elif task_name == "segmentation":
        metrics['mean_iou'] = calculate_mean_iou(all_preds, all_targets, num_classes)
        metrics['mean_dice'] = calculate_mean_dice(all_preds, all_targets, num_classes)

        for cls in range(num_classes):
            iou = calculate_class_iou(all_preds, all_targets, cls)
            dice = calculate_class_dice(all_preds, all_targets, cls)
            metrics[f'iou_class_{cls}'] = iou
            metrics[f'dice_class_{cls}'] = dice

    return metrics


def calculate_mean_iou(preds, targets, num_classes):
    """Calculate mean IoU for segmentation"""
    ious = []
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        target_cls = (targets == cls)

        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()

        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = float(intersection) / float(union)
        ious.append(iou)

    return np.mean(ious)


def calculate_mean_dice(preds, targets, num_classes):
    """Calculate mean Dice score for segmentation"""
    dice_scores = []
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        target_cls = (targets == cls)

        intersection = (pred_cls & target_cls).sum()
        total = pred_cls.sum() + target_cls.sum()

        if total == 0:
            dice = 1.0 if intersection == 0 else 0.0
        else:
            dice = (2.0 * float(intersection)) / float(total)
        dice_scores.append(dice)

    return np.mean(dice_scores)


def calculate_class_iou(preds, targets, class_id):
    """Calculate IoU for a specific class"""
    pred_cls = (preds == class_id)
    target_cls = (targets == class_id)

    intersection = (pred_cls & target_cls).sum()
    union = (pred_cls | target_cls).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    else:
        return float(intersection) / float(union)


def calculate_class_dice(preds, targets, class_id):
    """Calculate Dice score for a specific class"""
    pred_cls = (preds == class_id)
    target_cls = (targets == class_id)

    intersection = (pred_cls & target_cls).sum()
    total = pred_cls.sum() + target_cls.sum()

    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    else:
        return (2.0 * float(intersection)) / float(total)


def train_epoch(model, seg_loader, cls_loader, seg_criterion, cls_criterion, optimizer, device, epoch, cfg):
    model.train()
    total_loss = 0
    seg_loss_total = 0
    cls_loss_total = 0
    seg_dice_total = 0
    cls_acc_total = 0
    seg_batches = 0
    cls_batches = 0

    seg_all_preds = []
    seg_all_targets = []
    cls_all_preds = []
    cls_all_targets = []

    if epoch <= cfg.training.segmentation_only_epochs:
        task_mode = 'segmentation_only'
        print(f"🎯 Epoch {epoch}: Segmentation only")
    else:
        task_mode = 'both_tasks'
        print(f"🎯 Epoch {epoch}: Both tasks (random)")

    seg_iter = iter(seg_loader)
    cls_iter = iter(cls_loader)

    total_batches = max(len(seg_loader), len(cls_loader))

    progress_bar = tqdm(range(total_batches), desc="Training")

    for batch_idx in progress_bar:
        optimizer.zero_grad()
        batch_loss = 0

        if task_mode == 'segmentation_only' or (task_mode == 'both_tasks' and random.random() < 0.5):
            try:
                seg_images, seg_masks = next(seg_iter)
            except StopIteration:
                seg_iter = iter(seg_loader)
                seg_images, seg_masks = next(seg_iter)

            seg_images, seg_masks = seg_images.to(device), seg_masks.to(device)

            seg_outputs = model(seg_images, task='segmentation')
            seg_loss = seg_criterion(seg_outputs['segmentation'], seg_masks)
            batch_loss += seg_loss

            seg_loss_total += seg_loss.item()
            seg_dice_total += multiclass_dice_score(seg_outputs['segmentation'], seg_masks).item()
            seg_batches += 1

            seg_preds = torch.argmax(seg_outputs['segmentation'], dim=1).detach().cpu()
            seg_all_preds.append(seg_preds)
            seg_all_targets.append(seg_masks.detach().cpu())

        if task_mode == 'both_tasks':
            try:
                cls_images, cls_labels = next(cls_iter)
            except StopIteration:
                cls_iter = iter(cls_loader)
                cls_images, cls_labels = next(cls_iter)

            cls_images, cls_labels = cls_images.to(device), cls_labels.to(device)

            cls_outputs = model(cls_images, task='classification')
            cls_loss = cls_criterion(cls_outputs['classification'], cls_labels)
            batch_loss += cls_loss

            cls_loss_total += cls_loss.item()
            cls_acc_total += accuracy_score(cls_outputs['classification'], cls_labels).item()
            cls_batches += 1

            cls_preds = torch.argmax(cls_outputs['classification'], dim=1).detach().cpu()
            cls_all_preds.append(cls_preds)
            cls_all_targets.append(cls_labels.detach().cpu())

        if batch_loss > 0:
            batch_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += batch_loss.item()

        progress_bar.set_postfix({
            'Loss': f'{batch_loss.item():.4f}' if batch_loss > 0 else '0.0000'
        })

    train_metrics = {}

    if seg_all_preds:
        seg_all_preds = torch.cat(seg_all_preds).numpy()
        seg_all_targets = torch.cat(seg_all_targets).numpy()
        seg_metrics = calculate_comprehensive_metrics(
            seg_all_preds.flatten(), seg_all_targets.flatten(),
            cfg.model.seg_classes, "segmentation"
        )
        train_metrics.update({f'seg_{k}': v for k, v in seg_metrics.items()})

    if cls_all_preds:
        cls_all_preds = torch.cat(cls_all_preds).numpy()
        cls_all_targets = torch.cat(cls_all_targets).numpy()
        cls_metrics = calculate_comprehensive_metrics(
            cls_all_preds, cls_all_targets,
            cfg.model.cls_classes, "classification"
        )
        train_metrics.update({f'cls_{k}': v for k, v in cls_metrics.items()})

    avg_total_loss = total_loss / total_batches
    avg_seg_loss = seg_loss_total / seg_batches if seg_batches > 0 else 0
    avg_cls_loss = cls_loss_total / cls_batches if cls_batches > 0 else 0
    avg_seg_dice = seg_dice_total / seg_batches if seg_batches > 0 else 0
    avg_cls_acc = cls_acc_total / cls_batches if cls_batches > 0 else 0

    return avg_total_loss, avg_seg_loss, avg_cls_loss, avg_seg_dice, avg_cls_acc, train_metrics


def validate_epoch(model, seg_loader, cls_loader, seg_criterion, cls_criterion, device, cfg):
    model.eval()
    seg_loss_total = 0
    cls_loss_total = 0
    seg_dice_total = 0
    cls_acc_total = 0

    seg_all_preds = []
    seg_all_targets = []
    cls_all_preds = []
    cls_all_targets = []

    with torch.no_grad():
        for seg_images, seg_masks in tqdm(seg_loader, desc="🔍 Validating Segmentation"):
            seg_images, seg_masks = seg_images.to(device), seg_masks.to(device)

            seg_outputs = model(seg_images, task='segmentation')
            seg_loss = seg_criterion(seg_outputs['segmentation'], seg_masks)

            seg_loss_total += seg_loss.item()
            seg_dice_total += multiclass_dice_score(seg_outputs['segmentation'], seg_masks).item()

            seg_preds = torch.argmax(seg_outputs['segmentation'], dim=1).detach().cpu()
            seg_all_preds.append(seg_preds)
            seg_all_targets.append(seg_masks.detach().cpu())

        for cls_images, cls_labels in tqdm(cls_loader, desc="🔍 Validating Classification"):
            cls_images, cls_labels = cls_images.to(device), cls_labels.to(device)

            cls_outputs = model(cls_images, task='classification')
            cls_loss = cls_criterion(cls_outputs['classification'], cls_labels)

            cls_loss_total += cls_loss.item()
            cls_acc_total += accuracy_score(cls_outputs['classification'], cls_labels).item()

            cls_preds = torch.argmax(cls_outputs['classification'], dim=1).detach().cpu()
            cls_all_preds.append(cls_preds)
            cls_all_targets.append(cls_labels.detach().cpu())

    val_metrics = {}

    if seg_all_preds:
        seg_all_preds = torch.cat(seg_all_preds).numpy()
        seg_all_targets = torch.cat(seg_all_targets).numpy()
        seg_metrics = calculate_comprehensive_metrics(
            seg_all_preds.flatten(), seg_all_targets.flatten(),
            cfg.model.seg_classes, "segmentation"
        )
        val_metrics.update({f'seg_{k}': v for k, v in seg_metrics.items()})

    if cls_all_preds:
        cls_all_preds = torch.cat(cls_all_preds).numpy()
        cls_all_targets = torch.cat(cls_all_targets).numpy()
        cls_metrics = calculate_comprehensive_metrics(
            cls_all_preds, cls_all_targets,
            cfg.model.cls_classes, "classification"
        )
        val_metrics.update({f'cls_{k}': v for k, v in cls_metrics.items()})

    avg_seg_loss = seg_loss_total / len(seg_loader)
    avg_cls_loss = cls_loss_total / len(cls_loader)
    avg_seg_dice = seg_dice_total / len(seg_loader)
    avg_cls_acc = cls_acc_total / len(cls_loader)

    return avg_seg_loss, avg_cls_loss, avg_seg_dice, avg_cls_acc, val_metrics


def visualize_predictions(model, seg_loader, cls_loader, device, output_dir, num_samples=2):
    model.eval()
    fig, axes = plt.subplots(num_samples, 5, figsize=(25, 5 * num_samples))

    class_colors = {
        0: [0, 0, 0],
        1: [255, 0, 0],
        2: [0, 255, 0]
    }

    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']

    with torch.no_grad():
        for i, (seg_images, seg_masks) in enumerate(seg_loader):
            if i >= num_samples:
                break

            seg_images, seg_masks = seg_images.to(device), seg_masks.to(device)
            seg_outputs = model(seg_images, task='segmentation')

            pred_mask = torch.softmax(seg_outputs['segmentation'][0], dim=0).cpu().numpy()
            pred_class = np.argmax(pred_mask, axis=0)

            pred_colored = np.zeros((pred_class.shape[0], pred_class.shape[1], 3), dtype=np.uint8)
            for class_id, color in class_colors.items():
                pred_colored[pred_class == class_id] = color

            true_mask = seg_masks[0].cpu().numpy()
            true_colored = np.zeros((true_mask.shape[0], true_mask.shape[1], 3), dtype=np.uint8)
            for class_id, color in class_colors.items():
                true_colored[true_mask == class_id] = color

            image = seg_images[0].cpu().numpy().transpose(1, 2, 0)
            image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            image = np.clip(image, 0, 1)

            axes[i, 0].imshow(image)
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(true_colored)
            axes[i, 1].set_title('True Seg Mask')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred_colored)
            axes[i, 2].set_title('Pred Seg Mask')
            axes[i, 2].axis('off')

        for i, (cls_images, cls_labels) in enumerate(cls_loader):
            if i >= num_samples:
                break

            cls_images, cls_labels = cls_images.to(device), cls_labels.to(device)
            cls_outputs = model(cls_images, task='classification')

            pred_class = torch.argmax(cls_outputs['classification'][0]).cpu().item()
            true_class = cls_labels[0].cpu().item()

            image = cls_images[0].cpu().numpy().transpose(1, 2, 0)
            image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            image = np.clip(image, 0, 1)

            axes[i, 3].imshow(image)
            axes[i, 3].set_title('Classification Image')
            axes[i, 3].axis('off')

            axes[i, 4].text(0.1, 0.8, f'True: {class_names[true_class]}', fontsize=14, transform=axes[i, 4].transAxes)
            axes[i, 4].text(0.1, 0.6, f'Pred: {class_names[pred_class]}', fontsize=14, transform=axes[i, 4].transAxes)
            axes[i, 4].text(0.1, 0.4, f'Conf: {torch.softmax(cls_outputs["classification"][0], dim=0)[pred_class]:.3f}', fontsize=12, transform=axes[i, 4].transAxes)
            axes[i, 4].set_title('Classification Result')
            axes[i, 4].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_optimizer(model, cfg):
    optimizer_name = cfg.optimizer.name.lower()

    if optimizer_name == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay
        )
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay
        )
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
            momentum=0.9
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay
        )

    return optimizer


def create_scheduler(optimizer, cfg):
    scheduler_name = cfg.scheduler.name.lower()

    if scheduler_name == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=cfg.scheduler.patience,
            factor=cfg.scheduler.factor
        )
    elif scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.training.num_epochs,
            eta_min=1e-6
        )
    elif scheduler_name == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=getattr(cfg.scheduler, 'step_size', 30),
            gamma=getattr(cfg.scheduler, 'gamma', 0.1)
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=cfg.scheduler.patience,
            factor=cfg.scheduler.factor
        )

    return scheduler


class Trainer:
    def __init__(self, model, seg_criterion, cls_criterion, optimizer, scheduler, device, cfg):
        self.model = model
        self.seg_criterion = seg_criterion
        self.cls_criterion = cls_criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.cfg = cfg

        self.train_losses = []
        self.val_losses = []
        self.seg_train_losses = []
        self.seg_val_losses = []
        self.cls_train_losses = []
        self.cls_val_losses = []
        self.seg_dice_scores = []
        self.cls_accuracies = []

        self.train_metrics_history = []
        self.val_metrics_history = []

        self.best_seg_dice = 0.0
        self.best_cls_acc = 0.0
        self.best_combined_metric = 0.0

        if cfg.logging.use_wandb:
            wandb.init(
                project=cfg.logging.project_name,
                name=cfg.experiment_name,
                config=cfg.__dict__ if hasattr(cfg, '__dict__') else {}
            )

    def train(self, seg_train_loader, seg_test_loader, cls_train_loader, cls_test_loader):
        print(f"🚀 Starting multitask training for {self.cfg.training.num_epochs} epochs")

        for epoch in range(1, self.cfg.training.num_epochs + 1):
            train_loss, seg_train_loss, cls_train_loss, seg_dice, cls_acc, train_metrics = train_epoch(
                self.model, seg_train_loader, cls_train_loader,
                self.seg_criterion, self.cls_criterion, self.optimizer,
                self.device, epoch, self.cfg
            )

            val_seg_loss, val_cls_loss, val_seg_dice, val_cls_acc, val_metrics = validate_epoch(
                self.model, seg_test_loader, cls_test_loader,
                self.seg_criterion, self.cls_criterion, self.device, self.cfg
            )

            val_loss = val_seg_loss + val_cls_loss

            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.seg_train_losses.append(seg_train_loss)
            self.seg_val_losses.append(val_seg_loss)
            self.cls_train_losses.append(cls_train_loss)
            self.cls_val_losses.append(val_cls_loss)
            self.seg_dice_scores.append(val_seg_dice)
            self.cls_accuracies.append(val_cls_acc)

            self.train_metrics_history.append(train_metrics)
            self.val_metrics_history.append(val_metrics)

            print(f"📊 Epoch {epoch:3d} | "
                  f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                  f"Seg Dice: {val_seg_dice:.4f} | "
                  f"Cls Acc: {val_cls_acc:.4f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            if self.cfg.logging.use_wandb:
                log_dict = {
                    'epoch': epoch,
                    'train/total_loss': train_loss,
                    'train/seg_loss': seg_train_loss,
                    'train/cls_loss': cls_train_loss,
                    'val/total_loss': val_loss,
                    'val/seg_loss': val_seg_loss,
                    'val/cls_loss': val_cls_loss,
                    'val/seg_dice': val_seg_dice,
                    'val/cls_accuracy': val_cls_acc,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }

                for key, value in train_metrics.items():
                    if not isinstance(value, list):
                        log_dict[f'train/{key}'] = value

                for key, value in val_metrics.items():
                    if not isinstance(value, list):
                        log_dict[f'val/{key}'] = value

                wandb.log(log_dict)

            combined_metric = val_seg_dice + val_cls_acc

            if val_seg_dice > self.best_seg_dice:
                self.best_seg_dice = val_seg_dice
                torch.save(self.model.state_dict(),
                          os.path.join(self.cfg.output_base_dir, 'best_seg_model.pth'))

            if val_cls_acc > self.best_cls_acc:
                self.best_cls_acc = val_cls_acc
                torch.save(self.model.state_dict(),
                          os.path.join(self.cfg.output_base_dir, 'best_cls_model.pth'))

            if combined_metric > self.best_combined_metric:
                self.best_combined_metric = combined_metric
                torch.save(self.model.state_dict(),
                          os.path.join(self.cfg.output_base_dir, 'best_combined_model.pth'))

            if epoch % self.cfg.training.save_freq == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'best_seg_dice': self.best_seg_dice,
                    'best_cls_acc': self.best_cls_acc,
                    'best_combined_metric': self.best_combined_metric,
                }, os.path.join(self.cfg.output_base_dir, f'checkpoint_epoch_{epoch}.pth'))

        visualize_predictions(
            self.model, seg_test_loader, cls_test_loader,
            self.device, self.cfg.output_base_dir
        )

        torch.save(self.model.state_dict(),
                  os.path.join(self.cfg.output_base_dir, 'final_model.pth'))

        self.save_training_plots()
        self.save_comprehensive_report()

        print(f"🎉 Training completed!")
        print(f"Best Segmentation Dice: {self.best_seg_dice:.4f}")
        print(f"Best Classification Accuracy: {self.best_cls_acc:.4f}")
        print(f"Best Combined Metric: {self.best_combined_metric:.4f}")

    def save_training_plots(self):
        """Create comprehensive training visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        epochs = range(1, len(self.train_losses) + 1)

        axes[0, 0].plot(epochs, self.train_losses, label='Train', color='blue')
        axes[0, 0].plot(epochs, self.val_losses, label='Validation', color='red')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        axes[0, 1].plot(epochs, self.seg_train_losses, label='Train', color='blue')
        axes[0, 1].plot(epochs, self.seg_val_losses, label='Validation', color='red')
        axes[0, 1].set_title('Segmentation Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        axes[0, 2].plot(epochs, self.cls_train_losses, label='Train', color='blue')
        axes[0, 2].plot(epochs, self.cls_val_losses, label='Validation', color='red')
        axes[0, 2].set_title('Classification Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True)

        axes[1, 0].plot(epochs, self.seg_dice_scores, color='green')
        axes[1, 0].set_title('Segmentation Dice Score')
        axes[1, 0].grid(True)

        axes[1, 1].plot(epochs, self.cls_accuracies, color='purple')
        axes[1, 1].set_title('Classification Accuracy')
        axes[1, 1].grid(True)

        combined_metrics = [dice + acc for dice, acc in zip(self.seg_dice_scores, self.cls_accuracies)]
        axes[1, 2].plot(epochs, combined_metrics, color='orange')
        axes[1, 2].set_title('Combined Metric (Dice + Accuracy)')
        axes[1, 2].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.output_base_dir, 'training_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def save_comprehensive_report(self):
        """Save comprehensive training report"""
        report = {
            'experiment_name': self.cfg.experiment_name,
            'final_metrics': {
                'best_segmentation_dice': self.best_seg_dice,
                'best_classification_accuracy': self.best_cls_acc,
                'best_combined_metric': self.best_combined_metric,
                'final_seg_dice': self.seg_dice_scores[-1] if self.seg_dice_scores else 0,
                'final_cls_accuracy': self.cls_accuracies[-1] if self.cls_accuracies else 0
            },
            'training_config': {
                'num_epochs': self.cfg.training.num_epochs,
                'batch_size': self.cfg.training.batch_size,
                'learning_rate': self.cfg.optimizer.lr,
                'optimizer': self.cfg.optimizer.name,
                'scheduler': self.cfg.scheduler.name,
                'segmentation_loss': self.cfg.segmentation_loss.name,
                'classification_loss': self.cfg.classification_loss.name
            },
            'model_config': {
                'encoder_name': getattr(self.cfg.model, 'encoder_name', 'resnext50_32x4d'),
                'segmentation_head': getattr(self.cfg.model, 'segmentation_head', 'Segformer'),
                'seg_classes': getattr(self.cfg.model, 'seg_classes', 3),
                'cls_classes': getattr(self.cfg.model, 'cls_classes', 5)
            }
        }

        if self.val_metrics_history:
            report['final_validation_metrics'] = self.val_metrics_history[-1]

        with open(os.path.join(self.cfg.output_base_dir, 'training_report.json'), 'w') as f:
            json.dump(report, f, indent=2)

        print(f"📋 Comprehensive report saved to {self.cfg.output_base_dir}/training_report.json")