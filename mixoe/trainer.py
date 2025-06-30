from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd

import wandb
WANDB_AVAILABLE = True



def compute_comprehensive_segmentation_metrics(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 3):
    """Compute comprehensive segmentation metrics including per-class IoU and dice scores"""
    metrics = {}
    
    # Convert predictions to class predictions
    seg_pred_classes = torch.argmax(torch.softmax(preds, dim=1), dim=1)
    
    # Overall metrics
    dice_scores = []
    iou_scores = []
    pixel_accuracies = []
    
    for c in range(num_classes):
        pred_c = (seg_pred_classes == c)
        target_c = (targets == c)
        
        # Pixel accuracy for this class
        if torch.sum(target_c) > 0:  # Only compute if class exists in target
            pixel_acc = torch.sum(pred_c & target_c).float() / torch.sum(target_c).float()
            pixel_accuracies.append(pixel_acc.item())
        else:
            pixel_accuracies.append(1.0)
        
        # Dice score
        intersection = torch.sum(pred_c & target_c).float()
        union = torch.sum(pred_c).float() + torch.sum(target_c).float()
        if union > 0:
            dice = (2 * intersection) / union
            dice_scores.append(dice.item())
        else:
            dice_scores.append(1.0)  # Perfect score if both are empty
        
        # IoU score
        intersection = torch.sum(pred_c & target_c).float()
        union = torch.sum(pred_c | target_c).float()
        if union > 0:
            iou = intersection / union
            iou_scores.append(iou.item())
        else:
            iou_scores.append(1.0)  # Perfect score if both are empty
        
        # Per-class metrics
        metrics[f"seg_dice_class_{c}"] = dice_scores[-1]
        metrics[f"seg_iou_class_{c}"] = iou_scores[-1]
        metrics[f"seg_pixel_acc_class_{c}"] = pixel_accuracies[-1]
    
    # Average metrics
    metrics["seg_dice"] = np.mean(dice_scores)
    metrics["seg_mean_iou"] = np.mean(iou_scores)
    metrics["seg_mean_pixel_acc"] = np.mean(pixel_accuracies)
    
    # Overall pixel accuracy
    total_correct = torch.sum(seg_pred_classes == targets).float()
    total_pixels = torch.numel(targets)
    metrics["seg_overall_pixel_acc"] = (total_correct / total_pixels).item()
    
    return metrics


def compute_comprehensive_classification_metrics(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 5):
    """Compute comprehensive classification metrics with enhanced details"""
    
    # Convert to numpy for sklearn
    pred_classes = torch.argmax(preds, dim=1).detach().cpu().numpy()
    true_classes = targets.detach().cpu().numpy()
    pred_probs = torch.softmax(preds, dim=1).detach().cpu().numpy()
    
    metrics = {}
    
    # Basic accuracy
    accuracy = np.mean(pred_classes == true_classes)
    metrics["cls_accuracy"] = accuracy
    
    # Per-class accuracies (for DR grades)
    class_counts = np.bincount(true_classes, minlength=num_classes)
    for grade in range(num_classes):
        grade_mask = true_classes == grade
        if np.sum(grade_mask) > 0:
            grade_acc = np.mean(pred_classes[grade_mask] == true_classes[grade_mask])
            metrics[f"cls_accuracy_grade_{grade}"] = grade_acc
            metrics[f"cls_support_grade_{grade}"] = int(np.sum(grade_mask))
        else:
            metrics[f"cls_accuracy_grade_{grade}"] = 0.0
            metrics[f"cls_support_grade_{grade}"] = 0
    
    # Confusion matrix-based metrics
    cm = confusion_matrix(true_classes, pred_classes, labels=list(range(num_classes)))
    
    # Per-class precision, recall, f1
    precisions, recalls, f1s = [], [], []
    for grade in range(num_classes):
        if cm[grade, :].sum() > 0:  # If there are true samples of this class
            precision = cm[grade, grade] / max(cm[:, grade].sum(), 1e-8)  # TP / (TP + FP)
            recall = cm[grade, grade] / max(cm[grade, :].sum(), 1e-8)     # TP / (TP + FN)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            
            metrics[f"cls_precision_grade_{grade}"] = precision
            metrics[f"cls_recall_grade_{grade}"] = recall  
            metrics[f"cls_f1_grade_{grade}"] = f1
            
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        else:
            metrics[f"cls_precision_grade_{grade}"] = 0.0
            metrics[f"cls_recall_grade_{grade}"] = 0.0
            metrics[f"cls_f1_grade_{grade}"] = 0.0
            precisions.append(0.0)
            recalls.append(0.0)
            f1s.append(0.0)
    
    # Macro averages
    metrics["cls_macro_precision"] = np.mean(precisions)
    metrics["cls_macro_recall"] = np.mean(recalls)
    metrics["cls_macro_f1"] = np.mean(f1s)
    
    # Weighted averages
    total_samples = len(true_classes)
    if total_samples > 0:
        weights = class_counts / total_samples
        metrics["cls_weighted_precision"] = np.average(precisions, weights=weights)
        metrics["cls_weighted_recall"] = np.average(recalls, weights=weights)
        metrics["cls_weighted_f1"] = np.average(f1s, weights=weights)
    else:
        metrics["cls_weighted_precision"] = 0.0
        metrics["cls_weighted_recall"] = 0.0
        metrics["cls_weighted_f1"] = 0.0
    
    # Advanced metrics
    if len(np.unique(true_classes)) > 1 and len(np.unique(pred_classes)) > 1:
        metrics["cls_cohen_kappa"] = cohen_kappa_score(true_classes, pred_classes)
        metrics["cls_matthews_corr"] = matthews_corrcoef(true_classes, pred_classes)
        
        # ROC-AUC for multi-class (one-vs-rest)
        try:
            if num_classes == 2:
                metrics["cls_roc_auc"] = roc_auc_score(true_classes, pred_probs[:, 1])
            else:
                metrics["cls_roc_auc_ovr"] = roc_auc_score(true_classes, pred_probs, multi_class='ovr', average='macro')
                metrics["cls_roc_auc_ovo"] = roc_auc_score(true_classes, pred_probs, multi_class='ovo', average='macro')
        except ValueError:
            pass  # Skip if not enough classes represented
            
        # Average precision score
        try:
            if num_classes == 2:
                metrics["cls_avg_precision"] = average_precision_score(true_classes, pred_probs[:, 1])
            else:
                # For multi-class, compute per-class and average
                from sklearn.preprocessing import label_binarize
                true_binary = label_binarize(true_classes, classes=list(range(num_classes)))
                avg_prec_scores = []
                for i in range(num_classes):
                    if true_binary[:, i].sum() > 0:  # Only if class exists
                        avg_prec_scores.append(average_precision_score(true_binary[:, i], pred_probs[:, i]))
                if avg_prec_scores:
                    metrics["cls_avg_precision_macro"] = np.mean(avg_prec_scores)
        except (ValueError, ImportError):
            pass
            
    else:
        metrics["cls_cohen_kappa"] = 0.0
        metrics["cls_matthews_corr"] = 0.0
    
    # Confidence statistics
    max_probs = np.max(pred_probs, axis=1)
    metrics["cls_mean_confidence"] = np.mean(max_probs)
    metrics["cls_confidence_std"] = np.std(max_probs)
    
    # Prediction entropy (uncertainty measure)
    entropy = -np.sum(pred_probs * np.log(pred_probs + 1e-8), axis=1)
    metrics["cls_mean_entropy"] = np.mean(entropy)
    metrics["cls_entropy_std"] = np.std(entropy)
    
    return metrics, cm, pred_probs


def compute_metrics(preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], config=None):
    """Compute all metrics for both segmentation and classification tasks"""
    metrics = {}
    additional_data = {}
    
    # Segmentation metrics
    if "seg" in preds and not torch.all(targets["seg"] == -1):
        seg_targets = targets["seg"]
        seg_preds = preds["seg"]
        
        valid_samples = ~torch.all(seg_targets.view(seg_targets.size(0), -1) == -1, dim=1)
        
        if valid_samples.any():
            valid_seg_targets = seg_targets[valid_samples]
            valid_seg_preds = seg_preds[valid_samples]
            
            seg_classes = getattr(config.model, 'seg_classes', 3) if config else 3
            seg_metrics = compute_comprehensive_segmentation_metrics(
                valid_seg_preds, valid_seg_targets, 
                num_classes=seg_classes
            )
            metrics.update(seg_metrics)
    
    # Classification metrics  
    if "cls" in preds and not torch.all(targets["cls"] == -1):
        valid_mask = targets["cls"] != -1
        if valid_mask.any():
            valid_cls_targets = targets["cls"][valid_mask]
            valid_cls_preds = preds["cls"][valid_mask]
            
            cls_classes = getattr(config.model, 'cls_classes', 5) if config else 5
            cls_metrics, confusion_mat, pred_probabilities = compute_comprehensive_classification_metrics(
                valid_cls_preds, valid_cls_targets,
                num_classes=cls_classes
            )
            metrics.update(cls_metrics)
            additional_data['confusion_matrix'] = confusion_mat
            additional_data['prediction_probabilities'] = pred_probabilities
            additional_data['true_labels'] = valid_cls_targets.detach().cpu().numpy()
    
    return metrics, additional_data


def enhanced_log_to_wandb(metrics, epoch, prefix="train", additional_data=None):
    """Enhanced WandB logging with tables, confusion matrices, and better organization"""
    if not WANDB_AVAILABLE:
        return
        
    wandb_metrics = {}
    
    # Organize metrics by category
    loss_metrics = {}
    seg_metrics = {}
    cls_metrics = {}
    other_metrics = {}
    
    for key, value in metrics.items():
        # Ensure numpy types are converted to float
        if hasattr(value, 'item'):
            value = float(value.item())
        elif isinstance(value, np.number):
            value = float(value)
        
        # Categorize metrics
        if 'loss' in key.lower():
            loss_metrics[key] = value
        elif key.startswith('seg_'):
            seg_metrics[key] = value
        elif key.startswith('cls_'):
            cls_metrics[key] = value
        else:
            other_metrics[key] = value
    
    # Log categorized metrics
    for category, cat_metrics in [("loss", loss_metrics), ("segmentation", seg_metrics), 
                                  ("classification", cls_metrics), ("other", other_metrics)]:
        for key, value in cat_metrics.items():
            wandb_metrics[f"{prefix}/{category}/{key}"] = value
    
    # Add epoch
    wandb_metrics["epoch"] = epoch
    
    # Log basic metrics
    wandb.log(wandb_metrics)
    
    # Enhanced logging for validation data
    if prefix == "val" and additional_data:
        # Log confusion matrix if available
        if 'confusion_matrix' in additional_data:
            cm = additional_data['confusion_matrix']
            class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'][:cm.shape[0]]
            
            # Create confusion matrix plot
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
            plt.title(f'Confusion Matrix - Epoch {epoch}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Log to wandb
            wandb.log({f"{prefix}/confusion_matrix": wandb.Image(plt)})
            plt.close()
            
            # Create normalized confusion matrix
            # Handle zero sums to avoid division by zero warning
            row_sums = cm.sum(axis=1)[:, np.newaxis]
            row_sums[row_sums == 0] = 1  # Replace zero sums with 1 to avoid division by zero
            cm_normalized = cm.astype('float') / row_sums
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            plt.title(f'Normalized Confusion Matrix - Epoch {epoch}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            wandb.log({f"{prefix}/confusion_matrix_normalized": wandb.Image(plt)})
            plt.close()
        
        # Log prediction confidence histogram
        if 'prediction_probabilities' in additional_data:
            pred_probs = additional_data['prediction_probabilities']
            max_probs = np.max(pred_probs, axis=1)
            
            plt.figure(figsize=(10, 6))
            plt.hist(max_probs, bins=50, alpha=0.7, edgecolor='black')
            plt.title(f'Prediction Confidence Distribution - Epoch {epoch}')
            plt.xlabel('Maximum Probability')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            wandb.log({f"{prefix}/confidence_histogram": wandb.Image(plt)})
            plt.close()
        
        # Create per-class metrics table
        cls_table_data = []
        class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
        
        for i, class_name in enumerate(class_names):
            if f"cls_accuracy_grade_{i}" in metrics:
                cls_table_data.append([
                    class_name,
                    f"{metrics.get(f'cls_accuracy_grade_{i}', 0):.3f}",
                    f"{metrics.get(f'cls_precision_grade_{i}', 0):.3f}",
                    f"{metrics.get(f'cls_recall_grade_{i}', 0):.3f}",
                    f"{metrics.get(f'cls_f1_grade_{i}', 0):.3f}",
                    f"{metrics.get(f'cls_support_grade_{i}', 0)}"
                ])
        
        if cls_table_data:
            cls_table = wandb.Table(
                columns=["Class", "Accuracy", "Precision", "Recall", "F1-Score", "Support"],
                data=cls_table_data
            )
            wandb.log({f"{prefix}/per_class_metrics": cls_table})
        
        # Create segmentation metrics table if available
        seg_table_data = []
        seg_class_names = ['Background', 'Hard Exudates', 'Haemorrhages']
        
        for i, class_name in enumerate(seg_class_names):
            if f"seg_dice_class_{i}" in metrics:
                seg_table_data.append([
                    class_name,
                    f"{metrics.get(f'seg_dice_class_{i}', 0):.3f}",
                    f"{metrics.get(f'seg_iou_class_{i}', 0):.3f}",
                    f"{metrics.get(f'seg_pixel_acc_class_{i}', 0):.3f}"
                ])
        
        if seg_table_data:
            seg_table = wandb.Table(
                columns=["Class", "Dice Score", "IoU", "Pixel Accuracy"],
                data=seg_table_data
            )
            wandb.log({f"{prefix}/segmentation_metrics": seg_table})


def train_one_epoch(model, seg_loader, cls_loader, criterion, optimizer, device, epoch, config, scheduler= None):
    model.train()
    
    seg_iter = iter(seg_loader)
    cls_iter = iter(cls_loader)
    
    total_loss = 0
    seg_metrics = []
    cls_metrics = []
    loss_components = {"seg": [], "cls": [], "seg_weight": [], "cls_weight": []}
    
    max_batches = max(len(seg_loader), len(cls_loader))
    
    progress_bar = tqdm(range(max_batches), desc= f"🚀 Training Epoch {epoch}")
    
    for batch_idx in progress_bar:
        optimizer.zero_grad()
        
        try:
            seg_batch = next(seg_iter)
        except StopIteration:
            seg_iter = iter(seg_loader)
            seg_batch = next(seg_iter)
        
        try:
            cls_batch = next(cls_iter)
        except StopIteration:
            cls_iter = iter(cls_loader)
            cls_batch = next(cls_iter)
        
        seg_images = seg_batch["image"].to(device)
        seg_masks = seg_batch["seg"].to(device)
        cls_images = cls_batch["image"].to(device)
        cls_labels = cls_batch["cls"].to(device)
        
        images = torch.cat([seg_images, cls_images], dim=0)
        
        outputs = model(images)
        
        seg_outputs = {k: v[:seg_images.size(0)] for k, v in outputs.items()}
        cls_outputs = {k: v[seg_images.size(0):] for k, v in outputs.items()}
        
        targets = {
            "seg": seg_masks,
            "cls": cls_labels
        }
        
        preds = {
            "seg": seg_outputs["seg"],
            "cls": cls_outputs["cls"]
        }
        
        loss, loss_dict = criterion(preds, targets)
        
        # Gradient clipping
        if hasattr(config.training, 'gradient_clip_val') and config.training.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_val)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                loss_components[k].append(v.item())
            else:
                loss_components[k].append(float(v))
        
        # Compute metrics for this batch
        batch_seg_metrics, _ = compute_metrics(
            seg_outputs, 
            {"seg": seg_masks, "cls": torch.tensor([-1] * seg_masks.size(0), device=device)}, 
            config
        )
        batch_cls_metrics, _ = compute_metrics(
            cls_outputs, 
            {"seg": torch.tensor([-1] * cls_labels.size(0), device=device), "cls": cls_labels}, 
            config
        )
        
        seg_metrics.append(batch_seg_metrics)
        cls_metrics.append(batch_cls_metrics)
        
        # Update progress bar with emojis and key metrics
        current_seg_dice = batch_seg_metrics.get("seg_dice", 0)
        current_cls_acc = batch_cls_metrics.get("cls_accuracy", 0)
        
        progress_bar.set_postfix({
            '📉 Loss': f'{loss.item():.4f}',
            '🎯 Seg Dice': f'{current_seg_dice:.3f}',
            '🎪 Cls Acc': f'{current_cls_acc:.3f}',
            '⚖️ Seg W': f'{float(loss_dict["seg_weight"]):.3f}',
            '⚖️ Cls W': f'{float(loss_dict["cls_weight"]):.3f}'
        })
    
    # Calculate epoch averages
    avg_metrics = {
        "total_loss": total_loss / max_batches,
        "seg_loss": np.mean(loss_components["seg"]),
        "cls_loss": np.mean(loss_components["cls"]),
        "seg_weight": np.mean(loss_components["seg_weight"]),
        "cls_weight": np.mean(loss_components["cls_weight"]),
    }
    
    # Average all segmentation metrics
    for key in ["seg_dice", "seg_mean_iou"] + [f"seg_dice_class_{i}" for i in range(config.model.seg_classes)] + [f"seg_iou_class_{i}" for i in range(config.model.seg_classes)]:
        values = [m.get(key, 0) for m in seg_metrics if key in m]
        if values:
            avg_metrics[key] = np.mean(values)
    
    # Average all classification metrics
    cls_metric_keys = [
        "cls_accuracy", "cls_macro_precision", "cls_macro_recall", "cls_macro_f1",
        "cls_weighted_precision", "cls_weighted_recall", "cls_weighted_f1",
        "cls_cohen_kappa", "cls_matthews_corr"
    ] + [f"cls_accuracy_grade_{i}" for i in range(config.model.cls_classes)] + [f"cls_precision_grade_{i}" for i in range(config.model.cls_classes)] + [f"cls_recall_grade_{i}" for i in range(config.model.cls_classes)] + [f"cls_f1_grade_{i}" for i in range(config.model.cls_classes)]
    
    for key in cls_metric_keys:
        values = [m.get(key, 0) for m in cls_metrics if key in m]
        if values:
            avg_metrics[key] = np.mean(values)
    
    return avg_metrics


@torch.no_grad()
def validate(model, seg_loader, cls_loader, criterion, device, config):
    model.eval()
    
    seg_metrics = []
    cls_metrics = []
    total_val_loss = 0
    val_additional_data = None
    
    print("🔍 Validating Segmentation...")
    for batch in tqdm(seg_loader, desc="Seg Validation"):
        images = batch["image"].to(device)
        masks = batch["seg"].to(device)
        
        outputs = model(images)
        
        # Only compute metrics, skip validation loss for segmentation to avoid dummy tensor issues
        metrics, _ = compute_metrics(
            outputs, 
            {"seg": masks, "cls": torch.tensor([-1] * masks.size(0), device=device)}, 
            config
        )
        seg_metrics.append(metrics)
    
    print("🎪 Validating Classification...")
    for batch in tqdm(cls_loader, desc="Cls Validation"):
        images = batch["image"].to(device)
        labels = batch["cls"].to(device)
        
        outputs = model(images)
        
        # Only compute metrics, skip validation loss for classification to avoid dummy tensor issues
        metrics, additional_data = compute_metrics(
            outputs, 
            {"seg": torch.tensor([-1] * labels.size(0), device=device), "cls": labels}, 
            config
        )
        cls_metrics.append(metrics)
        
        # Store additional data for enhanced logging (only keep the last batch for validation)
        if additional_data:
            val_additional_data = additional_data
    
    val_metrics = {
        "val_total_loss": 0.0,  # Skip validation loss computation
    }
    
    # Average segmentation metrics  
    for key in ["seg_dice", "seg_mean_iou", "seg_mean_pixel_acc", "seg_overall_pixel_acc"]:
        values = [m.get(key, 0) for m in seg_metrics if key in m]
        if values:
            val_metrics[f"val_{key}"] = np.mean(values)
    
    # Add per-class segmentation metrics
    if config:
        for c in range(getattr(config.model, 'seg_classes', 3)):
            for metric_type in ['dice', 'iou', 'pixel_acc']:
                key = f"seg_{metric_type}_class_{c}"
                values = [m.get(key, 0) for m in seg_metrics if key in m]
                if values:
                    val_metrics[f"val_{key}"] = np.mean(values)
    
    # Average classification metrics
    cls_metric_keys = [
        "cls_accuracy", "cls_macro_precision", "cls_macro_recall", "cls_macro_f1",
        "cls_weighted_precision", "cls_weighted_recall", "cls_weighted_f1",
        "cls_cohen_kappa", "cls_matthews_corr", "cls_mean_confidence", "cls_confidence_std",
        "cls_mean_entropy", "cls_entropy_std"
    ]
    
    # Add ROC-AUC and average precision if available
    cls_metric_keys.extend(["cls_roc_auc", "cls_roc_auc_ovr", "cls_roc_auc_ovo", 
                           "cls_avg_precision", "cls_avg_precision_macro"])
    
    for key in cls_metric_keys:
        values = [m.get(key, 0) for m in cls_metrics if key in m]
        if values:
            val_metrics[f"val_{key}"] = np.mean(values)
    
    # Add per-class classification metrics
    if config:
        for grade in range(getattr(config.model, 'cls_classes', 5)):
            for metric_type in ['accuracy', 'precision', 'recall', 'f1', 'support']:
                key = f"cls_{metric_type}_grade_{grade}"
                values = [m.get(key, 0) for m in cls_metrics if key in m]
                if values:
                    if metric_type == 'support':
                        val_metrics[f"val_{key}"] = np.sum(values)  # Sum support counts
                    else:
                        val_metrics[f"val_{key}"] = np.mean(values)
    
    return val_metrics, val_additional_data


def visualize_predictions(model, seg_loader, cls_loader, device, config, epoch, save_dir):
    """Create and save prediction visualizations"""
    model.eval()
    num_samples = getattr(config.output, 'num_visualization_samples', 2)
    
    # Segmentation visualizations
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    axes = axes.reshape(num_samples, 4) if num_samples > 1 else axes.reshape(1, 4)
    
    class_colors = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0]]) / 255.0  # Background, Hard Exudates, Haemorrhages
    
    with torch.no_grad():
        for i, batch in enumerate(seg_loader):
            if i >= num_samples:
                break
                
            images = batch["image"].to(device)
            masks = batch["seg"].to(device)
            
            outputs = model(images)
            pred_mask = torch.softmax(outputs["seg"][0], dim=0).cpu().numpy()
            pred_class = np.argmax(pred_mask, axis=0)
            
            # Original image
            img = images[0].cpu().permute(1, 2, 0).numpy()
            img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
            axes[i, 0].imshow(img)
            axes[i, 0].set_title("Original Image")
            axes[i, 0].axis('off')
            
            # Ground truth mask
            gt_mask = masks[0].cpu().numpy()
            gt_colored = np.zeros((*gt_mask.shape, 3))
            for c in range(len(class_colors)):
                gt_colored[gt_mask == c] = class_colors[c]
            axes[i, 1].imshow(gt_colored)
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis('off')
            
            # Predicted mask
            pred_colored = np.zeros((*pred_class.shape, 3))
            for c in range(len(class_colors)):
                pred_colored[pred_class == c] = class_colors[c]
            axes[i, 2].imshow(pred_colored)
            axes[i, 2].set_title("Prediction")
            axes[i, 2].axis('off')
            
            # Confidence map
            confidence = np.max(pred_mask, axis=0)
            axes[i, 3].imshow(confidence, cmap='viridis')
            axes[i, 3].set_title("Confidence")
            axes[i, 3].axis('off')
    
    plt.tight_layout()
    seg_viz_path = os.path.join(save_dir, f"segmentation_predictions_epoch_{epoch}.png")
    plt.savefig(seg_viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Classification visualizations
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5*num_samples))
    axes = axes.reshape(num_samples, 2) if num_samples > 1 else axes.reshape(1, 2)
    
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    with torch.no_grad():
        for i, batch in enumerate(cls_loader):
            if i >= num_samples:
                break
                
            images = batch["image"].to(device)
            labels = batch["cls"].to(device)
            
            outputs = model(images)
            pred_probs = torch.softmax(outputs["cls"][0], dim=0).cpu().numpy()
            pred_class = np.argmax(pred_probs)
            true_class = labels[0].cpu().item()
            
            # Original image
            img = images[0].cpu().permute(1, 2, 0).numpy()
            img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f"True: {class_names[true_class]}, Pred: {class_names[pred_class]}")
            axes[i, 0].axis('off')
            
            # Prediction probabilities
            axes[i, 1].bar(class_names, pred_probs)
            axes[i, 1].set_title("Prediction Probabilities")
            axes[i, 1].set_ylabel("Probability")
            axes[i, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    cls_viz_path = os.path.join(save_dir, f"classification_predictions_epoch_{epoch}.png")
    plt.savefig(cls_viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return seg_viz_path, cls_viz_path


def plot_training_curves(train_history, val_history, config, save_dir):
    """Create comprehensive training plots"""
    epochs = range(1, len(train_history) + 1)
    
    # Create comprehensive subplot layout
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('🎯 MixOE Training Progress Dashboard', fontsize=16, fontweight='bold')
    
    # Loss curves
    axes[0, 0].plot(epochs, [h['total_loss'] for h in train_history], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, [h.get('val_total_loss', 0) for h in val_history], 'r-', label='Val', linewidth=2)
    axes[0, 0].set_title('📉 Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Segmentation metrics
    axes[0, 1].plot(epochs, [h.get('seg_dice', 0) for h in train_history], 'b-', label='Train Dice', linewidth=2)
    axes[0, 1].plot(epochs, [h.get('val_seg_dice', 0) for h in val_history], 'r-', label='Val Dice', linewidth=2)
    axes[0, 1].plot(epochs, [h.get('seg_mean_iou', 0) for h in train_history], 'g--', label='Train IoU', linewidth=2)
    axes[0, 1].plot(epochs, [h.get('val_seg_mean_iou', 0) for h in val_history], 'orange', linestyle='--', label='Val IoU', linewidth=2)
    axes[0, 1].set_title('🎯 Segmentation Metrics')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Classification accuracy
    axes[0, 2].plot(epochs, [h.get('cls_accuracy', 0) for h in train_history], 'b-', label='Train Acc', linewidth=2)
    axes[0, 2].plot(epochs, [h.get('val_cls_accuracy', 0) for h in val_history], 'r-', label='Val Acc', linewidth=2)
    axes[0, 2].set_title('🎪 Classification Accuracy')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Accuracy')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Loss components
    axes[1, 0].plot(epochs, [h.get('seg_loss', 0) for h in train_history], 'b-', label='Seg Loss', linewidth=2)
    axes[1, 0].plot(epochs, [h.get('cls_loss', 0) for h in train_history], 'r-', label='Cls Loss', linewidth=2)
    axes[1, 0].set_title('📊 Loss Components')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Task weights
    axes[1, 1].plot(epochs, [h.get('seg_weight', 0) for h in train_history], 'b-', label='Seg Weight', linewidth=2)
    axes[1, 1].plot(epochs, [h.get('cls_weight', 0) for h in train_history], 'r-', label='Cls Weight', linewidth=2)
    axes[1, 1].set_title('⚖️ Task Weights')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Weight')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Advanced classification metrics
    axes[1, 2].plot(epochs, [h.get('cls_macro_f1', 0) for h in train_history], 'b-', label='Train F1', linewidth=2)
    axes[1, 2].plot(epochs, [h.get('val_cls_macro_f1', 0) for h in val_history], 'r-', label='Val F1', linewidth=2)
    axes[1, 2].plot(epochs, [h.get('cls_cohen_kappa', 0) for h in train_history], 'g--', label='Train Kappa', linewidth=2)
    axes[1, 2].plot(epochs, [h.get('val_cls_cohen_kappa', 0) for h in val_history], 'orange', linestyle='--', label='Val Kappa', linewidth=2)
    axes[1, 2].set_title('📈 Advanced Cls Metrics')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # Per-class segmentation dice scores
    seg_classes = config.model.seg_classes
    for c in range(min(seg_classes, 3)):  # Limit to 3 classes for visualization
        axes[2, 0].plot(epochs, [h.get(f'seg_dice_class_{c}', 0) for h in train_history], 
                       label= f'Class {c}', linewidth=2)
    axes[2, 0].set_title('🎯 Per-Class Dice Scores')
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Dice Score')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Per-grade classification accuracies
    cls_classes = config.model.cls_classes
    for grade in range(min(cls_classes, 5)):  # DR grades
        axes[2, 1].plot(epochs, [h.get(f'cls_accuracy_grade_{grade}', 0) for h in train_history],
                       label= f'Grade {grade}', linewidth=2)
    axes[2, 1].set_title('🎪 Per-Grade Accuracies')
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('Accuracy')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    # Validation summary
    if val_history:
        latest_val = val_history[-1]
        summary_text = f"""📊 Latest Validation Metrics:
🎯 Seg Dice: {latest_val.get('val_seg_dice', 0):.3f}
🎯 Seg mIoU: {latest_val.get('val_seg_mean_iou', 0):.3f}
🎪 Cls Acc: {latest_val.get('val_cls_accuracy', 0):.3f}
🎪 Cls F1: {latest_val.get('val_cls_macro_f1', 0):.3f}
🎪 Cohen κ: {latest_val.get('val_cls_cohen_kappa', 0):.3f}"""
        
        axes[2, 2].text(0.1, 0.5, summary_text, transform= axes[2, 2].transAxes, 
                       fontsize=12, verticalalignment='center',
                       bbox= dict(boxstyle="round, pad=0.3", facecolor="lightblue", alpha=0.8))
        axes[2, 2].set_title('📋 Validation Summary')
        axes[2, 2].axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def save_detailed_report(train_history, val_history, config, save_dir, best_metrics):
    """Save a detailed training report"""
    report_path = os.path.join(save_dir, 'training_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("🎯 MIXOE MULTITASK TRAINING REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Configuration summary
        f.write("📋 CONFIGURATION:\n")
        f.write(f"  • Model: {getattr(config.model, 'backbone_name', 'N/A')}\n")
        f.write(f"  • Seg Classes: {config.model.seg_classes}\n")
        f.write(f"  • Cls Classes: {config.model.cls_classes}\n")
        f.write(f"  • Image Size: {config.preprocessing.image_size}\n")
        f.write(f"  • Batch Size: {config.training.batch_size}\n")
        f.write(f"  • Learning Rate: {config.training.learning_rate}\n")
        f.write(f"  • Epochs: {len(train_history)}\n\n")
        
        # Best metrics summary
        f.write("🏆 BEST METRICS:\n")
        for metric_name, (value, epoch) in best_metrics.items():
            f.write(f"  • {metric_name}: {value:.4f} (Epoch {epoch})\n")
        f.write("\n")
        
        # Final metrics
        if val_history:
            final_val = val_history[-1]
            f.write("📊 FINAL VALIDATION METRICS:\n")
            f.write(f"  🎯 Segmentation:\n")
            f.write(f"    - Dice Score: {final_val.get('val_seg_dice', 0):.4f}\n")
            f.write(f"    - Mean IoU: {final_val.get('val_seg_mean_iou', 0):.4f}\n")
            
            for c in range(config.model.seg_classes):
                dice_key = f'val_seg_dice_class_{c}'
                iou_key = f'val_seg_iou_class_{c}'
                if dice_key in final_val:
                    f.write(f"    - Class {c} Dice: {final_val[dice_key]:.4f}\n")
                if iou_key in final_val:
                    f.write(f"    - Class {c} IoU: {final_val[iou_key]:.4f}\n")
            
            f.write(f"  🎪 Classification:\n")
            f.write(f"    - Accuracy: {final_val.get('val_cls_accuracy', 0):.4f}\n")
            f.write(f"    - Macro F1: {final_val.get('val_cls_macro_f1', 0):.4f}\n")
            f.write(f"    - Weighted F1: {final_val.get('val_cls_weighted_f1', 0):.4f}\n")
            f.write(f"    - Cohen Kappa: {final_val.get('val_cls_cohen_kappa', 0):.4f}\n")
            f.write(f"    - Matthews Corr: {final_val.get('val_cls_matthews_corr', 0):.4f}\n")
            
            for grade in range(config.model.cls_classes):
                acc_key = f'val_cls_accuracy_grade_{grade}'
                f1_key = f'val_cls_f1_grade_{grade}'
                if acc_key in final_val:
                    f.write(f"    - Grade {grade} Accuracy: {final_val[acc_key]:.4f}\n")
                if f1_key in final_val:
                    f.write(f"    - Grade {grade} F1: {final_val[f1_key]:.4f}\n")
    
    print(f"📋 Detailed report saved to: {report_path}")
    return report_path 