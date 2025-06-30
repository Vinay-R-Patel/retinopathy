import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_fscore_support,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    cohen_kappa_score, matthews_corrcoef
)
import seaborn as sns
import wandb
from collections import defaultdict
import json


def accuracy_score(pred, target):
    pred = torch.argmax(pred, dim=1)
    correct = (pred == target).float()
    return correct.mean()


def top_k_accuracy(pred, target, k=2):
    """Calculate top-k accuracy"""
    _, pred_k = torch.topk(pred, k, dim=1)
    target_expanded = target.unsqueeze(1).expand_as(pred_k)
    correct = (pred_k == target_expanded).any(dim=1).float()
    return correct.mean()


def class_wise_accuracy(pred, target, num_classes=5):
    """Calculate per-class accuracy"""
    pred = torch.argmax(pred, dim=1)
    class_accuracies = []

    for cls in range(num_classes):
        cls_mask = (target == cls)
        if cls_mask.sum() > 0:
            cls_correct = (pred[cls_mask] == target[cls_mask]).float().mean()
            class_accuracies.append(cls_correct.item())
        else:
            class_accuracies.append(0.0)

    return class_accuracies


def calculate_comprehensive_metrics(pred_probs, pred_classes, target, num_classes=5):
    """Calculate comprehensive metrics for classification"""
    target_np = target.detach().cpu().numpy()
    pred_classes_np = pred_classes.detach().cpu().numpy()
    pred_probs_np = pred_probs.detach().cpu().numpy()

    precision, recall, f1, support = precision_recall_fscore_support(
        target_np, pred_classes_np, average=None, zero_division=0
    )

    precision_macro = precision_recall_fscore_support(target_np, pred_classes_np, average='macro', zero_division=0)[0]
    recall_macro = precision_recall_fscore_support(target_np, pred_classes_np, average='macro', zero_division=0)[1]
    f1_macro = precision_recall_fscore_support(target_np, pred_classes_np, average='macro', zero_division=0)[2]

    precision_micro = precision_recall_fscore_support(target_np, pred_classes_np, average='micro', zero_division=0)[0]
    recall_micro = precision_recall_fscore_support(target_np, pred_classes_np, average='micro', zero_division=0)[1]
    f1_micro = precision_recall_fscore_support(target_np, pred_classes_np, average='micro', zero_division=0)[2]

    precision_weighted = precision_recall_fscore_support(target_np, pred_classes_np, average='weighted', zero_division=0)[0]
    recall_weighted = precision_recall_fscore_support(target_np, pred_classes_np, average='weighted', zero_division=0)[1]
    f1_weighted = precision_recall_fscore_support(target_np, pred_classes_np, average='weighted', zero_division=0)[2]

    balanced_acc = balanced_accuracy_score(target_np, pred_classes_np)
    kappa = cohen_kappa_score(target_np, pred_classes_np)
    mcc = matthews_corrcoef(target_np, pred_classes_np)

    try:
        auc_scores = []
        for i in range(num_classes):
            if len(np.unique(target_np)) > 1:
                y_true_binary = (target_np == i).astype(int)
                if np.sum(y_true_binary) > 0 and np.sum(1 - y_true_binary) > 0:
                    auc = roc_auc_score(y_true_binary, pred_probs_np[:, i])
                    auc_scores.append(auc)
                else:
                    auc_scores.append(0.5)
            else:
                auc_scores.append(0.5)
        auc_macro = np.mean(auc_scores)
    except:
        auc_scores = [0.5] * num_classes
        auc_macro = 0.5

    class_accuracies = []
    for cls in range(num_classes):
        cls_mask = (target_np == cls)
        if np.sum(cls_mask) > 0:
            cls_acc = np.mean(pred_classes_np[cls_mask] == target_np[cls_mask])
            class_accuracies.append(cls_acc)
        else:
            class_accuracies.append(0.0)

    return {
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
        'f1_per_class': f1.tolist(),
        'support_per_class': support.tolist(),
        'accuracy_per_class': class_accuracies,
        'auc_per_class': auc_scores,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'balanced_accuracy': balanced_acc,
        'cohen_kappa': kappa,
        'matthews_corrcoef': mcc,
        'auc_macro': auc_macro,
    }


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device, cfg):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.cfg = cfg

        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.best_acc = 0
        self.best_f1 = 0
        self.best_balanced_acc = 0

        self.train_metrics_history = []
        self.val_metrics_history = []
        self.class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
        self.num_classes = len(self.class_names)

        self.learning_rates = []
        self.gradient_norms = []

        self.train_class_accs_history = {cls: [] for cls in range(self.num_classes)}
        self.val_class_accs_history = {cls: [] for cls in range(self.num_classes)}
        self.val_class_f1_history = {cls: [] for cls in range(self.num_classes)}

        os.makedirs(cfg.output_dir, exist_ok=True)

        self.metrics_dir = os.path.join(cfg.output_dir, 'metrics')
        os.makedirs(self.metrics_dir, exist_ok=True)

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        total_acc = 0
        all_preds = []
        all_probs = []
        all_labels = []

        gradient_norms = []

        pbar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)

            loss = self.criterion(outputs, labels)
            loss.backward()

            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            gradient_norms.append(total_norm)

            self.optimizer.step()

            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            acc = accuracy_score(outputs, labels)
            top2_acc = top_k_accuracy(outputs, labels, k=2)

            total_loss += loss.item()
            total_acc += acc.item()

            all_preds.extend(preds.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{acc.item():.4f}',
                'Top2': f'{top2_acc.item():.4f}',
                'GradNorm': f'{total_norm:.2f}'
            })

        avg_loss = total_loss / len(dataloader)
        avg_acc = total_acc / len(dataloader)
        avg_grad_norm = np.mean(gradient_norms)

        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        train_metrics = calculate_comprehensive_metrics(
            torch.tensor(all_probs), torch.tensor(all_preds), torch.tensor(all_labels), self.num_classes
        )
        train_metrics['loss'] = avg_loss
        train_metrics['accuracy'] = avg_acc
        train_metrics['gradient_norm'] = avg_grad_norm

        class_accs = class_wise_accuracy(
            torch.tensor(all_probs), torch.tensor(all_labels), self.num_classes
        )

        return avg_loss, avg_acc, train_metrics, class_accs

    def validate_epoch(self, dataloader, epoch):
        self.model.eval()
        total_loss = 0
        total_acc = 0
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f"Validation Epoch {epoch+1}")
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                acc = accuracy_score(outputs, labels)
                top2_acc = top_k_accuracy(outputs, labels, k=2)

                total_loss += loss.item()
                total_acc += acc.item()

                all_preds.extend(preds.detach().cpu().numpy())
                all_probs.extend(probs.detach().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{acc.item():.4f}',
                    'Top2': f'{top2_acc.item():.4f}'
                })

        avg_loss = total_loss / len(dataloader)
        avg_acc = total_acc / len(dataloader)

        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        val_metrics = calculate_comprehensive_metrics(
            torch.tensor(all_probs), torch.tensor(all_preds), torch.tensor(all_labels), self.num_classes
        )
        val_metrics['loss'] = avg_loss
        val_metrics['accuracy'] = avg_acc

        class_accs = class_wise_accuracy(
            torch.tensor(all_probs), torch.tensor(all_labels), self.num_classes
        )

        return avg_loss, avg_acc, val_metrics, class_accs, all_preds, all_labels

    def train(self, train_loader, val_loader):
        for epoch in range(self.cfg.training.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.cfg.training.num_epochs}")
            print("=" * 80)

            train_loss, train_acc, train_metrics, train_class_accs = self.train_epoch(train_loader, epoch)

            val_loss, val_acc, val_metrics, val_class_accs, val_preds, val_labels = self.validate_epoch(val_loader, epoch)

            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            self.train_metrics_history.append(train_metrics)
            self.val_metrics_history.append(val_metrics)

            for cls in range(self.num_classes):
                self.train_class_accs_history[cls].append(train_class_accs[cls])
                self.val_class_accs_history[cls].append(val_class_accs[cls])
                self.val_class_f1_history[cls].append(val_metrics['f1_per_class'][cls])

            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            self.gradient_norms.append(train_metrics['gradient_norm'])

            self.print_epoch_summary(epoch, train_metrics, val_metrics, current_lr)

            if self.cfg.logging.use_wandb:
                self.log_to_wandb(epoch, train_metrics, val_metrics, current_lr)

            is_best_acc = val_acc > self.best_acc
            is_best_f1 = val_metrics['f1_macro'] > self.best_f1
            is_best_balanced = val_metrics['balanced_accuracy'] > self.best_balanced_acc

            if is_best_acc:
                self.best_acc = val_acc
                self.save_checkpoint(epoch, is_best=True, metric_name='accuracy')
                print(f"🎯 New best accuracy: {self.best_acc:.4f}")

            if is_best_f1:
                self.best_f1 = val_metrics['f1_macro']
                self.save_checkpoint(epoch, is_best=True, metric_name='f1_macro')
                print(f"🏆 New best F1-score: {self.best_f1:.4f}")

            if is_best_balanced:
                self.best_balanced_acc = val_metrics['balanced_accuracy']
                self.save_checkpoint(epoch, is_best=True, metric_name='balanced_accuracy')
                print(f"⚖️ New best balanced accuracy: {self.best_balanced_acc:.4f}")

            if (epoch + 1) % self.cfg.training.save_freq == 0:
                self.save_epoch_metrics(epoch, train_metrics, val_metrics)
                self.save_confusion_matrix(val_labels, val_preds, epoch)
                self.save_checkpoint(epoch, is_best=False)

            if epoch == self.cfg.training.num_epochs - 1:
                self.generate_final_report(val_labels, val_preds, val_metrics)

        self.save_comprehensive_plots()
        self.save_metrics_summary()

        return {
            'best_accuracy': self.best_acc,
            'best_f1_macro': self.best_f1,
            'best_balanced_accuracy': self.best_balanced_acc
        }

    def print_epoch_summary(self, epoch, train_metrics, val_metrics, current_lr):
        """Print comprehensive epoch summary"""
        print(f"📊 Training Metrics:")
        print(f"   Loss: {train_metrics['loss']:.4f} | Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"   F1-Macro: {train_metrics['f1_macro']:.4f} | Balanced Acc: {train_metrics['balanced_accuracy']:.4f}")
        print(f"   Gradient Norm: {train_metrics['gradient_norm']:.3f}")

        print(f"📈 Validation Metrics:")
        print(f"   Loss: {val_metrics['loss']:.4f} | Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"   F1-Macro: {val_metrics['f1_macro']:.4f} | Balanced Acc: {val_metrics['balanced_accuracy']:.4f}")
        print(f"   AUC-Macro: {val_metrics['auc_macro']:.4f} | Cohen's Kappa: {val_metrics['cohen_kappa']:.4f}")

        print(f"🎯 Per-Class F1 Scores:")
        for i, (class_name, f1_score) in enumerate(zip(self.class_names, val_metrics['f1_per_class'])):
            print(f"   {class_name}: {f1_score:.4f}")

        print(f"🔧 Learning Rate: {current_lr:.6f}")
        print("-" * 80)

    def log_to_wandb(self, epoch, train_metrics, val_metrics, current_lr):
        """Comprehensive Wandb logging"""
        log_dict = {
            "epoch": epoch + 1,
            "learning_rate": current_lr,
            "train/loss": train_metrics['loss'],
            "train/accuracy": train_metrics['accuracy'],
            "train/f1_macro": train_metrics['f1_macro'],
            "train/balanced_accuracy": train_metrics['balanced_accuracy'],
            "train/gradient_norm": train_metrics['gradient_norm'],
            "val/loss": val_metrics['loss'],
            "val/accuracy": val_metrics['accuracy'],
            "val/f1_macro": val_metrics['f1_macro'],
            "val/f1_micro": val_metrics['f1_micro'],
            "val/f1_weighted": val_metrics['f1_weighted'],
            "val/balanced_accuracy": val_metrics['balanced_accuracy'],
            "val/cohen_kappa": val_metrics['cohen_kappa'],
            "val/matthews_corrcoef": val_metrics['matthews_corrcoef'],
            "val/auc_macro": val_metrics['auc_macro'],
        }

        for i, class_name in enumerate(self.class_names):
            log_dict[f"val/f1_{class_name.lower().replace(' ', '_')}"] = val_metrics['f1_per_class'][i]
            log_dict[f"val/precision_{class_name.lower().replace(' ', '_')}"] = val_metrics['precision_per_class'][i]
            log_dict[f"val/recall_{class_name.lower().replace(' ', '_')}"] = val_metrics['recall_per_class'][i]
            log_dict[f"val/accuracy_{class_name.lower().replace(' ', '_')}"] = val_metrics['accuracy_per_class'][i]
            log_dict[f"val/auc_{class_name.lower().replace(' ', '_')}"] = val_metrics['auc_per_class'][i]

        wandb.log(log_dict)

    def save_epoch_metrics(self, epoch, train_metrics, val_metrics):
        """Save detailed metrics for each epoch"""
        metrics_data = {
            'epoch': epoch + 1,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'timestamp': str(torch.cuda.current_device()) if torch.cuda.is_available() else 'cpu'
        }

        metrics_file = os.path.join(self.metrics_dir, f'epoch_{epoch+1}_metrics.json')
        with open(metrics_file, 'w') as f:
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
                    return float(obj)
                elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
                    return int(obj)
                return obj

            json.dump(metrics_data, f, indent=2, default=convert_numpy)

    def save_confusion_matrix(self, y_true, y_pred, epoch):
        """Save confusion matrix for current epoch"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)

        if epoch == -1:
            plt.title('Final Confusion Matrix')
            cm_path = os.path.join(self.cfg.output_dir, 'final_confusion_matrix.png')
            wandb_key = "final_confusion_matrix"
        else:
            plt.title(f'Confusion Matrix - Epoch {epoch+1}')
            cm_path = os.path.join(self.metrics_dir, f'confusion_matrix_epoch_{epoch+1}.png')
            wandb_key = f"confusion_matrix_epoch_{epoch+1}"

        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()

        if self.cfg.logging.use_wandb:
            wandb.log({wandb_key: wandb.Image(cm_path)})

    def save_checkpoint(self, epoch, is_best=False, metric_name='accuracy'):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': self.best_acc,
            'best_f1': self.best_f1,
            'best_balanced_acc': self.best_balanced_acc,
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
            'train_metrics_history': self.train_metrics_history,
            'val_metrics_history': self.val_metrics_history,
            'config': self.cfg
        }

        if is_best:
            checkpoint_path = os.path.join(self.cfg.output_dir, f'best_model_{metric_name}.pth')
        else:
            checkpoint_path = os.path.join(self.cfg.output_dir, f'checkpoint_epoch_{epoch+1}.pth')

        torch.save(checkpoint, checkpoint_path)

    def generate_final_report(self, y_true, y_pred, val_metrics):
        """Generate comprehensive final training report"""
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)

        report_path = os.path.join(self.cfg.output_dir, 'final_classification_report.txt')
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("COMPREHENSIVE TRAINING REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Model: {self.cfg.model.name}\n")
            f.write(f"Image Size: {self.cfg.image_size}\n")
            f.write(f"Batch Size: {self.cfg.training.batch_size}\n")
            f.write(f"Total Epochs: {self.cfg.training.num_epochs}\n")
            f.write(f"Loss Function: {self.cfg.loss.name}\n\n")

            f.write("BEST METRICS ACHIEVED:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Best Accuracy: {self.best_acc:.4f}\n")
            f.write(f"Best F1-Macro: {self.best_f1:.4f}\n")
            f.write(f"Best Balanced Accuracy: {self.best_balanced_acc:.4f}\n\n")

            f.write("FINAL VALIDATION METRICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Accuracy: {val_metrics['accuracy']:.4f}\n")
            f.write(f"Balanced Accuracy: {val_metrics['balanced_accuracy']:.4f}\n")
            f.write(f"F1-Macro: {val_metrics['f1_macro']:.4f}\n")
            f.write(f"F1-Micro: {val_metrics['f1_micro']:.4f}\n")
            f.write(f"F1-Weighted: {val_metrics['f1_weighted']:.4f}\n")
            f.write(f"Cohen's Kappa: {val_metrics['cohen_kappa']:.4f}\n")
            f.write(f"Matthews Correlation: {val_metrics['matthews_corrcoef']:.4f}\n")
            f.write(f"AUC-Macro: {val_metrics['auc_macro']:.4f}\n\n")

            f.write("PER-CLASS METRICS:\n")
            f.write("-" * 30 + "\n")
            for i, class_name in enumerate(self.class_names):
                f.write(f"{class_name}:\n")
                f.write(f"  Precision: {val_metrics['precision_per_class'][i]:.4f}\n")
                f.write(f"  Recall: {val_metrics['recall_per_class'][i]:.4f}\n")
                f.write(f"  F1-Score: {val_metrics['f1_per_class'][i]:.4f}\n")
                f.write(f"  Accuracy: {val_metrics['accuracy_per_class'][i]:.4f}\n")
                f.write(f"  AUC: {val_metrics['auc_per_class'][i]:.4f}\n")
                f.write(f"  Support: {val_metrics['support_per_class'][i]}\n\n")

            f.write("\nDETAILED CLASSIFICATION REPORT:\n")
            f.write("-" * 40 + "\n")
            f.write(classification_report(y_true, y_pred, target_names=self.class_names))

        self.save_confusion_matrix(y_true, y_pred, epoch=-1)

        if self.cfg.logging.use_wandb:
            wandb.log({
                "final_classification_report": report,
                "final_best_accuracy": self.best_acc,
                "final_best_f1_macro": self.best_f1,
                "final_best_balanced_accuracy": self.best_balanced_acc
            })

    def save_comprehensive_plots(self):
        """Save comprehensive training visualization plots"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Comprehensive Training Analysis', fontsize=16, fontweight='bold')

        epochs = range(1, len(self.train_losses) + 1)

        axes[0, 0].plot(epochs, self.train_losses, label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.val_losses, label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Training vs Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(epochs, self.train_accs, label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, self.val_accs, label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_title('Training vs Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        val_f1_scores = [metrics['f1_macro'] for metrics in self.val_metrics_history]
        axes[0, 2].plot(epochs, val_f1_scores, label='F1-Macro', linewidth=2, color='green')
        axes[0, 2].set_title('F1-Score Over Time')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('F1-Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        axes[1, 0].plot(epochs, self.learning_rates, linewidth=2, color='orange')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(epochs, self.gradient_norms, linewidth=2, color='red')
        axes[1, 1].set_title('Gradient Norms')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Gradient Norm')
        axes[1, 1].grid(True, alpha=0.3)

        for cls in range(self.num_classes):
            axes[1, 2].plot(epochs, self.val_class_accs_history[cls],
                            label=self.class_names[cls], linewidth=2)
        axes[1, 2].set_title('Per-Class Validation Accuracy')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Accuracy')
        axes[1, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 2].grid(True, alpha=0.3)

        for cls in range(self.num_classes):
            axes[2, 0].plot(epochs, self.val_class_f1_history[cls],
                            label=self.class_names[cls], linewidth=2)
        axes[2, 0].set_title('Per-Class F1 Scores')
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('F1-Score')
        axes[2, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[2, 0].grid(True, alpha=0.3)

        balanced_accs = [metrics['balanced_accuracy'] for metrics in self.val_metrics_history]
        axes[2, 1].plot(epochs, balanced_accs, linewidth=2, color='purple')
        axes[2, 1].set_title('Balanced Accuracy Over Time')
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('Balanced Accuracy')
        axes[2, 1].grid(True, alpha=0.3)

        auc_scores = [metrics['auc_macro'] for metrics in self.val_metrics_history]
        axes[2, 2].plot(epochs, auc_scores, linewidth=2, color='brown')
        axes[2, 2].set_title('AUC-Macro Over Time')
        axes[2, 2].set_xlabel('Epoch')
        axes[2, 2].set_ylabel('AUC-Macro')
        axes[2, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plots_path = os.path.join(self.cfg.output_dir, 'comprehensive_training_analysis.png')
        plt.savefig(plots_path, dpi=300, bbox_inches='tight')
        plt.close()

        if self.cfg.logging.use_wandb:
            wandb.log({"comprehensive_training_analysis": wandb.Image(plots_path)})

    def save_metrics_summary(self):
        """Save comprehensive metrics summary as JSON"""
        summary = {
            'experiment_config': {
                'model_name': self.cfg.model.name,
                'image_size': self.cfg.image_size,
                'batch_size': self.cfg.training.batch_size,
                'epochs': self.cfg.training.num_epochs,
                'loss_function': self.cfg.loss.name,
            },
            'best_metrics': {
                'best_accuracy': float(self.best_acc),
                'best_f1_macro': float(self.best_f1),
                'best_balanced_accuracy': float(self.best_balanced_acc),
            },
            'final_metrics': self.val_metrics_history[-1] if self.val_metrics_history else {},
            'training_summary': {
                'total_epochs_trained': len(self.train_losses),
                'final_learning_rate': self.learning_rates[-1] if self.learning_rates else None,
                'avg_gradient_norm': float(np.mean(self.gradient_norms)) if self.gradient_norms else None,
            },
            'class_names': self.class_names,
        }

        summary_path = os.path.join(self.cfg.output_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                return obj

            json.dump(summary, f, indent=2, default=convert_numpy)


def create_optimizer(model, cfg):
    optimizer_name = cfg.optimizer.name.lower()
    lr = cfg.optimizer.lr
    weight_decay = cfg.optimizer.weight_decay

    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = cfg.optimizer.get('momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizer


def create_scheduler(optimizer, cfg):
    if not hasattr(cfg, 'scheduler') or cfg.scheduler.name == 'none':
        return None

    scheduler_name = cfg.scheduler.name.lower()

    if scheduler_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.scheduler.step_size,
            gamma=cfg.scheduler.gamma
        )
    elif scheduler_name == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=cfg.scheduler.patience,
            factor=cfg.scheduler.factor
        )
    elif scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.training.num_epochs
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    return scheduler