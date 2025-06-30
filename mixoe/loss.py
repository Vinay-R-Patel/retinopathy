from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyLoss(nn.Module):
    def __init__(self, seg_criterion, cls_criterion):
        super().__init__()
        self.seg_loss = seg_criterion
        self.cls_loss = cls_criterion

        self.log_vars = nn.Parameter(torch.zeros(2))

    def forward(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        l_seg = self.seg_loss(preds["seg"], targets["seg"])
        l_cls = self.cls_loss(preds["cls"], targets["cls"])

        precision = torch.exp(-self.log_vars)

        total_loss = (
            precision[0] * l_seg + self.log_vars[0] +
            precision[1] * l_cls + self.log_vars[1]
        ) * 0.5

        return total_loss, {
            "seg": l_seg.detach(),
            "cls": l_cls.detach(),
            "seg_weight": precision[0].detach(),
            "cls_weight": precision[1].detach()
        }


class EnhancedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, num_classes=5):
        super().__init__()
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        else:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)

        alpha_t = self.alpha.to(pred.device)[target]
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


class CombinedClassificationLoss(nn.Module):
    def __init__(self, ce_weight=0.4, focal_weight=0.4, smooth_weight=0.2,
                 gamma=2.0, smoothing=0.1, num_classes=5):
        super().__init__()
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.smooth_weight = smooth_weight

        self.ce_loss = nn.CrossEntropyLoss()
        self.focal_loss = EnhancedFocalLoss(gamma=gamma, num_classes=num_classes)
        self.label_smooth_loss = LabelSmoothingCrossEntropy(smoothing=smoothing, num_classes=num_classes)

    def forward(self, pred, target):
        ce_loss = self.ce_loss(pred, target)
        focal_loss = self.focal_loss(pred, target)
        smooth_loss = self.label_smooth_loss(pred, target)

        total_loss = (self.ce_weight * ce_loss +
                      self.focal_weight * focal_loss +
                      self.smooth_weight * smooth_loss)

        return total_loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, num_classes=5):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, pred, target):
        log_prob = F.log_softmax(pred, dim=1)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        return torch.mean(torch.sum(-true_dist * log_prob, dim=1))


class FocalDiceTverskyLoss(nn.Module):
    def __init__(self, focal_alpha=1, focal_gamma=2, tversky_alpha=0.5, tversky_beta=0.5,
                 focal_weight=0.33, dice_weight=0.33, tversky_weight=0.34, num_classes=3):
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        self.num_classes = num_classes

    def focal_loss(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()

    def dice_loss(self, pred, target):
        pred = F.softmax(pred, dim=1)
        dice_scores = []

        for cls in range(self.num_classes):
            pred_cls = pred[:, cls, :, :]
            target_cls = (target == cls).float()

            intersection = (pred_cls * target_cls).sum(dim=(1, 2))
            union = pred_cls.sum(dim=(1, 2)) + target_cls.sum(dim=(1, 2))

            dice = (2 * intersection + 1) / (union + 1)
            dice_scores.append(dice.mean())

        return 1 - torch.stack(dice_scores).mean()

    def tversky_loss(self, pred, target):
        pred = F.softmax(pred, dim=1)
        tversky_scores = []

        for cls in range(self.num_classes):
            pred_cls = pred[:, cls, :, :]
            target_cls = (target == cls).float()

            tp = (pred_cls * target_cls).sum(dim=(1, 2))
            fp = (pred_cls * (1 - target_cls)).sum(dim=(1, 2))
            fn = ((1 - pred_cls) * target_cls).sum(dim=(1, 2))

            tversky = (tp + 1) / (tp + self.tversky_alpha * fp + self.tversky_beta * fn + 1)
            tversky_scores.append(tversky.mean())

        return 1 - torch.stack(tversky_scores).mean()

    def forward(self, pred, target):
        focal_loss = self.focal_loss(pred, target)
        dice_loss = self.dice_loss(pred, target)
        tversky_loss = self.tversky_loss(pred, target)

        total_loss = (self.focal_weight * focal_loss +
                      self.dice_weight * dice_loss +
                      self.tversky_weight * tversky_loss)

        return total_loss


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    pred = torch.softmax(pred, dim=1)
    num_classes = pred.size(1)

    target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

    intersection = torch.sum(pred * target_one_hot, dim=(2, 3))
    union = torch.sum(pred + target_one_hot, dim=(2, 3))

    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def dice_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int, smooth: float = 1e-5) -> torch.Tensor:
    pred = torch.softmax(pred, dim=1)
    dice_scores = []

    for cls in range(num_classes):
        pred_cls = pred[:, cls, :, :]
        target_cls = (target == cls).float()

        intersection = (pred_cls * target_cls).sum(dim=(1, 2))
        union = pred_cls.sum(dim=(1, 2)) + target_cls.sum(dim=(1, 2))

        dice = (2 * intersection + smooth) / (union + smooth)
        dice_scores.append(dice.mean())

    return torch.stack(dice_scores).mean()


def focal_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    ce_loss = F.cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()


def create_loss_functions(config):
    """Create segmentation and classification loss functions based on config"""

    seg_loss_name = getattr(config.loss, 'seg_loss_name', 'dice')

    if seg_loss_name == "focal_dice_tversky":
        seg_criterion = FocalDiceTverskyLoss(
            focal_alpha=getattr(config.loss, 'focal_alpha', 1),
            focal_gamma=getattr(config.loss, 'focal_gamma', 2),
            tversky_alpha=getattr(config.loss, 'tversky_alpha', 0.5),
            tversky_beta=getattr(config.loss, 'tversky_beta', 0.5),
            focal_weight=getattr(config.loss, 'focal_weight', 0.33),
            dice_weight=getattr(config.loss, 'dice_weight', 0.33),
            tversky_weight=getattr(config.loss, 'tversky_weight', 0.34),
            num_classes=config.model.seg_classes
        )
    else:
        seg_criterion = lambda pred, target: dice_loss(pred, target)

    cls_loss_name = getattr(config.loss, 'cls_loss_name', 'focal')

    if cls_loss_name == "enhanced_focal":
        alpha_weights = getattr(config.loss, 'alpha_weights', None)
        cls_criterion = EnhancedFocalLoss(
            alpha=alpha_weights,
            gamma=getattr(config.loss, 'gamma', 2.0),
            num_classes=config.model.cls_classes
        )
    elif cls_loss_name == "combined":
        cls_criterion = CombinedClassificationLoss(
            ce_weight=getattr(config.loss, 'ce_weight', 0.4),
            focal_weight=getattr(config.loss, 'focal_weight', 0.4),
            smooth_weight=getattr(config.loss, 'smooth_weight', 0.2),
            gamma=getattr(config.loss, 'gamma', 2.0),
            smoothing=getattr(config.loss, 'smoothing', 0.1),
            num_classes=config.model.cls_classes
        )
    elif cls_loss_name == "label_smoothing":
        cls_criterion = LabelSmoothingCrossEntropy(
            smoothing=getattr(config.loss, 'smoothing', 0.1),
            num_classes=config.model.cls_classes
        )
    else:
        cls_criterion = lambda pred, target: focal_loss(
            pred, target,
            alpha=getattr(config.loss, 'alpha', 0.25),
            gamma=getattr(config.loss, 'gamma', 2.0)
        )

    if getattr(config.loss, 'use_uncertainty', True):
        criterion = UncertaintyLoss(seg_criterion, cls_criterion)
    else:
        seg_weight = getattr(config.loss, 'seg_weight', 0.5)
        cls_weight = getattr(config.loss, 'cls_weight', 0.5)

        def combined_criterion(preds, targets):
            seg_loss = seg_criterion(preds["seg"], targets["seg"])
            cls_loss = cls_criterion(preds["cls"], targets["cls"])
            total_loss = seg_weight * seg_loss + cls_weight * cls_loss

            return total_loss, {
                "seg": seg_loss.detach(),
                "cls": cls_loss.detach(),
                "seg_weight": seg_weight,
                "cls_weight": cls_weight
            }

        criterion = combined_criterion

    return criterion