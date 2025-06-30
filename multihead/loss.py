import torch
import torch.nn as nn
import torch.nn.functional as F


def multiclass_dice_score(pred, target, num_classes=3, smooth=1):
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


def accuracy_score(pred, target):
    pred = torch.argmax(pred, dim=1)
    correct = (pred == target).float()
    return correct.mean()


class SegmentationLoss(nn.Module):
    def __init__(self, alpha=0.5, num_classes=3):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        dice_loss_val = 1 - multiclass_dice_score(pred, target, self.num_classes)
        return self.alpha * ce_loss + (1 - self.alpha) * dice_loss_val


class ClassificationLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class EnhancedFocalLoss(nn.Module):
    def __init__(self, alpha = None, gamma=2.0, num_classes=5):
        super().__init__()
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        else:
            self.alpha = torch.tensor(alpha, dtype = torch.float32)
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)

        # Get alpha for each target
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
        self.focal_loss = EnhancedFocalLoss(gamma = gamma, num_classes = num_classes)
        self.label_smooth_loss = LabelSmoothingCrossEntropy(smoothing = smoothing, num_classes = num_classes)

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

        # Create smoothed labels
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


def create_criterion(cfg):
    # Create segmentation criterion
    seg_loss_name = getattr(cfg.segmentation_loss, 'name', 'combined')

    if seg_loss_name == "combined":
        seg_criterion = SegmentationLoss(
            alpha = getattr(cfg.segmentation_loss, 'alpha', 0.5),
            num_classes = getattr(cfg.model, 'seg_classes', 3)
        )
    elif seg_loss_name == "focal_dice_tversky":
        seg_criterion = FocalDiceTverskyLoss(
            focal_alpha = getattr(cfg.segmentation_loss, 'focal_alpha', 1),
            focal_gamma = getattr(cfg.segmentation_loss, 'focal_gamma', 2),
            tversky_alpha = getattr(cfg.segmentation_loss, 'tversky_alpha', 0.5),
            tversky_beta = getattr(cfg.segmentation_loss, 'tversky_beta', 0.5),
            focal_weight = getattr(cfg.segmentation_loss, 'focal_weight', 0.33),
            dice_weight = getattr(cfg.segmentation_loss, 'dice_weight', 0.33),
            tversky_weight = getattr(cfg.segmentation_loss, 'tversky_weight', 0.34),
            num_classes = getattr(cfg.model, 'seg_classes', 3)
        )
    else:
        seg_criterion = nn.CrossEntropyLoss()

    # Create classification criterion
    cls_loss_name = getattr(cfg.classification_loss, 'name', 'focal')

    if cls_loss_name == "focal":
        cls_criterion = ClassificationLoss(
            alpha = getattr(cfg.classification_loss, 'alpha', 0.25),
            gamma = getattr(cfg.classification_loss, 'gamma', 2.0)
        )
    elif cls_loss_name == "enhanced_focal":
        alpha_weights = getattr(cfg.classification_loss, 'alpha_weights', None)
        cls_criterion = EnhancedFocalLoss(
            alpha = alpha_weights,
            gamma = getattr(cfg.classification_loss, 'gamma', 2.0),
            num_classes = getattr(cfg.model, 'cls_classes', 5)
        )
    elif cls_loss_name == "combined":
        cls_criterion = CombinedClassificationLoss(
            ce_weight = getattr(cfg.classification_loss, 'ce_weight', 0.4),
            focal_weight = getattr(cfg.classification_loss, 'focal_weight', 0.4),
            smooth_weight = getattr(cfg.classification_loss, 'smooth_weight', 0.2),
            gamma = getattr(cfg.classification_loss, 'gamma', 2.0),
            smoothing = getattr(cfg.classification_loss, 'smoothing', 0.1),
            num_classes = getattr(cfg.model, 'cls_classes', 5)
        )
    elif cls_loss_name == "label_smoothing":
        cls_criterion = LabelSmoothingCrossEntropy(
            smoothing = getattr(cfg.classification_loss, 'smoothing', 0.1),
            num_classes = getattr(cfg.model, 'cls_classes', 5)
        )
    else:
        cls_criterion = nn.CrossEntropyLoss()

    return seg_criterion, cls_criterion