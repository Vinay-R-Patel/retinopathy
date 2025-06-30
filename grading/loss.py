import torch
import torch.nnasnn
import torch.nn.functionalasF
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=5, reduction='mean'):
        super().__init__()
self.alpha= alpha
self.gamma= gamma
self.num_classes= num_classes
self.reduction= reduction


if isinstance(alpha,(float, int)):
            self.alpha= torch.ones(num_classes)*alpha
elif isinstance(alpha, list):
            self.alpha= torch.tensor(alpha, dtype= torch.float32)
else:
            self.alpha= alpha

def forward(self, pred, target):
        if self.alphais not None:
            if self.alpha.device!= pred.device:
                self.alpha= self.alpha.to(pred.device)
alpha_t= self.alpha[target]
else:
            alpha_t=1.0

ce_loss= F.cross_entropy(pred, target, reduction='none')
pt= torch.exp(-ce_loss)
focal_loss= alpha_t*(1-pt)**self.gamma*ce_loss

if self.reduction=='mean':
            return focal_loss.mean()
elif self.reduction=='sum':
            return focal_loss.sum()
else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, num_classes=5):
        super().__init__()
self.smoothing= smoothing
self.num_classes= num_classes

def forward(self, pred, target):
        confidence=1.0-self.smoothing
logprobs= F.log_softmax(pred, dim=-1)
nll_loss=-logprobs.gather(dim=-1, index= target.unsqueeze(1))
nll_loss= nll_loss.squeeze(1)
smooth_loss=-logprobs.mean(dim=-1)
loss= confidence*nll_loss+self.smoothing*smooth_loss
return loss.mean()


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights= None, num_classes=5):
        super().__init__()
self.num_classes= num_classes
if class_weightsis not None:
            self.class_weights= torch.tensor(class_weights, dtype= torch.float32)
else:
            self.class_weights= None

def forward(self, pred, target):
        if self.class_weightsis not None:
            if self.class_weights.device!= pred.device:
                self.class_weights= self.class_weights.to(pred.device)
return F.cross_entropy(pred, target, weight= self.class_weights)
else:
            return F.cross_entropy(pred, target)


class PolyLoss(nn.Module):
    def __init__(self, epsilon=1.0, num_classes=5):
        super().__init__()
self.epsilon= epsilon
self.num_classes= num_classes

def forward(self, pred, target):

        ce_loss= F.cross_entropy(pred, target, reduction='none')


pt= torch.exp(-ce_loss)


poly_loss= ce_loss+self.epsilon*(1-pt)

return poly_loss.mean()


class OnlineHardExampleMiningLoss(nn.Module):
    def __init__(self, ratio=0.25, num_classes=5):
        super().__init__()
self.ratio= ratio
self.num_classes= num_classes

def forward(self, pred, target):

        ce_loss= F.cross_entropy(pred, target, reduction='none')


batch_size= pred.size(0)
num_hard= max(1, int(batch_size*self.ratio))


hard_losses, _= torch.topk(ce_loss, num_hard)

return hard_losses.mean()


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
self.gamma_neg= gamma_neg
self.gamma_pos= gamma_pos
self.clip= clip
self.eps= eps

def forward(self, pred, target):

        num_classes= pred.size(1)
target_one_hot= F.one_hot(target, num_classes).float()


pred_sigmoid= torch.sigmoid(pred)


if self.clipis not Noneandself.clip>0:
            pred_sigmoid= torch.clamp(pred_sigmoid, self.clip,1.0-self.clip)


pos_loss= target_one_hot*torch.log(pred_sigmoid+self.eps)*torch.pow(1-pred_sigmoid, self.gamma_pos)
neg_loss=(1-target_one_hot)*torch.log(1-pred_sigmoid+self.eps)*torch.pow(pred_sigmoid, self.gamma_neg)

loss=-(pos_loss+neg_loss)
return loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, ce_weight=0.5, focal_weight=0.3, smooth_weight=0.2,
gamma=2.0, smoothing=0.1, num_classes=5):
        super().__init__()
self.alpha= alpha
self.ce_weight= ce_weight
self.focal_weight= focal_weight
self.smooth_weight= smooth_weight

self.ce_loss= nn.CrossEntropyLoss()
self.focal_loss= FocalLoss(alpha= alpha, gamma= gamma, num_classes= num_classes)
self.smooth_loss= LabelSmoothingCrossEntropy(smoothing= smoothing, num_classes= num_classes)

def forward(self, pred, target):
        ce= self.ce_loss(pred, target)
focal= self.focal_loss(pred, target)
smooth= self.smooth_loss(pred, target)

return self.ce_weight*ce+self.focal_weight*focal+self.smooth_weight*smooth


class DistributionFocalLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, num_classes=5):
        super().__init__()
self.alpha= alpha
self.beta= beta
self.num_classes= num_classes

def forward(self, pred, target):

        ce_loss= F.cross_entropy(pred, target, reduction='none')


p= F.softmax(pred, dim=1)


p_t= p.gather(1, target.unsqueeze(1)).squeeze(1)


focal_term=(1-p_t)**self.alpha


loss= focal_term*ce_loss

return loss.mean()


def create_criterion(cfg):
    loss_config= cfg.loss.lossif hasattr(cfg.loss,'loss')elsecfg.loss
loss_name= loss_config.name
num_classes= getattr(cfg.model,'num_classes',5)

if loss_name=="crossentropy":
        if hasattr(loss_config,'label_smoothing')andloss_config.label_smoothing>0:
            criterion= LabelSmoothingCrossEntropy(
smoothing= loss_config.label_smoothing,
num_classes= num_classes
)
elif hasattr(loss_config,'class_weights')andloss_config.class_weights:
            criterion= WeightedCrossEntropyLoss(
class_weights= loss_config.class_weights,
num_classes= num_classes
)
else:
            criterion= nn.CrossEntropyLoss()

elif loss_name=="focal":
        alpha= getattr(loss_config,'alpha',0.25)
gamma= getattr(loss_config,'gamma',2.0)
criterion= FocalLoss(alpha= alpha, gamma= gamma, num_classes= num_classes)

elif loss_name=="label_smoothing":
        smoothing= getattr(loss_config,'smoothing',0.1)
criterion= LabelSmoothingCrossEntropy(smoothing= smoothing, num_classes= num_classes)

elif loss_name=="poly":
        epsilon= getattr(loss_config,'epsilon',1.0)
criterion= PolyLoss(epsilon= epsilon, num_classes= num_classes)

elif loss_name=="ohem":
        ratio= getattr(loss_config,'ratio',0.25)
criterion= OnlineHardExampleMiningLoss(ratio= ratio, num_classes= num_classes)

elif loss_name=="asymmetric":
        gamma_neg= getattr(loss_config,'gamma_neg',4)
gamma_pos= getattr(loss_config,'gamma_pos',1)
clip= getattr(loss_config,'clip',0.05)
criterion= AsymmetricLoss(gamma_neg= gamma_neg, gamma_pos= gamma_pos, clip= clip)

elif loss_name=="combined":
        alpha= getattr(loss_config,'alpha',0.5)
ce_weight= getattr(loss_config,'ce_weight',0.5)
focal_weight= getattr(loss_config,'focal_weight',0.3)
smooth_weight= getattr(loss_config,'smooth_weight',0.2)
gamma= getattr(loss_config,'gamma',2.0)
smoothing= getattr(loss_config,'smoothing',0.1)
criterion= CombinedLoss(
alpha= alpha, ce_weight= ce_weight, focal_weight= focal_weight,
smooth_weight= smooth_weight, gamma= gamma, smoothing= smoothing,
num_classes= num_classes
)

elif loss_name=="distribution_focal":
        alpha= getattr(loss_config,'alpha',1.0)
beta= getattr(loss_config,'beta',1.0)
criterion= DistributionFocalLoss(alpha= alpha, beta= beta, num_classes= num_classes)

else:
        raiseValueError(f"Unknown loss function: {loss_name}")

return criterion