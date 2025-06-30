import torch
import torch.nnasnn
import torch.nn.functionalasF


def calculate_class_wise_metrics(pred, target, num_classes=3, smooth=1):
    """
    Calculate comprehensive class-wise metrics including Dice, IoU, Precision, and Recall
    Returns both per-class metrics and overall averages
    """
pred= torch.softmax(pred, dim=1)
pred_classes= torch.argmax(pred, dim=1)

metrics={
'dice_per_class':[],
'iou_per_class':[],
'precision_per_class':[],
'recall_per_class':[]
}

class_names=['background','hard_exudates','haemorrhages']

for clsin range(num_classes):

        pred_cls=(pred_classes== cls).float()
target_cls=(target== cls).float()


tp=(pred_cls*target_cls).sum()
fp=(pred_cls*(1-target_cls)).sum()
fn=((1-pred_cls)*target_cls).sum()
tn=((1-pred_cls)*(1-target_cls)).sum()


dice=(2*tp+smooth)/(2*tp+fp+fn+smooth)


iou=(tp+smooth)/(tp+fp+fn+smooth)


precision=(tp+smooth)/(tp+fp+smooth)


recall=(tp+smooth)/(tp+fn+smooth)

metrics['dice_per_class'].append(dice.item())
metrics['iou_per_class'].append(iou.item())
metrics['precision_per_class'].append(precision.item())
metrics['recall_per_class'].append(recall.item())


metrics['dice_mean']= sum(metrics['dice_per_class'])/len(metrics['dice_per_class'])
metrics['dice_mean_fg']= sum(metrics['dice_per_class'][1:])/len(metrics['dice_per_class'][1:])

metrics['iou_mean']= sum(metrics['iou_per_class'])/len(metrics['iou_per_class'])
metrics['iou_mean_fg']= sum(metrics['iou_per_class'][1:])/len(metrics['iou_per_class'][1:])

metrics['precision_mean']= sum(metrics['precision_per_class'])/len(metrics['precision_per_class'])
metrics['recall_mean']= sum(metrics['recall_per_class'])/len(metrics['recall_per_class'])


for i, class_namein enumerate(class_names):
        metrics[f'dice_{class_name}']= metrics['dice_per_class'][i]
metrics[f'iou_{class_name}']= metrics['iou_per_class'][i]
metrics[f'precision_{class_name}']= metrics['precision_per_class'][i]
metrics[f'recall_{class_name}']= metrics['recall_per_class'][i]

return metrics


def multiclass_dice_score(pred, target, num_classes=3, smooth=1):
    """Legacy function for backward compatibility"""
metrics= calculate_class_wise_metrics(pred, target, num_classes, smooth)
return torch.tensor(metrics['dice_mean'])


def multiclass_dice_loss_dif ferentiable(pred, target, num_classes=3, smooth=1):
    """Dif ferentiable version of dice loss that preserves gradients"""
pred= torch.softmax(pred, dim=1)
total_loss=0

for clsin range(num_classes):
        pred_cls= pred[:, cls]
target_cls=(target== cls).float()

intersection=(pred_cls*target_cls).sum()
union= pred_cls.sum()+target_cls.sum()

dice=(2*intersection+smooth)/(union+smooth)
total_loss+=1-dice

return total_loss/num_classes


def multiclass_dice_loss(pred, target, num_classes=3, smooth=1):
    return multiclass_dice_loss_dif ferentiable(pred, target, num_classes, smooth)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, num_classes=3, reduction='mean'):
        super().__init__()
self.alpha= alpha
self.gamma= gamma
self.num_classes= num_classes
self.reduction= reduction

def forward(self, pred, target):
        ce_loss= F.cross_entropy(pred, target, reduction='none')
pt= torch.exp(-ce_loss)
focal_loss= self.alpha*(1-pt)**self.gamma*ce_loss

if self.reduction=='mean':
            return focal_loss.mean()
elif self.reduction=='sum':
            return focal_loss.sum()
else:
            return focal_loss


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, num_classes=3, smooth=1):
        super().__init__()
self.alpha= alpha
self.beta= beta
self.num_classes= num_classes
self.smooth= smooth

def forward(self, pred, target):
        pred= torch.softmax(pred, dim=1)
total_loss=0

for clsin range(self.num_classes):
            pred_cls= pred[:, cls]
target_cls=(target== cls).float()

tp=(pred_cls*target_cls).sum()
fp=(pred_cls*(1-target_cls)).sum()
fn=((1-pred_cls)*target_cls).sum()

tversky=(tp+self.smooth)/(tp+self.alpha*fp+self.beta*fn+self.smooth)
total_loss+=1-tversky

return total_loss/self.num_classes


class FocalTverskyLoss(nn.Module):
    def __init__(self, focal_alpha=1, focal_gamma=2, tversky_alpha=0.5, tversky_beta=0.5,
loss_alpha=0.5, num_classes=3):
        super().__init__()
self.focal= FocalLoss(alpha= focal_alpha, gamma= focal_gamma, num_classes= num_classes)
self.tversky= TverskyLoss(alpha= tversky_alpha, beta= tversky_beta, num_classes= num_classes)
self.loss_alpha= loss_alpha

def forward(self, pred, target):
        focal_loss= self.focal(pred, target)
tversky_loss= self.tversky(pred, target)
return self.loss_alpha*focal_loss+(1-self.loss_alpha)*tversky_loss


class FocalDiceLoss(nn.Module):
    def __init__(self, focal_alpha=1, focal_gamma=2, loss_alpha=0.5, num_classes=3):
        super().__init__()
self.focal= FocalLoss(alpha= focal_alpha, gamma= focal_gamma, num_classes= num_classes)
self.loss_alpha= loss_alpha
self.num_classes= num_classes

def forward(self, pred, target):
        focal_loss= self.focal(pred, target)
dice_loss= multiclass_dice_loss_dif ferentiable(pred, target, self.num_classes)
return self.loss_alpha*focal_loss+(1-self.loss_alpha)*dice_loss


class ICILoss(nn.Module):
    def __init__(self, num_classes=3, beta=0.9999, gamma=2.0, epsilon=1e-7):
        super().__init__()
self.num_classes= num_classes
self.beta= beta
self.gamma= gamma
self.epsilon= epsilon
self.class_weights= None

def _compute_class_weights(self, target):
        """Compute class weights based on effective number of samples"""

class_counts= torch.zeros(self.num_classes, device= target.device)
for cin range(self.num_classes):
            class_counts[c]=(target== c).sum().float()


class_counts= class_counts+self.epsilon


effective_num=1.0-torch.pow(self.beta, class_counts)


weights=(1.0-self.beta)/effective_num


weights= weights/weights.sum()*self.num_classes

return weights

def forward(self, pred, target):

        class_weights= self._compute_class_weights(target)


pred_softmax= F.softmax(pred, dim=1)


target_one_hot= F.one_hot(target, num_classes= self.num_classes).permute(0,3,1,2).float()


ce_loss= F.cross_entropy(pred, target, weight= class_weights, reduction='none')


pt= torch.exp(-ce_loss)
focal_weight=(1-pt)**self.gamma


ici_loss= focal_weight*ce_loss

return ici_loss.mean()


class FocalDiceTverskyLoss(nn.Module):
    def __init__(self, focal_alpha=1, focal_gamma=2, tversky_alpha=0.5, tversky_beta=0.5,
focal_weight=0.33, dice_weight=0.33, tversky_weight=0.34, num_classes=3):
        super().__init__()
self.focal= FocalLoss(alpha= focal_alpha, gamma= focal_gamma, num_classes= num_classes)
self.tversky= TverskyLoss(alpha= tversky_alpha, beta= tversky_beta, num_classes= num_classes)
self.focal_weight= focal_weight
self.dice_weight= dice_weight
self.tversky_weight= tversky_weight
self.num_classes= num_classes


total_weight= focal_weight+dice_weight+tversky_weight
self.focal_weight= focal_weight/total_weight
self.dice_weight= dice_weight/total_weight
self.tversky_weight= tversky_weight/total_weight

def forward(self, pred, target):
        focal_loss= self.focal(pred, target)
dice_loss= multiclass_dice_loss_dif ferentiable(pred, target, self.num_classes)
tversky_loss= self.tversky(pred, target)

combined_loss=(self.focal_weight*focal_loss+
self.dice_weight*dice_loss+
self.tversky_weight*tversky_loss)

return combined_loss


class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=0.5, num_classes=3):
        super().__init__()
self.alpha= alpha
self.ce= nn.CrossEntropyLoss()
self.num_classes= num_classes

def forward(self, pred, target):
        ce_loss= self.ce(pred, target)
dice_loss_val= multiclass_dice_loss_dif ferentiable(pred, target, self.num_classes)
return self.alpha*dice_loss_val+(1-self.alpha)*ce_loss


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, num_classes=3):
        super().__init__()
self.alpha= alpha
self.ce= nn.CrossEntropyLoss()
self.num_classes= num_classes

def forward(self, pred, target):
        ce_loss= self.ce(pred, target)
dice_loss_val= multiclass_dice_loss_dif ferentiable(pred, target, self.num_classes)
return self.alpha*ce_loss+(1-self.alpha)*dice_loss_val


def create_criterion(cfg):
    if cfg.loss.name=="combined":
        criterion= CombinedLoss(alpha= cfg.loss.alpha, num_classes= cfg.model.num_classes)
elif cfg.loss.name=="focal":
        alpha= getattr(cfg.loss,'alpha',1)
gamma= getattr(cfg.loss,'gamma',2)
criterion= FocalLoss(alpha= alpha, gamma= gamma, num_classes= cfg.model.num_classes)
elif cfg.loss.name=="tversky":
        alpha= getattr(cfg.loss,'alpha',0.5)
beta= getattr(cfg.loss,'beta',0.5)
criterion= TverskyLoss(alpha= alpha, beta= beta, num_classes= cfg.model.num_classes)
elif cfg.loss.name=="dice":
        criterion= lambdapred, target:multiclass_dice_loss_dif ferentiable(pred, target, cfg.model.num_classes)
elif cfg.loss.name=="focal_tversky":
        focal_alpha= getattr(cfg.loss,'focal_alpha',1)
focal_gamma= getattr(cfg.loss,'focal_gamma',2)
tversky_alpha= getattr(cfg.loss,'tversky_alpha',0.5)
tversky_beta= getattr(cfg.loss,'tversky_beta',0.5)
loss_alpha= getattr(cfg.loss,'loss_alpha',0.5)
criterion= FocalTverskyLoss(
focal_alpha= focal_alpha, focal_gamma= focal_gamma,
tversky_alpha= tversky_alpha, tversky_beta= tversky_beta,
loss_alpha= loss_alpha, num_classes= cfg.model.num_classes
)
elif cfg.loss.name=="focal_dice":
        focal_alpha= getattr(cfg.loss,'focal_alpha',1)
focal_gamma= getattr(cfg.loss,'focal_gamma',2)
loss_alpha= getattr(cfg.loss,'loss_alpha',0.5)
criterion= FocalDiceLoss(
focal_alpha= focal_alpha, focal_gamma= focal_gamma,
loss_alpha= loss_alpha, num_classes= cfg.model.num_classes
)
elif cfg.loss.name=="ici":
        beta= getattr(cfg.loss,'beta',0.9999)
gamma= getattr(cfg.loss,'gamma',2.0)
epsilon= getattr(cfg.loss,'epsilon',1e-7)
criterion= ICILoss(
num_classes= cfg.model.num_classes, beta= beta, gamma= gamma, epsilon= epsilon
)
elif cfg.loss.name=="focal_dice_tversky":
        focal_alpha= getattr(cfg.loss,'focal_alpha',1)
focal_gamma= getattr(cfg.loss,'focal_gamma',2)
tversky_alpha= getattr(cfg.loss,'tversky_alpha',0.5)
tversky_beta= getattr(cfg.loss,'tversky_beta',0.5)
focal_weight= getattr(cfg.loss,'focal_weight',0.33)
dice_weight= getattr(cfg.loss,'dice_weight',0.33)
tversky_weight= getattr(cfg.loss,'tversky_weight',0.34)
criterion= FocalDiceTverskyLoss(
focal_alpha= focal_alpha, focal_gamma= focal_gamma,
tversky_alpha= tversky_alpha, tversky_beta= tversky_beta,
focal_weight= focal_weight, dice_weight= dice_weight, tversky_weight= tversky_weight,
num_classes= cfg.model.num_classes
)
elif cfg.loss.name=="dice_cross_entropy":
        alpha= getattr(cfg.loss,'alpha',0.5)
criterion= DiceCrossEntropyLoss(alpha= alpha, num_classes= cfg.model.num_classes)
elif cfg.loss.name=="cross_entropy":
        criterion= nn.CrossEntropyLoss()
else:
        criterion= nn.CrossEntropyLoss()
return criterion