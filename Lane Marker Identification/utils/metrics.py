import torch
import torch.nn.functional as F

def sigmoid_miou(logits, target, eps=1e-6):
    # logits: (B,1,H,W), target: (B,1,H,W)
    pred = torch.sigmoid(logits)
    pred = (pred > 0.5).float()
    inter = (pred * target).sum(dim=(1,2,3))
    union = (pred + target - pred*target).sum(dim=(1,2,3)) + eps
    iou = inter / union
    return iou.mean()

def f1_score(logits, target, eps=1e-6):
    pred = (torch.sigmoid(logits) > 0.5).float()
    tp = (pred*target).sum(dim=(1,2,3))
    fp = (pred*(1-target)).sum(dim=(1,2,3))
    fn = ((1-pred)*target).sum(dim=(1,2,3))
    f1 = (2*tp) / (2*tp + fp + fn + eps)
    return f1.mean()

def dice_loss(logits, target, eps=1e-6):
    p = torch.sigmoid(logits)
    num = 2*(p*target).sum(dim=(1,2,3))
    den = p.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + eps
    return 1 - (num/den).mean()

def bce_dice(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)
