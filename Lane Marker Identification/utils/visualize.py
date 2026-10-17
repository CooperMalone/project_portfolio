import os, cv2, numpy as np
import torch

def overlay_mask(rgb, mask, alpha=0.5):
    """rgb: HxWx3 (uint8), mask: HxW (0/1)"""
    color = np.zeros_like(rgb)
    color[..., 1] = 255  # green overlay
    mask3 = np.repeat(mask[..., None], 3, axis=2).astype(np.uint8)
    out = cv2.addWeighted(rgb, 1.0, color, alpha, 0)
    out = np.where(mask3==1, out, rgb)
    return out

@torch.no_grad()
def save_pred_overlay(img_tensor, logit_tensor, save_path):
    # img_tensor: 1x3xHxW normalized (imagenet); we unnormalize for saving
    x = img_tensor[0].cpu().numpy().transpose(1,2,0)  # HWC
    mean = np.array([0.485,0.456,0.406]); std = np.array([0.229,0.224,0.225])
    x = (x*std + mean)
    x = np.clip(x*255, 0, 255).astype(np.uint8)

    pred = (torch.sigmoid(logit_tensor[0]) > 0.5).cpu().numpy()[0]  # HxW
    over = overlay_mask(x, pred.astype(np.uint8))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(over, cv2.COLOR_RGB2BGR))
