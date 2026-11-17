# models/scnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import ResNet18Feature


def _onehot_col(W: int, i: int, device, dtype):
    """Return a (1,1,1,W) one-hot selecting column i."""
    v = torch.zeros((1, 1, 1, W), device=device, dtype=dtype)
    v[..., i] = 1
    return v


def _onehot_row(H: int, j: int, device, dtype):
    """Return a (1,1,H,1) one-hot selecting row j."""
    v = torch.zeros((1, 1, H, 1), device=device, dtype=dtype)
    v[..., j, 0] = 1
    return v


class SCNNPass(nn.Module):
    """
    Spatial CNN message passing (rows + cols) without in-place updates
    so autograd can track everything safely.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU()

    def _sweep_rows(self, x: torch.Tensor, reverse: bool = False):
        # x: B,C,H,W   -> functional accumulation along width
        B, C, H, W = x.shape
        out = x
        rng = range(W - 1) if not reverse else range(W - 1, 0, -1)
        # We update column i+1 (or i-1 when reverse) from its neighbor.
        for k in rng:
            i_src = k if not reverse else k
            i_tgt = k + 1 if not reverse else k - 1
            # neighbor slice to propagate from: B,C,H,1
            src = out[..., i_src : i_src + 1]
            src = src.contiguous()
            msg = self.conv(self.act(src))  # B,C,H,1
            # add message at column i_tgt using one-hot mask (no in-place)
            col_mask = _onehot_col(W, i_tgt, out.device, out.dtype)  # 1,1,1,W
            out = out + msg * col_mask  # broadcast multiply -> add
        return out

    def _sweep_cols(self, x: torch.Tensor, reverse: bool = False):
        # x: B,C,H,W   -> functional accumulation along height
        B, C, H, W = x.shape
        out = x
        rng = range(H - 1) if not reverse else range(H - 1, 0, -1)
        for k in rng:
            j_src = k if not reverse else k
            j_tgt = k + 1 if not reverse else k - 1
            src = out[:, :, j_src : j_src + 1, :]  # B,C,1,W
            msg = self.conv(self.act(src))         # B,C,1,W
            row_mask = _onehot_row(H, j_tgt, out.device, out.dtype)  # 1,1,H,1
            out = out + msg * row_mask
        return out

    def forward(self, x: torch.Tensor, rows: bool = True, cols: bool = True):
        y = x
        if rows:
            y = self._sweep_rows(y, reverse=False)
            y = self._sweep_rows(y, reverse=True)
        if cols:
            y = self._sweep_cols(y, reverse=False)
            y = self._sweep_cols(y, reverse=True)
        return self.bn(y)


class SCNNNet(nn.Module):
    """
    ResNet-18 backbone -> reduce -> SCNN pass -> refine -> 1x1 head
    -> upsample logits to exactly match input size
    """
    def __init__(self, rows: bool = True, cols: bool = True,
                 up_factor: int = 8,            # accepted but ignored
                 pretrained_backbone: bool = True):
        super().__init__()
        self.backbone = ResNet18Feature(pretrained=pretrained_backbone)
        mid = 128

        self.reduce = nn.Sequential(
            nn.Conv2d(self.backbone.out_channels, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(),
        )

        self.scnn = SCNNPass(mid)
        self.refine = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(),
        )

        self.head = nn.Conv2d(mid, 1, kernel_size=1)
        self.rows = rows
        self.cols = cols

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)                 # B, 512, H/32, W/32
        z = self.reduce(feats)                   # B, 128, H/32, W/32
        z = self.scnn(z, rows=self.rows, cols=self.cols)
        z = self.refine(z)
        logits = self.head(z)                    # B, 1, H/32, W/32
        logits = F.interpolate(logits, size=(x.size(2), x.size(3)),
                               mode='bilinear', align_corners=False)
        return logits
