import os, torch, numpy as np
from tqdm import tqdm
from config import cfg
from datasets.tusimple import make_loaders
from models.scnn import SCNNNet
from utils.metrics import sigmoid_miou, f1_score

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader = make_loaders(cfg, augment_train=False)

    model = SCNNNet(rows=True, cols=True, up_factor=8, pretrained_backbone=False).to(device)
    ckpt_path = os.path.join(cfg.out_dir, "best.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"]); model.eval()

    ious, f1s = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Eval"):
            img = batch["image"].to(device)
            mask = batch["mask"].to(device)
            logits = model(img)
            ious.append(sigmoid_miou(logits, mask).item())
            f1s.append(f1_score(logits, mask).item())

    print(f"Validation mIoU: {np.mean(ious):.4f}  F1: {np.mean(f1s):.4f}")

if __name__ == "__main__":
    main()
