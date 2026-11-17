import os, random, json, csv, numpy as np, torch
from tqdm import tqdm
from config import cfg
from datasets.tusimple import make_loaders
from models.scnn import SCNNNet
from utils.metrics import bce_dice, sigmoid_miou, f1_score
from utils.visualize import save_pred_overlay

torch.backends.cudnn.benchmark = True


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def _init_writers(out_dir: str):
    """Create/open metrics files."""
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "metrics.jsonl")
    csv_path = os.path.join(out_dir, "metrics.csv")

    # Create CSV with header if it doesn't exist
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "val_loss", "miou", "f1"])
            w.writeheader()
    return jsonl_path, csv_path


def main():
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    train_loader, val_loader = make_loaders(cfg, augment_train=True)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # --- Model/optim ---
    model = SCNNNet(rows=True, cols=True, up_factor=8, pretrained_backbone=True).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    # --- Logging sinks ---
    jsonl_path, csv_path = _init_writers(cfg.out_dir)

    # --- Optional resume from checkpoint ---
    start_epoch = 1
    # Prefer explicit resume path in config if provided; else comment out or set to "".
    resume_ckpt = getattr(cfg, "resume_ckpt", "").strip()
    if resume_ckpt and os.path.exists(resume_ckpt):
        print(f"Resuming from checkpoint: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"→ Loaded epoch {start_epoch - 1}")
    else:
        if resume_ckpt:
            print(f"⚠️ resume_ckpt set but not found: {resume_ckpt} — starting fresh.")
        else:
            print("Starting fresh (no resume).")

    # --- Training loop ---
    for epoch in range(start_epoch, cfg.epochs + 1):
        # Train
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"Train {epoch}/{cfg.epochs}")
        for batch in pbar:
            img = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                logits = model(img)
                loss = bce_dice(logits, mask)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            losses.append(loss.item())
            pbar.set_postfix(loss=np.mean(losses))

        # Validate
        model.eval()
        v_losses, v_ious, v_f1s = [], [], []
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                img = batch["image"].to(device)
                mask = batch["mask"].to(device)
                logits = model(img)
                v_losses.append(bce_dice(logits, mask).item())
                v_ious.append(sigmoid_miou(logits, mask).item())
                v_f1s.append(f1_score(logits, mask).item())

                # a couple overlays each epoch
                if i < 2:
                    save_pred_overlay(batch["image"], logits,
                        os.path.join(cfg.out_dir, f"ep{epoch}_ex{i}.png"))

        miou = float(np.mean(v_ious))
        f1   = float(np.mean(v_f1s))
        vloss = float(np.mean(v_losses))

        # Print the three metrics EVERY epoch
        print(f"Epoch {epoch}: val_loss={vloss:.4f} mIoU={miou:.4f} F1={f1:.4f}")

        # Append to JSONL and CSV
        rec = {"epoch": epoch, "val_loss": vloss, "miou": miou, "f1": f1}
        with open(jsonl_path, "a") as jf:
            jf.write(json.dumps(rec) + "\n")
        with open(csv_path, "a", newline="") as cf:
            w = csv.DictWriter(cf, fieldnames=["epoch", "val_loss", "miou", "f1"])
            w.writerow(rec)

        # Periodic epoch checkpoints only (no best.pth)
        if epoch % cfg.save_every == 0:
            ep_path = os.path.join(cfg.out_dir, f"ep{epoch}.pth")
            torch.save({"model": model.state_dict(), "epoch": epoch}, ep_path)
            print(f"Saved checkpoint: {ep_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
