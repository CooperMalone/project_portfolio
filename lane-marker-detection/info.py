import os, re, glob, csv, sys, random
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# --- project imports ---
from config import cfg

# --- OVERRIDE PATHS FOR THIS EVAL RUN ---
cfg.data_root = "/work/cse479/cooperm/data/culane"
cfg.train_list = "train.txt"   # joined with data_root inside the loader
cfg.val_list   = "val.txt"
from datasets.tusimple import make_loaders
from models.scnn import SCNNNet
from utils.metrics import bce_dice, sigmoid_miou, f1_score

# ---------- CONFIG KNOBS ----------
RUN_DIR = "/work/cse479/cooperm/runs/scnn_48e_iso_124_20251030_230641"
OUT_DIR = Path("plots"); OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT_DIR / "recomputed_metrics.csv"
PNG_PATH = OUT_DIR / "recomputed_metrics.png"
EVAL_EVERY = 1          # evaluate every Nth checkpoint (set to 2 or 4 to speed up)
VAL_MAX_BATCHES = 0     # 0 = full val; or cap to e.g. 200 batches for speed
NUM_WORKERS = 2         # dataloader workers
# ----------------------------------

def main():
    # keep things stable & light
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | cuda={torch.cuda.is_available()}", flush=True)

    # find checkpoints ep*.pth
    ckpts = sorted(
        glob.glob(os.path.join(RUN_DIR, "ep*.pth")),
        key=lambda p: int(re.search(r"ep(\d+)\.pth$", p).group(1))
    )
    assert ckpts, f"No ep*.pth in {RUN_DIR}"

    # data (val only)
    # You can point cfg.val_list to val_small.txt if you want an even faster pass.
    _, val_loader = make_loaders(cfg, augment_train=False)
    print(f"val batches total: {len(val_loader)}", flush=True)

    # model
    model = SCNNNet(rows=True, cols=True, up_factor=8, pretrained_backbone=False).to(device)
    model.eval()

    # CSV init (append-safe)
    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "val_loss", "miou", "f1"])
        if write_header:
            w.writeheader()

    # evaluate
    recs = []
    with torch.inference_mode():
        for i, ck in enumerate(ckpts, start=1):
            if (i - 1) % EVAL_EVERY != 0:
                continue
            m = re.search(r"ep(\d+)\.pth$", ck)
            ep = int(m.group(1)) if m else -1
            print(f"\n=== EVAL epoch {ep} from {os.path.basename(ck)} ===", flush=True)

            try:
                state = torch.load(ck, map_location=device)
                model.load_state_dict(state["model"], strict=False)
            except Exception as e:
                print(f"!! skip {ck}: load error: {e}", flush=True)
                continue

            v_losses, v_ious, v_f1s = [], [], []
            for b_i, batch in enumerate(tqdm(val_loader, desc=f"epoch {ep} eval", leave=False)):
                img  = batch["image"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                logits = model(img)
                v_losses.append(bce_dice(logits, mask).item())
                v_ious.append(sigmoid_miou(logits, mask).item())
                v_f1s.append(f1_score(logits, mask).item())

                if VAL_MAX_BATCHES and (b_i + 1) >= VAL_MAX_BATCHES:
                    break

            rec = {
                "epoch": ep,
                "val_loss": float(np.mean(v_losses)) if v_losses else float("nan"),
                "miou": float(np.mean(v_ious)) if v_ious else float("nan"),
                "f1": float(np.mean(v_f1s)) if v_f1s else float("nan"),
            }
            print(f"epoch {ep}: val_loss={rec['val_loss']:.4f} mIoU={rec['miou']:.4f} F1={rec['f1']:.4f}", flush=True)
            recs.append(rec)

            # append incrementally
            with open(CSV_PATH, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["epoch", "val_loss", "miou", "f1"])
                w.writerow(rec)
                f.flush()
                os.fsync(f.fileno())

            # free some memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # plot (simple)
    try:
        import matplotlib.pyplot as plt
        recs = sorted(recs, key=lambda r: r["epoch"])
        xs = [r["epoch"] for r in recs]
        miou = [r["miou"] for r in recs]
        f1   = [r["f1"] for r in recs]
        vls  = [r["val_loss"] for r in recs]
        plt.figure()
        plt.plot(xs, miou, label="mIoU")
        plt.plot(xs, f1,   label="F1")
        plt.plot(xs, vls,  label="Val Loss")
        plt.xlabel("Epoch"); plt.title("Validation metrics vs epoch (recomputed)")
        plt.legend(); plt.tight_layout()
        plt.savefig(PNG_PATH, dpi=160)
        print(f"\nSaved: {CSV_PATH}\nSaved: {PNG_PATH}", flush=True)
    except Exception as e:
        print(f"(plot skipped) {e}", flush=True)

if __name__ == "__main__":
    main()
