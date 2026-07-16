# config.py
from dataclasses import dataclass

@dataclass
class Config:
    # --- DATA ---
    # Root of your organized CULane data (images/, masks/, train.txt, val.txt)
    data_root: str = "/tmp/culane_run"
   
    train_list: str = "/work/cse479/cooperm/data/culane/train.txt"        
    val_list: str = "/work/cse479/cooperm/data/culane/val.txt"          

    # Image size (H, W)
    img_height: int = 288
    img_width:  int = 800

    # --- TRAINING ---
    epochs: int = 96
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    amp: bool = True            # mixed precision on GPU
    seed: int = 42
    

    # DataLoader workers (set to match your sbatch cpus-per-task)
    num_workers: int = 8
    # Save checkpoints/PNGs every N epochs
    save_every: int = 1

    # --- OUTPUTS ---
    # Where to write ep*.png, ep*.pth, best.pth
    resume_ckpt: str = "/work/cse479/cooperm/runs/scnn_120251114_102220/ep87.pth"
    out_dir: str = "/work/cse479/cooperm/runs/scnn_120251116_181642"

# required by train.py
cfg = Config()



