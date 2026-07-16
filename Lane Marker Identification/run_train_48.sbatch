#!/bin/bash
#SBATCH --job-name=scnn_96e_resume
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint='gpu_v100&gpu_32gb'
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail
export PYTHONUNBUFFERED=1
export xml_catalog_files_libxml2=${xml_catalog_files_libxml2-}

# --- Load Anaconda and activate your /work env ---
module load anaconda
for CAND in \
  "/util/opt/anaconda3/etc/profile.d/conda.sh" \
  "/util/opt/anaconda/etc/profile.d/conda.sh" \
  "/util/opt/anaconda/deployed-conda-envs/etc/profile.d/conda.sh" \
  "/util/opt/anaconda/25.3.1/etc/profile.d/conda.sh"
do
  [ -f "$CAND" ] && source "$CAND" && break
done
conda activate /work/cse479/cooperm/envs/scnn_192

# --- Quick sanity ---
python - <<'PY'
import torch, sys
print("python:", sys.executable)
print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
print("cuda built:", torch.backends.cuda.is_built(), "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available(): print("GPU:", torch.cuda.get_device_name(0))
PY
nvidia-smi || true

# --- Stage dataset to local scratch (optional but speeds up I/O) ---
echo ">>> Copying dataset to local scratch..."
SCRATCH_DIR="${SLURM_TMPDIR:-/tmp}/culane_run"
mkdir -p "$SCRATCH_DIR"
rsync -a --info=progress2 /work/cse479/cooperm/data/culane/ "$SCRATCH_DIR"/
echo ">>> Dataset staged at $SCRATCH_DIR"

# --- Patch config for 96 epochs, local data_root, and correct split paths ---
python - <<'PY'
from pathlib import Path
import re, time, os

cfgp = Path("config.py")
cfg = cfgp.read_text()

# set epochs
cfg = re.sub(r'epochs:\s*int\s*=\s*\d+', 'epochs: int = 96', cfg)

# set data_root to local scratch
cfg = re.sub(r'data_root:\s*str\s*=\s*".+?"',
             f'data_root: str = "{os.environ.get("SLURM_TMPDIR","/tmp")}/culane_run"', cfg)

# set split files to absolute paths inside /work/cse479/cooperm/data/culane/
cfg = re.sub(r'train_list:\s*str\s*=\s*".+?"',
             'train_list: str = "/work/cse479/cooperm/data/culane/train.txt"', cfg)
cfg = re.sub(r'val_list:\s*str\s*=\s*".+?"',
             'val_list: str = "/work/cse479/cooperm/data/culane/val.txt"', cfg)

# new output directory
stamp = time.strftime("%Y%m%d_%H%M%S")
cfg = re.sub(r'out_dir:\s*str\s*=\s*".+?"',
             f'out_dir: str = "/work/cse479/cooperm/runs/scnn_1{stamp}"', cfg)

# use full train/val
cfg = cfg.replace('train_small.txt','train.txt').replace('val_small.txt','val.txt')

cfgp.write_text(cfg)
print("✅ config.py patched: epochs=96, data_root=local scratch, correct split files, new out_dir.")
PY

# --- Checkpoint path ---
CKPT="/work/cse479/cooperm/runs/scnn_120251114_102220/ep87.pth"
if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint not found at $CKPT"
  exit 1
fi
echo ">>> Resuming from checkpoint: $CKPT"
python - <<PY
import torch
ckpt = torch.load("$CKPT", map_location="cpu")
print("Checkpoint epoch:", ckpt.get("epoch"), "mIoU:", ckpt.get("miou", "n/a"))
PY

# --- Run training ---
cd /work/cse479/cooperm/scnn_lane
export RESUME_CKPT="$CKPT"
python -u train.py
