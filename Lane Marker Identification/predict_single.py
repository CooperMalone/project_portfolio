# predict_single.py
import torch, cv2, numpy as np, os
from models.scnn import SCNNNet
from utils.visualize import save_pred_overlay

# === SETTINGS ===
root = "/work/cse479/cooperm/data/culane"
rel_path = "driver_23_30frame/05171102_0766.MP4/03590.jpg"
checkpoint = "/work/cse479/cooperm/runs/scnn_120251116_181642/ep96.pth"
out_path = "/work/cse479/cooperm/predict_1.png"

# === LOAD MODEL ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SCNNNet(rows=True, cols=True, up_factor=8, pretrained_backbone=False).to(device)
ckpt = torch.load(checkpoint, map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()

# === LOAD IMAGE ===
img_path = os.path.join(root, "images", rel_path)
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img, (800, 288), interpolation=cv2.INTER_LINEAR)
inp = img_resized.astype(np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
inp = (inp - mean) / std
inp = np.transpose(inp, (2,0,1))[None, ...]
inp = torch.from_numpy(inp).float().to(device)

# === PREDICT ===
with torch.no_grad():
    logits = model(inp)
save_pred_overlay(inp.cpu(), logits.cpu(), out_path)

print(f"✅ Saved overlay to {out_path}")
