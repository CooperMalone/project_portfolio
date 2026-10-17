import os, cv2, numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

class LaneDataset(Dataset):
    def __init__(self, data_root, split_list, img_size=(288, 800), augment=False):
        self.root = data_root
        self.img_dir = os.path.join(data_root, "images")
        self.mask_dir = os.path.join(data_root, "masks")
        self.img_h, self.img_w = img_size
        self.augment = augment
        with open(os.path.join(data_root, split_list), "r") as f:
            self.items = [x.strip() for x in f if x.strip()]

    def __len__(self):
        return len(self.items)

    def _read_img_mask(self, name):
        img_path = os.path.join(self.img_dir, name)
        base = os.path.splitext(name)[0]
        mask_path = os.path.join(self.mask_dir, base + ".png")

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)

        img = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)

        if self.augment and np.random.rand() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1, :])
            mask = np.ascontiguousarray(mask[:, ::-1])

        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))
        mask = mask[None, :, :]

        return torch.from_numpy(img), torch.from_numpy(mask)

    def __getitem__(self, idx):
        name = self.items[idx]
        img, mask = self._read_img_mask(name)
        return {"image": img.float(), "mask": mask.float(), "name": name}


def make_loaders(cfg, augment_train=True):
    tr = LaneDataset(cfg.data_root, cfg.train_list, (cfg.img_height, cfg.img_width), augment=augment_train)
    va = LaneDataset(cfg.data_root, cfg.val_list, (cfg.img_height, cfg.img_width), augment=False)
    train_loader = DataLoader(tr, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(va, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader
