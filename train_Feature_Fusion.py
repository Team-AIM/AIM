import os
import json
import time
import numpy as np
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from monai.networks.nets import SwinUNETR
from monai.losses import DiceLoss


# -------------------------
# Config
# -------------------------
class InitParser(object):
    def __init__(self):
        self.gpu_id = 0
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.batch_size = 1
        self.num_epoch = 10000
        self.init_epoch = 1
        self.patience = 15

        # 모달리티 순서/인덱스
        self.modality_order = ("t1", "t1ce", "t2", "flair")
        self.mod_to_idx = {m: i for i, m in enumerate(self.modality_order)}
        self.sel_indices = [self.mod_to_idx[m] for m in self.modality_order]

  
        # 경로 설정
        self.data_path = "C:/Users/admin/Desktop/LAB/BraTS-MEN-Train"
        self.json_list = "C:/Users/admin/Desktop/LAB/json_balanced/bal_train_5fold_data.json"

        # 베이스 모델 체크포인트 경로
        self.ckpt_et = "C:/Users/admin/Desktop/LAB/SwinUNETR_sep_model/swinunetr_et.pt"
        self.ckpt_tc = "C:/Users/admin/Desktop/LAB/SwinUNETR_sep_model/swinunetr_tc.pt"
        self.ckpt_flair = "C:/Users/admin/Desktop/LAB/SwinUNETR_sep_model/swinunetr_flair.pt"

        # 출력 경로
        self.output_path = "C:/Users/admin/Desktop/LAB/Output_SwinUNETR_Fusion"
        os.makedirs(self.output_path, exist_ok=True)
        self.log_txt = os.path.join(self.output_path, "train_log_fusion.txt")
        self.log_file = os.path.join(self.output_path, "output_fusion.log")

        self.model_file = os.path.join(self.output_path, "Best_Fusion.pth")


        # 리사이즈 타깃 크기
        self.target_size = (128, 128, 128)

        # DataLoader 성능
        self.num_workers_tr = 4
        self.num_workers_val = 2
        self.pin_memory = True
        self.persistent_workers = True


# -------------------------
# JSON Loader
# -------------------------
def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)
    json_data = json_data[key]

    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr, val = [], []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)
    return tr, val


# -------------------------
# Dataset
# -------------------------
class FusionSet(Dataset):
    def __init__(self, data_list, sel_indices, label_map=None, target_size=(128,128,128)):
        self.data_list = list(data_list)
        self.sel_indices = sel_indices
        self.target_size = target_size
        self.label_map = label_map

        print(f"Dataset: {len(self.data_list)} samples")

    def resize_3d(self, img, is_label=False):
        img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()  # [1,1,D,H,W]
        if is_label:
            img_r = F.interpolate(img_t, size=self.target_size, mode="nearest")
        else:
            img_r = F.interpolate(img_t, size=self.target_size, mode="trilinear", align_corners=False)
        return img_r.squeeze().numpy()

    @staticmethod
    def normalize(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

    def __getitem__(self, idx):
        d = self.data_list[idx]

        # 이미지 4채널 로드/정규화/리사이즈
        imgs = [nib.load(d["image"][i]).get_fdata() for i in self.sel_indices]
        imgs = [self.normalize(im) for im in imgs]
        imgs = [self.resize_3d(im, is_label=False) for im in imgs]
        data = np.stack(imgs, axis=0).astype(np.float32)  # [4,D,H,W]

        # 라벨: 멀티클래스(0:bg,1:TC,2:FLAIR,3:ET)
        label = nib.load(d["label"]).get_fdata()
        label = self.resize_3d(label, is_label=True).astype(np.int64)  # [D,H,W]

        return torch.from_numpy(data), torch.from_numpy(label)

    def __len__(self):
        return len(self.data_list)


# -------------------------
# Fusion Heads
# -------------------------
class Fusion1x1(nn.Module):
    def __init__(self, in_ch=3, out_ch=4):
        super().__init__()
        self.head = nn.Conv3d(in_ch, out_ch, kernel_size=1)
    def forward(self, x):
        return self.head(x)


class FusionSmall(nn.Module):
    def __init__(self, in_ch=3, out_ch=4, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, hidden, 3, padding=1),
            nn.InstanceNorm3d(hidden), nn.ReLU(inplace=True),
            nn.Conv3d(hidden, hidden, 3, padding=1),
            nn.InstanceNorm3d(hidden), nn.ReLU(inplace=True),
            nn.Conv3d(hidden, out_ch, 1)
        )
    def forward(self, x):
        return self.net(x)


# -------------------------
# Metrics
# -------------------------
def dice_per_class_mc(logits, target, eps=1e-6):
    prob = F.softmax(logits, dim=1)               # [B,4,D,H,W]
    onehot = torch.zeros_like(prob)
    onehot.scatter_(1, target.unsqueeze(1), 1.0)  # [B,4,D,H,W]

    dices = []
    for c in range(prob.shape[1]):
        p = prob[:, c]
        g = onehot[:, c]
        inter = (p * g).sum()
        denom = p.sum() + g.sum()
        d = (2 * inter + eps) / (denom + eps)
        dices.append(d.item())
    return dices  # [bg, TC, FLAIR, ET]


# -------------------------
# Train / Val Loops
# -------------------------
def train_epoch(net_et, net_tc, net_fl, fusion, loader, opt, loss_fn, device):
    fusion.train()
    total_loss = 0.0
    for idx, (data, label) in enumerate(loader):
        data, label = data.to(device).float(), label.to(device).long()

        with torch.no_grad():
            le = net_et(data)  # [B,1,D,H,W] (ET 로짓)
            lt = net_tc(data)  # [B,1,D,H,W] (TC 로짓)
            lf = net_fl(data)  # [B,1,D,H,W] (FLAIR 로짓)

        feats = torch.cat([le, lt, lf], dim=1)     # [B,3,D,H,W]
        out = fusion(feats)                        # [B,4,D,H,W]

        loss = loss_fn(out, label)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()
    return total_loss / max(1, len(loader))


def val_epoch(net_et, net_tc, net_fl, fusion, loader, device):
    fusion.eval()
    ce = nn.CrossEntropyLoss()
    dice_mc = DiceLoss(to_onehot_y=True, softmax=True, include_background=True)

    losses = []
    dices_all = []
    with torch.no_grad():
        for idx, (data, label) in enumerate(loader):
            data, label = data.to(device).float(), label.to(device).long()

            le = net_et(data)
            lt = net_tc(data)
            lf = net_fl(data)
            feats = torch.cat([le, lt, lf], dim=1)
            logits = fusion(feats)

            loss = 0.5 * ce(logits, label) + 0.5 * dice_mc(logits, label.unsqueeze(1))
            losses.append(loss.item())

            dices = dice_per_class_mc(logits, label)  # [bg,TC,FLAIR,ET]
            dices_all.append(dices)

    if len(losses) == 0:
        return 0.0, [0.0, 0.0, 0.0, 0.0]

    mean_loss = float(np.mean(losses))
    mean_dices = np.array(dices_all).mean(axis=0).tolist()
    return mean_loss, mean_dices


# -------------------------
# Main
# -------------------------
def main(args, fold=1, use_small_fusion=False):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu_id)

    tr, val = datafold_read(args.json_list, args.data_path, fold=fold, key="training")

    train_ds = FusionSet(tr, args.sel_indices, target_size=args.target_size)
    val_ds   = FusionSet(val, args.sel_indices, target_size=args.target_size)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers_tr, pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers_val, pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers
    )

    def load_base(ckpt_path):
        net = SwinUNETR(
            img_size=args.target_size,
            in_channels=len(args.sel_indices),  # 4
            out_channels=1,                     # binary
            feature_size=48,
            use_checkpoint=True
        ).to(device)
        sd = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(sd)
        net.eval()
        for p in net.parameters():
            p.requires_grad = False
        return net

    net_et = load_base(args.ckpt_et)
    net_tc = load_base(args.ckpt_tc)
    net_fl = load_base(args.ckpt_flair)

    # Fusion 헤드
    fusion = (FusionSmall(3, 4, 16) if use_small_fusion else Fusion1x1(3, 4)).to(device)

    opt = torch.optim.AdamW(fusion.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ce = nn.CrossEntropyLoss()
    dice_mc = DiceLoss(to_onehot_y=True, softmax=True, include_background=True)
    def loss_fn(logits, target):
        return 0.5 * ce(logits, target) + 0.5 * dice_mc(logits, target.unsqueeze(1))

    # 로그
    log_file = open(args.log_file, "a")
    def print_log(*msg):
        text = " ".join(str(m) for m in msg)
        log_file.write(text + "\n"); log_file.flush()

    best_score = -1.0  # (TC,FLAIR,ET) 평균 Dice
    epochs_no_improve = 0

    with open(args.log_txt, "w", encoding="utf-8") as logf:
        for epoch in range(args.init_epoch, args.init_epoch + args.num_epoch):
            print_log(time.ctime(), f"Epoch {epoch}")

            train_loss = train_epoch(net_et, net_tc, net_fl, fusion, train_loader, opt, loss_fn, device)
            val_loss, dices = val_epoch(net_et, net_tc, net_fl, fusion, val_loader, device)
            dice_bg, dice_tc, dice_fl, dice_et = dices
            mean_tumor = float(np.mean([dice_tc, dice_fl, dice_et]))

            line = f"Epoch {epoch}: Loss={train_loss:.4f}, ValLoss={val_loss:.4f}, Dice(bg/TC/FL/ET)=({dice_bg:.4f}/{dice_tc:.4f}/{dice_fl:.4f}/{dice_et:.4f}), mean_tumor={mean_tumor:.4f}"
            print_log(line)
            logf.write(line + "\n")

            if mean_tumor > best_score:
                best_score = mean_tumor
                epochs_no_improve = 0
                torch.save(fusion.state_dict(), args.model_file)
                print_log(f"New best fusion saved. mean_tumor={best_score:.4f}")
            else:
                epochs_no_improve += 1
                print_log(f"No improvement. Patience {epochs_no_improve}/{args.patience}")

            if epochs_no_improve >= args.patience:
                print_log(f"Early stopping at epoch {epoch}. Best mean_tumor={best_score:.4f}")
                break

    print_log(f"Training finished. Best mean_tumor Dice={best_score:.4f}")
    log_file.close()

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# -------------------------
# Inference helper
# -------------------------
@torch.no_grad()
def infer_volume(net_et, net_tc, net_fl, fusion, data, device="cuda"):
    data = data.to(device).float()          # [1,4,D,H,W]
    le = net_et(data); lt = net_tc(data); lf = net_fl(data)
    feats = torch.cat([le, lt, lf], dim=1)
    logits = fusion(feats)
    prob = F.softmax(logits, dim=1)         # [1,4,D,H,W]
    pred = torch.argmax(prob, dim=1)        # [1,D,H,W]
    return pred, prob


if __name__ == "__main__":
    args = InitParser()
    main(args, fold=1, use_small_fusion=False)
