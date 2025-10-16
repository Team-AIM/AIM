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
    def __init__(self, label_type="et"): # et, tc, flair 바꿔서 3번 학습하기
        self.gpu_id = 0
        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.batch_size = 2
        self.num_epoch = 10000
        self.init_epoch = 1
        self.patience = 10

        # task 설정
        self.label_type = label_type.lower()
        self.modality_order = ("t1", "t1ce", "t2", "flair")
        self.mod_to_idx = {m: i for i, m in enumerate(self.modality_order)}

        self.task_to_modalities = {
            "et": ["t1", "t1ce", "t2", "flair"],
            "tc": ["t1", "t1ce", "t2", "flair"],
            "flair": ["t1", "t1ce", "t2", "flair"],
        }
        self.task_to_labelid = {"et": 3, "tc": 1, "flair": 2}

        if self.label_type not in self.task_to_modalities:
            raise ValueError(f"Unknown label type: {self.label_type}")

        self.sel_indices = [self.mod_to_idx[m] for m in self.task_to_modalities[self.label_type]]
        self.label_value = self.task_to_labelid[self.label_type]

        # path
        self.data_path = "C:/Users/admin/Documents/AIM_LAB/BraTS2025/BraTS-MEN-Train"
        self.json_list = "C:/Users/admin/Documents/AIM_LAB/train_5fold_data.json"
        self.output_path = f"C:/Users/admin/Documents/AIM_LAB/output_{self.label_type}"
        os.makedirs(self.output_path, exist_ok=True)

        # 결과 파일 경로
        self.log_txt = os.path.join(self.output_path, f"train_log_swinunetr_{self.label_type}.txt")
        self.log_file = os.path.join(self.output_path, f"output_{self.label_type}.log")
        self.model_file = os.path.join(self.output_path, f"swinunetr_{self.label_type}.pt")


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
class MySet(Dataset):
    def __init__(self, data_list, sel_indices, label_value, target_size=(128, 128, 128)):
        self.data_list = []
        self.sel_indices = sel_indices
        self.label_value = label_value
        self.target_size = target_size

        for d in data_list:
            label = nib.load(d["label"]).get_fdata()
            if np.sum(label == self.label_value) > 0:
                self.data_list.append(d)

        print(f"Filtered dataset: {len(self.data_list)}/{len(data_list)} samples kept "
              f"(label {self.label_value})")

    def resize_3d(self, img, is_label=False):
        img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()  # [1,1,D,H,W]
        if is_label:
            img_r = F.interpolate(img_t, size=self.target_size, mode="nearest")
        else:
            img_r = F.interpolate(img_t, size=self.target_size, mode="trilinear", align_corners=False)
        return img_r.squeeze().numpy()

    def __getitem__(self, idx):
        d = self.data_list[idx]

        imgs = [nib.load(d["image"][i]).get_fdata() for i in self.sel_indices]
        imgs = [self.normalize(im) for im in imgs]
        imgs = [self.resize_3d(im, is_label=False) for im in imgs]
        data = np.stack(imgs, axis=0).astype(np.float32)

        label = nib.load(d["label"]).get_fdata()
        label = (label == self.label_value).astype(np.float32)
        label = self.resize_3d(label, is_label=True)

        return torch.from_numpy(data), torch.from_numpy(label)

    @staticmethod
    def normalize(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

    def __len__(self):
        return len(self.data_list)


# -------------------------
# Dice Metric
# -------------------------
def dice_coeff(seg, gt, threshold=0.5, eps=1e-7):
    seg = torch.sigmoid(seg)
    seg = (seg > threshold).float()
    intersection = (seg * gt).sum()
    return (2. * intersection + eps) / (seg.sum() + gt.sum() + eps)


# -------------------------
# Train / Val Loop
# -------------------------
def train_epoch(net, loader, optimizer, cost, epoch, max_epochs, print_log):
    net.train()
    total_loss = 0
    for idx, (data, label) in enumerate(loader):
        data, label = data.cuda(), label.cuda()
        output = net(data)

        loss = cost(output, label.unsqueeze(1))  # [B,1,D,H,W]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        print_log(f"Train {epoch}/{max_epochs-1} {idx+1}/{len(loader)} loss: {loss.item():.4f}")

    return total_loss / len(loader)


def test_epoch(net, loader, epoch, max_epochs, print_log):
    net.eval()
    dice_scores = []
    with torch.no_grad():
        for idx, (data, label) in enumerate(loader):
            data, label = data.cuda(), label.cuda()
            output = net(data)

            dice = dice_coeff(output, label)
            dice_scores.append(dice.item())

            print_log(f"Val {epoch}/{max_epochs-1} {idx+1}/{len(loader)}, dice: {dice:.4f}")
    return np.mean(dice_scores)


# -------------------------
# Main
# -------------------------
def main(args, fold=0):
    torch.cuda.set_device(args.gpu_id)

    tr, val = datafold_read(args.json_list, args.data_path, fold=fold, key="training")
    train_loader = DataLoader(MySet(tr, args.sel_indices, args.label_value),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(MySet(val, args.sel_indices, args.label_value),
                            batch_size=1, shuffle=False)

    # SwinUNETR 모델 선언
    net = SwinUNETR(
        img_size=(128, 128, 128),
        in_channels=len(args.sel_indices),
        out_channels=1,
        feature_size=48,
        use_checkpoint=True,
    ).cuda()

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    cost = DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")

    # 로그 저장
    log_file = open(args.log_file, "a")

    def print_log(*msg):
        text = " ".join(str(m) for m in msg)
        print(text, flush=True)
        log_file.write(text + "\n")
        log_file.flush()

    best_dice = 0.
    epochs_no_improve = 0 

    with open(args.log_txt, "w") as logf:
        for epoch in range(args.init_epoch, args.init_epoch + args.num_epoch):
            print_log(time.ctime(), f"Epoch {epoch}")

            train_loss = train_epoch(net, train_loader, optimizer, cost, epoch, args.num_epoch, print_log)
            val_dice = test_epoch(net, val_loader, epoch, args.num_epoch, print_log)

            line = f"Epoch {epoch}: Loss={train_loss:.4f}, Dice={val_dice:.4f}"
            print_log(line)
            logf.write(line + "\n")

            if val_dice > best_dice:
                best_dice = val_dice
                epochs_no_improve = 0 
                torch.save(net.state_dict(), args.model_file)
                print_log(f"New best model saved with Dice={best_dice:.4f}")
            else:
                epochs_no_improve += 1
                print_log(f"No improvement. Patience counter: {epochs_no_improve}/{args.patience}")

            # Early stopping check
            if epochs_no_improve >= args.patience:
                print_log(f"Early stopping triggered at epoch {epoch}")
                break

    print_log(f"Training finished. Best Dice={best_dice:.4f}")
    log_file.close()


if __name__ == "__main__":
    args = InitParser(label_type="et") # et, tc, flair 바꿔서 3번 학습하기
    main(args, fold=1)