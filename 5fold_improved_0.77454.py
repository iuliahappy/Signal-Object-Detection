import os
import copy
import math
import time
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from torch.cuda.amp import GradScaler, autocast


SEED = 42
NUM_CLASSES = 5
EPOCHS = 70
WARMUP_EPOCHS = 5
BATCH_SIZE = 64
LR = 1.5e-3  
ETA_MIN = 1e-5
WEIGHT_DECAY = 5e-4
DROPOUT = 0.30
LABEL_SMOOTHING = 0.05
EMA_DECAY = 0.999
GRAD_CLIP = 2.0
PATIENCE = 30
N_FOLDS = 5

IMG_H = 128           
IMG_W = 64            

BASE_PATH = '/kaggle/input/datasets/iuliamariapopescu/signal-object-detection'
OUT_PATH = os.getcwd()
NUM_WORKERS = 4

# system setup
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")

#data augmentation
class PerImageStandardize:
    """Scade media si imparte la std pe fiecare imagine -> invarianta la gain."""
    def __call__(self, x):
        m = x.mean()
        s = x.std().clamp_min(1e-6)
        return (x - m) / s

class SpecAugment:
    """Mascheaza benzi pe axa de frecventa si pe axa de timp (set la 0 = media)."""
    def __init__(self, n_freq=2, n_time=2, max_freq=10, max_time=20, p=0.5):
        self.n_freq, self.n_time = n_freq, n_time
        self.max_freq, self.max_time = max_freq, max_time
        self.p = p

    def __call__(self, x):
        if random.random() > self.p:
            return x
        _, H, W = x.shape
        for _ in range(self.n_freq):
            w = random.randint(1, self.max_freq)
            f0 = random.randint(0, max(0, W - w))
            x[:, :, f0:f0 + w] = 0.0
        for _ in range(self.n_time):
            t = random.randint(1, self.max_time)
            t0 = random.randint(0, max(0, H - t))
            x[:, t0:t0 + t, :] = 0.0
        return x

train_tf = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.RandomHorizontalFlip(p=0.5),                       # mirror pe frecventa
    transforms.RandomAffine(degrees=0, translate=(0.03, 0.06)),   # mic shift timp/freq
    transforms.ToTensor(),
    PerImageStandardize(),
    SpecAugment(p=0.5),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    PerImageStandardize(),
])

class SignalDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, str(self.df['id'].iloc[idx]))
        if not os.path.exists(img_path):
            img_path = img_path + '.png'

        # green channel
        img = Image.open(img_path).convert('RGB').getchannel('G')
        if self.transform:
            img = self.transform(img)

        if self.is_test:
            return img
        return img, int(self.df['label'].iloc[idx]) - 1

#model
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.short = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.short(x)
        return F.relu(out, inplace=True)

class SignalResNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=DROPOUT, widths=(32, 64, 128, 256)):
        super().__init__()
        w0, w1, w2, w3 = widths
        self.stem = nn.Sequential(
            nn.Conv2d(1, w0, 3, 1, 1, bias=False),
            nn.BatchNorm2d(w0),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(BasicBlock(w0, w0, 1), BasicBlock(w0, w0, 1))
        self.layer2 = nn.Sequential(BasicBlock(w0, w1, 2), BasicBlock(w1, w1, 1))
        self.layer3 = nn.Sequential(BasicBlock(w1, w2, 2), BasicBlock(w2, w2, 1))
        self.layer4 = nn.Sequential(BasicBlock(w2, w3, 2), BasicBlock(w3, w3, 1))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(w3, w3),
            nn.BatchNorm1d(w3),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),
            nn.Linear(w3, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return self.head(x)

# EMA
class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.m = copy.deepcopy(model).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e_v, m_v in zip(self.m.state_dict().values(), model.state_dict().values()):
            if e_v.dtype.is_floating_point:
                e_v.mul_(self.decay).add_(m_v.detach(), alpha=1.0 - self.decay)
            else:
                e_v.copy_(m_v)   


# LR SCHEDULE: warmup linear -> cosine
def lr_at(epoch):
    if epoch < WARMUP_EPOCHS:
        return LR * float(epoch + 1) / float(WARMUP_EPOCHS)
    prog = (epoch - WARMUP_EPOCHS) / max(1, (EPOCHS - WARMUP_EPOCHS))
    return ETA_MIN + 0.5 * (LR - ETA_MIN) * (1.0 + math.cos(math.pi * prog))


# TTA: identity + frequence flip (W axis)
@torch.no_grad()
def predict_tta(model, loader):
    model.eval()
    out = []
    for x in loader:
        x = x.to(device, non_blocking=True)
        p = F.softmax(model(x), dim=1)
        p = p + F.softmax(model(torch.flip(x, [3])), dim=1)
        out.append((p/2.0).cpu().numpy())
    return np.concatenate(out, axis=0)

# function for training the model

def train_epoch(model, loader, crit, opt, ema, scaler):
    model.train()
    tl, tc, tt = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type = device.type):
            o = model(x)
            loss = crit(o, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(opt)
        scaler.update()
        ema.update(model)
        tl += loss.item() * x.size(0)
        tc += (o.max(1)[1] == y).sum().item()
        tt += x.size(0)
    return tl/tt, 100.0 * tc/tt

@torch.no_grad()
def evaluate(model, loader, crit):
    model.eval()
    vl, vc, vt = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        o = model(x)
        loss = crit(o, y)
        vl += loss.item() * x.size(0)
        vc += (o.max(1)[1] == y).sum().item()
        vt += x.size(0)
    return vl/vt, 100.0 * vc/vt


def main():
    train_df = pd.read_csv(os.path.join(BASE_PATH, 'train.csv'))
    test_df = pd.read_csv(os.path.join(BASE_PATH, 'test.csv'))
    print(f"Data loaded. Train: {len(train_df)} | Test: {len(test_df)}")
    print("Train label distribution:")
    print(train_df['label'].value_counts().sort_index())

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    test_set = SignalDataset(test_df, os.path.join(BASE_PATH, 'test'), val_tf, is_test=True)
    test_loader = DataLoader(test_set, BATCH_SIZE * 2, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    all_fold_test_probs = np.zeros((len(test_df), NUM_CLASSES))
    oof_acc_list = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, train_df['label'])):
        print(f"\nFOLD {fold + 1}/{N_FOLDS} ---")
        tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
        va_df = train_df.iloc[va_idx].reset_index(drop=True)

        tr_set = SignalDataset(tr_df, os.path.join(BASE_PATH, 'train'), train_tf)
        va_set = SignalDataset(va_df, os.path.join(BASE_PATH, 'train'), val_tf)
        tr_loader = DataLoader(tr_set, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                               pin_memory=True, drop_last=True)
        va_loader = DataLoader(va_set, BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS,
                               pin_memory=True)

        model = SignalResNet().to(device)
        ema = EMA(model)
        crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scaler = torch.amp.GradScaler(device.type)

        best_acc = 0.0
        best_wts = copy.deepcopy(ema.m.state_dict())
        no_impr = 0
        t0 = time.time()

        for ep in range(EPOCHS):
            for g in opt.param_groups:
                g['lr'] = lr_at(ep)

            tl, ta = train_epoch(model, tr_loader, crit, opt, ema, scaler)
            _, va_acc = evaluate(model, va_loader, crit)
            _, ema_acc = evaluate(ema.m, va_loader, crit)

            cur = max(va_acc, ema_acc)
            print(f"  Ep {ep + 1:3d}/{EPOCHS} | LR {opt.param_groups[0]['lr']:.5f} | "
                  f"Tr {ta:5.2f}% | Val {va_acc:5.2f}% | EMA {ema_acc:5.2f}% | "
                  f"{(time.time() - t0) / 60:.1f}m")

            if cur > best_acc:
                best_acc = cur
                src = ema.m if ema_acc >= va_acc else model
                best_wts = copy.deepcopy(src.state_dict())
                no_impr = 0
            else:
                no_impr += 1
            if no_impr >= PATIENCE:
                print(f"Early stopping ep {ep + 1}. Best: {best_acc:.2f}%")
                break

        print(f"Fold {fold + 1} Best Val Acc: {best_acc:.2f}%")
        oof_acc_list.append(best_acc)

        model.load_state_dict(best_wts)
        all_fold_test_probs += predict_tta(model, test_loader) / N_FOLDS

        del model, ema, opt, tr_loader, va_loader
        torch.cuda.empty_cache()

        print(f"Overall OOF Accuracy: {np.mean(oof_acc_list):.2f}%")


    final_preds = all_fold_test_probs.argmax(axis=1) + 1
    sub = pd.DataFrame({'id': test_df['id'], 'label': final_preds})
    out_file = os.path.join(OUT_PATH, 'sample_submission.csv')
    sub.to_csv(out_file, index=False)

    # saving probabilities for future pseudo-labeling
    np.save(os.path.join(OUT_PATH, 'test_probs.npy'), all_fold_test_probs)

    print("\nFinal test set prediction distribution:")
    counts = sub['label'].value_counts().sort_index()
    for c, count in counts.items():
        print(f"  Class {c}: {count:4d} ({count / len(sub) * 100:.1f}%)")
    print(f"\nSubmission file saved: {out_file}")

if __name__ == '__main__':
    main()
