import math
import copy
import time
import random
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
import shutil
import glob


SEED_TO_TRAIN = 7
ALL_SEEDS = [7]

NUM_CLASSES     = 5
NUM_EPOCHS      = 30
BATCH_SIZE      = 64
BASE_LR         = 3e-4
WEIGHT_DECAY    = 5e-4
WARMUP_EPOCHS   = 5
LABEL_SMOOTHING = 0.1      
EMA_DECAY       = 0.999
GRAD_CLIP       = 1.0
BN_MOMENTUM     = 0.05
DROPOUT         = 0.4       
NUM_WORKERS     = 2

BASE_CHANNELS   = 128       # var6 had 64
BLOCKS_PER_STG  = 3         # var6 had 2
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")


#dataset
class SignalDataset(Dataset):
    def __init__(self, df, image_directory, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.image_directory = image_directory
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = f"{self.image_directory}/{self.df['id'].iloc[idx]}"
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        if self.is_test:
            return image
        label = int(self.df['label'].iloc[idx]) - 1  
        return image, label

#data augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomApply([transforms.RandomRotation(degrees=15, fill=0)], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25]),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25]),
])


#arhitecture
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=self.alpha,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# squeeze-and-Excitation - recalibrating importance for every channel
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, max(1, channels // reduction), 1, bias=False)
        self.fc2 = nn.Conv2d(max(1, channels // reduction), channels, 1, bias=False)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class BasicBlock(nn.Module):
    """Bloc rezidual cu SE attention."""
    def __init__(self, in_c, out_c, stride=1, bn_momentum=BN_MOMENTUM):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c, momentum=bn_momentum)
        self.act = nn.SiLU()
        self.se = SEBlock(out_c)
        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c, momentum=bn_momentum),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)   
        return self.act(out + identity)


class SignalResNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, base_channels=BASE_CHANNELS,
                 blocks_per_stage=BLOCKS_PER_STG, dropout=DROPOUT,
                 bn_momentum=BN_MOMENTUM):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, c, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c, momentum=bn_momentum),
            nn.SiLU(),
        )
        stages = []
        in_c = c
        for i, out_c in enumerate([c, c*2, c*4, c*8]):
            stride = 1 if i == 0 else 2
            blocks = []
            for b in range(blocks_per_stage):
                s = stride if b == 0 else 1
                blocks.append(BasicBlock(in_c, out_c, stride=s, bn_momentum=bn_momentum))
                in_c = out_c
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.head_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_c, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.head_pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


# EMA
class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ep, p in zip(self.module.parameters(), model.parameters()):
            ep.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for eb, b in zip(self.module.buffers(), model.buffers()):
            eb.copy_(b)


# WARMUP + COSINE LR
class WarmupCosineLR:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.opt = optimizer
        self.we = warmup_epochs
        self.te = total_epochs
        self.bl = base_lr
        self.ml = min_lr
        self.e = 0

    def step(self):
        if self.e < self.we:
            lr = self.bl * (self.e + 1) / self.we
        else:
            t = (self.e - self.we) / max(1, self.te - self.we)
            lr = self.ml + 0.5 * (self.bl - self.ml) * (1 + math.cos(math.pi * t))
        for pg in self.opt.param_groups:
            pg['lr'] = lr
        self.e += 1
        return lr


def get_param_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith('.bias'):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]


# function for training the model
def train_one_epoch(model, loader, criterion, optimizer, ema, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    per_class_correct = np.zeros(NUM_CLASSES)
    per_class_total   = np.zeros(NUM_CLASSES)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        lbl_cpu  = labels.cpu().numpy()
        pred_cpu = preds.cpu().numpy()
        for c in range(NUM_CLASSES):
            mask = lbl_cpu == c
            per_class_total[c]   += mask.sum()
            per_class_correct[c] += ((pred_cpu == c) & mask).sum()

    per_class_acc = 100.0 * per_class_correct/np.maximum(per_class_total, 1)
    return total_loss/total, 100.0 * correct/total, per_class_acc



#meta save
def save_meta(seed, val_acc, ema_acc):
    meta_path = f'model_seed{seed}_meta.txt'
    with open(meta_path, 'w') as f:
        f.write(f"val_acc={val_acc:.4f}\n")
        f.write(f"ema_acc={ema_acc:.4f}\n")
    print(f"  Meta salvat in {meta_path}")


def load_meta(seed):
    meta_path = f'model_seed{seed}_meta.txt'
    if not os.path.exists(meta_path):
        return None, None
    with open(meta_path, 'r') as f:
        lines = f.readlines()
    return float(lines[0].strip().split('=')[1]), float(lines[1].strip().split('=')[1])

#function2 for training the model
def train_one_model(seed, train_df, val_df):
    print(f"\n{'='*60}")
    print(f"  ANTRENEZ MODEL CU SEED={seed} (SE-ResNet + Sampler)")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    BASE = '/kaggle/input/datasets/iuliamariapopescu/signal-object-detection'
    train_set = SignalDataset(train_df, f'{BASE}/train', transform=train_transform)
    val_set = SignalDataset(val_df, f'{BASE}/train', transform=val_transform)


    # WeightedRandomSampler — balances classes at sampling level
    # Instead of penalizing in loss, we select images from each class equally
    class_counts = train_df['label'].value_counts().sort_index().values
    weights = 1.0/class_counts
    sample_weights = weights[train_df['label'].values - 1]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    use_pin = (device.type == 'cuda')
    loader_kw = dict(
        num_workers=NUM_WORKERS,
        pin_memory=use_pin,
        persistent_workers=(NUM_WORKERS > 0),
    )
    # Atentie: sampler si shuffle=True sunt mutual exclusive — nu le pune pe amandoua
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler,   **loader_kw)
    val_loader = DataLoader(val_set,   batch_size=BATCH_SIZE*2, shuffle=False,   **loader_kw)

    model = SignalResNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params/1e6:.1f}M")

    ema = EMA(model, decay=EMA_DECAY)

    # FocalLoss with light class weights — sampler does the main balancing
    # light weights keep class 1 under control without destabilizing the workout
    cw = torch.tensor([0.7, 1.1, 1.1, 1.1, 1.1], dtype=torch.float32).to(device)
    criterion      = FocalLoss(alpha=cw, gamma=2.0, label_smoothing=LABEL_SMOOTHING)
    eval_criterion = nn.CrossEntropyLoss()

    param_grps = get_param_groups(model, WEIGHT_DECAY)
    optimizer  = torch.optim.AdamW(param_grps, lr=BASE_LR, betas=(0.9, 0.999))
    scheduler  = WarmupCosineLR(optimizer, WARMUP_EPOCHS, NUM_EPOCHS, BASE_LR)

    best_val_acc     = 0.0
    best_val_acc_ema = 0.0
    model_path = f'model_seed{seed}.pth'
    ema_path = f'model_seed{seed}_ema.pth'
    t0 = time.time()

    for epoch in range(NUM_EPOCHS):
        lr = scheduler.step()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, ema, device)
        val_loss, val_acc, per_class = evaluate(model, val_loader, eval_criterion, device)
        _, ema_acc, _ = evaluate(ema.module, val_loader, eval_criterion, device)
        elapsed = (time.time() - t0)/60.0

        print(
            f"  Epoch {epoch+1:3d}/{NUM_EPOCHS} | LR {lr:.5f} | "
            f"Train L {tr_loss:.4f} A {tr_acc:5.2f}% | "
            f"Val L {val_loss:.4f} A {val_acc:5.2f}% | "
            f"EMA {ema_acc:5.2f}% | t={elapsed:.1f}min"
        )

        if (epoch + 1) % 25 == 0:
            pc = " | ".join([f"C{i+1}:{per_class[i]:.1f}%" for i in range(NUM_CLASSES)])
            print(f"  per-class: {pc}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"Salved! Val Acc: {best_val_acc:.2f}%")

        if ema_acc > best_val_acc_ema:
            best_val_acc_ema = ema_acc
            torch.save(ema.module.state_dict(), ema_path)
            print(f"EMA saved! EMA Acc: {best_val_acc_ema:.2f}%")

    elapsed_total = (time.time() - t0) / 60.0
    print(f"\n Seed {seed} finished in {elapsed_total:.1f} min.")
    print(f"Best Val Acc: {best_val_acc:.2f}% | Best EMA Acc: {best_val_acc_ema:.2f}%")

    save_meta(seed, best_val_acc, best_val_acc_ema)
    return best_val_acc, best_val_acc_ema

#tta extended - 8 augmentations per image
@torch.no_grad()
def predict_tta(model, loader):
    model.eval()
    all_probs = []

    for images in loader:
        images = images.to(device, non_blocking=True)

        p  = F.softmax(model(images), dim=1)
        p += F.softmax(model(torch.flip(images, dims=[3])), dim=1)
        p += F.softmax(model(torch.flip(images, dims=[2])), dim=1)
        p += F.softmax(model(torch.flip(images, dims=[2, 3])), dim=1)

        rot90 = torch.rot90(images, k=1, dims=[2, 3])
        p += F.softmax(model(rot90), dim=1)
        p += F.softmax(model(torch.flip(rot90, dims=[3])), dim=1)
        p += F.softmax(model(torch.flip(rot90, dims=[2])), dim=1)
        p += F.softmax(model(torch.flip(rot90, dims=[2, 3])), dim=1)

        p /= 8.0
        all_probs.append(p.cpu())

    return torch.cat(all_probs, dim=0)


#function for generating the submission
def generate_submission():
    print(f"\n{'='*60}")
    print("Generating submission with ponderated ensemble + TTA x8")
    print(f"{'='*60}")

    BASE = '/kaggle/input/datasets/iuliamariapopescu/signal-object-detection'
    test_df  = pd.read_csv(f'{BASE}/test.csv')
    test_set = SignalDataset(test_df, f'{BASE}/test', transform=val_transform, is_test=True)
    use_pin  = (device.type == 'cuda')
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=use_pin)

    def load_model(path):
        m = SignalResNet().to(device)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        m.eval()
        return m

    candidates = []
    for seed in ALL_SEEDS:
        val_acc, ema_acc = load_meta(seed)
        if val_acc is None:
            print(f"Seed {seed}: meta.txt nu exista, sar peste.")
            continue
        if os.path.exists(f'model_seed{seed}.pth'):
            candidates.append((f'model_seed{seed}.pth', val_acc))
            print(f"Found: model_seed{seed}.pth (val_acc={val_acc:.2f}%)")
        if os.path.exists(f'model_seed{seed}_ema.pth'):
            candidates.append((f'model_seed{seed}_ema.pth', ema_acc))
            print(f"Found: model_seed{seed}_ema.pth (ema_acc={ema_acc:.2f}%)")

    if not candidates:
        print("ERROR! No model found! Train first!")
        return

    print(f"\nTotal models in ensemble: {len(candidates)}")
    total_weight = sum(w for _, w in candidates)
    probs_sum = None

    for path, weight in candidates:
        print(f"Processing {path} (weight={weight/total_weight*100:.1f}%)...")
        m = load_model(path)
        p = predict_tta(m, test_loader) * (weight / total_weight)
        probs_sum = p if probs_sum is None else probs_sum + p
        del m

    preds = probs_sum.argmax(dim=1).numpy() + 1 

    print("\nPredictions dstribution for test set:")
    dist = pd.Series(preds).value_counts(normalize=True).sort_index() * 100
    expected = [22.6, 19.4, 19.4, 19.4, 19.2]
    for c in range(1, NUM_CLASSES + 1):
        a = dist.get(c, 0.0)
        e = expected[c - 1]
        diff = a - e
        sign = '+' if diff >= 0 else ''
        print(f"Class {c}: {a:.1f}% (asteptat ~{e}%, diff {sign}{diff:.1f}%)")

    out = pd.DataFrame({'id': test_df['id'], 'label': preds})
    out.to_csv('sample_submission.csv', index=False)
    print("sample_submission has been successfully created!")


# ============================================================
# 14) MAIN
# ============================================================
def main():
    fisiere_input = (
        glob.glob('/kaggle/input/**/*.pth', recursive=True) +
        glob.glob('/kaggle/input/**/*meta.txt', recursive=True)
    )
    for fisier in fisiere_input:
        try:
            shutil.copy(fisier, '.')
        except shutil.SameFileError:
            pass

    if SEED_TO_TRAIN is None:
        print("SEED_TO_TRAIN = None -> generating submission with existing models")
        generate_submission()
        return

    BASE = '/kaggle/input/datasets/iuliamariapopescu/signal-object-detection'
    full_train_df = pd.read_csv(f'{BASE}/train.csv')
    train_df, val_df = train_test_split(full_train_df, test_size=0.1, random_state=42, stratify=full_train_df['label'])
    print(f"Train: {len(train_df)} | Validation: {len(val_df)}")

    train_one_model(SEED_TO_TRAIN, train_df, val_df)

    models_done = [s for s in ALL_SEEDS if os.path.exists(f'model_seed{s}.pth')]
    print(f"Train models: {models_done}")
    print(f"Remaining models: {[s for s in ALL_SEEDS if s not in models_done]}")

    if set(models_done) == set(ALL_SEEDS):
        print("All models ready! Now, we generate submission!")
        generate_submission()
    else:
        remaining = [s for s in ALL_SEEDS if s not in models_done]

        print(f"\nChange SEED_TO_TRAIN = {remaining[0]} and run again!")



if __name__ == "__main__":
    main()