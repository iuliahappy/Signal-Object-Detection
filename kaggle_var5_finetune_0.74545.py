# fine tuning for var4
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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split



SEED = 42
NUM_CLASSES = 5
BASE_CHANNELS = 64
BLOCKS_PER_STG = 2
DROPOUT = 0.2        
BN_MOMENTUM = 0.05

# fine-tune 
FT_EPOCHS = 35
BATCH_SIZE = 64
FT_BASE_LR = 5e-5        
FT_WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2
LABEL_SMOOTHING = 0.05          
EMA_DECAY = 0.999
EMA_WARMUP_EP = 3 # we don't start updating EMA until epoch 3
GRAD_CLIP = 1.0
NUM_WORKERS = 2

CKPT_INPUT = 'best_model.pth'   # from var4
CKPT_OUT = 'best_model_ft.pth'  # where do we save (fine-tune)
CKPT_OUT_EMA  = 'best_model_ft_ema.pth'  # EMA fine-tune

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)



if torch.backends.mps.is_available():
    device =  torch.device("mps")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

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


#less augmentation for fine-tune
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomApply([transforms.RandomRotation(degrees=8, fill=0)], p=0.4),
    transforms.RandomApply(
        [transforms.RandomAffine(0, translate=(0.04, 0.04), scale=(0.96, 1.04), fill=0)],
        p=0.3,
    ),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25]),
    # transforms.RandomErasing(p=0.20, scale=(0.02, 0.15), value=0.0), # we don't use this anymore because of class 1 bias
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25]),
])


#arhitecture
class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, bn_momentum=BN_MOMENTUM):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c, momentum=bn_momentum)
        self.act = nn.SiLU()
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

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.head_pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


# EMA with WARMUP
class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.started = False

    @torch.no_grad()
    def update(self, model):
        if not self.started:
            # Initial: copiem parametrii direct (nu mediam de la 0)
            for ep, p in zip(self.module.parameters(), model.parameters()):
                ep.copy_(p.detach())
            self.started = True
            return
        for ep, p in zip(self.module.parameters(), model.parameters()):
            ep.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for eb, b in zip(self.module.buffers(), model.buffers()):
            eb.copy_(b)


# WARMUP + COSINE LR
class WarmupCosineLR:
    def __init__(self, opt, warmup, total, base_lr, min_lr=1e-7):
        self.opt, self.we, self.te, self.bl, self.ml = opt, warmup, total, base_lr, min_lr
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


# CLASS WEIGHTS — fix for class 1 bias
def compute_class_weights(labels, num_classes=NUM_CLASSES):
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    base_weights = len(labels)/(num_classes*counts)
    # extra boost for non-1 classes
    boost = np.array([0.85, 1.05, 1.05, 1.05, 1.05])
    weights = base_weights * boost
    return torch.tensor(weights, dtype=torch.float32)

# function for training the model
def train_one_epoch(model, loader, criterion, optimizer, ema, device, grad_clip=GRAD_CLIP):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        total_loss += loss.item() * images.size(0)
        with torch.no_grad():
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
    return total_loss/total, 100.0 * correct/total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    per_class_correct = np.zeros(NUM_CLASSES)
    per_class_total = np.zeros(NUM_CLASSES)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        # Per-class stats
        lbl_cpu = labels.cpu().numpy()
        pred_cpu = preds.cpu().numpy()
        for c in range(NUM_CLASSES):
            mask = lbl_cpu == c
            per_class_total[c] += mask.sum()
            per_class_correct[c] += ((pred_cpu == c) & mask).sum()
    per_class_acc = 100.0 * per_class_correct / np.maximum(per_class_total, 1)
    return total_loss / total, 100.0 * correct / total, per_class_acc


#main
def main():
    print(f"Using device: {device}")

    if not os.path.exists(CKPT_INPUT):
        raise FileNotFoundError(f"{CKPT_INPUT} not found")

    #data
    train_df = pd.read_csv('train.csv')
    train_df, val_df = train_test_split(
        train_df, test_size=0.1, random_state=SEED, stratify=train_df['label']
    )
    print(f"Train: {len(train_df)} | Validation: {len(val_df)}")

    train_set = SignalDataset(train_df, 'train', transform=train_transform)
    val_set   = SignalDataset(val_df, 'train', transform=val_transform)

    loader_kw = dict(num_workers=NUM_WORKERS, pin_memory=False,
                     persistent_workers=(NUM_WORKERS > 0))
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,     shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE * 2, shuffle=False, **loader_kw)

    # class weights 
    train_labels = train_df['label'].values - 1   
    class_weights = compute_class_weights(train_labels)
    print(f"\nClass weights (inverse freq + boost non-1):")
    for c in range(NUM_CLASSES):
        cnt = (train_labels == c).sum()
        print(f"  Clasa {c+1}: count={cnt}, weight={class_weights[c]:.4f}")

    # model from cvar 4
    print(f"\nLoad {CKPT_INPUT}...")
    model = SignalResNet().to(device)
    state = torch.load(CKPT_INPUT, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print("The model was successfully uploaded!")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.2f}M parameters")

    # checking start accuracy
    crit_eval = nn.CrossEntropyLoss()
    val_loss0, val_acc0, per_class0 = evaluate(model, val_loader, crit_eval, device)
    print(f"\nStart: Val Loss {val_loss0:.4f} | Val Acc {val_acc0:.2f}%")
    print(f"Per-class acc: " + " | ".join([f"C{i+1}:{per_class0[i]:.1f}%" for i in range(NUM_CLASSES)]))

    # EMA
    ema = EMA(model, decay=EMA_DECAY)

    # loss/optimizer/scheduler
    cw = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=LABEL_SMOOTHING)
    param_grps = get_param_groups(model, FT_WEIGHT_DECAY)
    optimizer = torch.optim.AdamW(param_grps, lr=FT_BASE_LR)
    scheduler = WarmupCosineLR(optimizer, WARMUP_EPOCHS, FT_EPOCHS, FT_BASE_LR)

    # fine-tune
    print("\nStart fine-tuning...")
    best_val_acc = val_acc0
    best_val_acc_ema = 0.0

    # eval criterion without weights 
    eval_criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    t0 = time.time()
    for epoch in range(FT_EPOCHS):
        lr = scheduler.step()

        # EMA activation after warmout
        active_ema = ema if epoch >= EMA_WARMUP_EP else None

        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, active_ema, device)
        val_loss, val_acc, per_class = evaluate(model, val_loader, eval_criterion, device)

        if epoch >= EMA_WARMUP_EP:
            val_loss_ema, val_acc_ema, per_class_ema = evaluate(ema.module, val_loader, eval_criterion, device)
        else:
            val_loss_ema, val_acc_ema = float('nan'), float('nan')

        elapsed = (time.time() - t0) / 60.0
        print(
            f"Epoch {epoch+1:2d}/{FT_EPOCHS} | LR {lr:.6f} | "
            f"Train L {tr_loss:.4f} A {tr_acc:5.2f}% | "
            f"Val L {val_loss:.4f} A {val_acc:5.2f}% | "
            f"EMA A {val_acc_ema if not math.isnan(val_acc_ema) else 0:5.2f}% | "
            f"t={elapsed:.1f}min"
        )
        # Per-class doar la fiecare 5 epoci sau cand e best
        if (epoch + 1) % 5 == 0 or val_acc > best_val_acc:
            print(f"  per-class: " + " | ".join([f"C{i+1}:{per_class[i]:.1f}%" for i in range(NUM_CLASSES)]))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CKPT_OUT)
            print(f"  -> Salvat! Best Val Acc: {best_val_acc:.2f}%")
        if not math.isnan(val_acc_ema) and val_acc_ema > best_val_acc_ema:
            best_val_acc_ema = val_acc_ema
            torch.save(ema.module.state_dict(), CKPT_OUT_EMA)
            print(f"  -> Salvat EMA! Best EMA Val Acc: {best_val_acc_ema:.2f}%")

    print(f"\nFine-tune terminat in {(time.time()-t0)/60:.1f} min.")
    print(f"Best Val Acc: {best_val_acc:.2f}% | Best EMA: {best_val_acc_ema:.2f}%")

    # SUBMISSION cu TTA + 4 models 4 models ensemble
    print("\nSUBMISSION cu TTA + 4 models 4 models ensemble...")
    test_df = pd.read_csv('test.csv')
    test_set = SignalDataset(test_df, 'test', transform=val_transform, is_test=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE * 2, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=False)

    def load_ckpt(path):
        m = SignalResNet().to(device)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        m.eval()
        return m

    @torch.no_grad()
    def predict_tta(m, loader):
        all_probs = []
        for images in loader:
            images = images.to(device, non_blocking=True)
            p  = F.softmax(m(images), dim=1)
            p += F.softmax(m(torch.flip(images, dims=[3])), dim=1)
            p += F.softmax(m(torch.flip(images, dims=[2])), dim=1)
            p += F.softmax(m(torch.flip(images, dims=[2, 3])), dim=1)
            p /= 4.0
            all_probs.append(p.cpu())
        return torch.cat(all_probs, dim=0)

    # 4 models
    candidates = [
        ('best_model.pth',         1.0),  # var4
        ('best_model_ema.pth',     1.0),  # var4 EMA
        (CKPT_OUT,                 1.2),  # fine-tune 
        (CKPT_OUT_EMA,             1.2),  # fine-tune EMA
    ]
    available = [(p, w) for p, w in candidates if os.path.exists(p)]
    print(f"Using {len(available)} models in ensemble:")
    for p, w in available:
        print(f"{p}(weight={w})")

    total_w = sum(w for _, w in available)
    probs_sum = None
    for path, weight in available:
        m = load_ckpt(path)
        p = predict_tta(m, test_loader) * (weight / total_w)
        probs_sum = p if probs_sum is None else probs_sum + p
        del m

    preds = probs_sum.argmax(dim=1).numpy() + 1

    out = pd.DataFrame({'id': test_df['id'], 'label': preds})
    out.to_csv('sample_submission.csv', index=False)

    print("\nPredictions dstribution for test set:")
    print(pd.Series(preds).value_counts().sort_index())
    print("\nEstimated distribution (~22.6%, ~19.4% ~19.4% ~19.4% ~19.4%):")
    expected = [22.6, 19.4, 19.4, 19.4, 19.2]
    actual = pd.Series(preds).value_counts(normalize=True).sort_index() * 100
    for c in range(1, NUM_CLASSES + 1):
        a = actual.get(c, 0.0)
        e = expected[c - 1]
        diff = a - e
        sign = '+' if diff >= 0 else ''
        print(f"Class {c}: {a:.1f}% (asteptat ~{e}%, diff {sign}{diff:.1f}%)")
    print("sample_submission has been successfully created!")

if __name__ == "__main__":
    main()
