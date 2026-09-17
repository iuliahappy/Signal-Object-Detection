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


SEED_TO_TRAIN = 42 # we change the seed for every run
ALL_SEEDS = [42, 43, 44, 45, 46] # seeds 

NUM_CLASSES = 5
NUM_EPOCHS = 150
BATCH_SIZE = 64
BASE_LR = 3e-4
WEIGHT_DECAY = 5e-4
WARMUP_EPOCHS = 5
LABEL_SMOOTHING = 0.08
MIXUP_PROB = 0.5
MIXUP_ALPHA  = 0.2
CUTMIX_ALPHA = 1.0
EMA_DECAY = 0.999
GRAD_CLIP = 1.0
BN_MOMENTUM = 0.05
DROPOUT = 0.3
NUM_WORKERS = 2
BASE_CHANNELS = 64
BLOCKS_PER_STG = 2

# penalization for class 1
CLASS_WEIGHT_VALUES = [0.60, 1.10, 1.10, 1.10, 1.10]


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
class AddGaussianNoise:
    def __init__(self, std=0.02, p=0.25):
        self.std = std
        self.p = p

    def __call__(self, t):
        if random.random() < self.p:
            return t + torch.randn_like(t) * self.std
        return t

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomApply([transforms.RandomRotation(degrees=15, fill=0)], p=0.7),
    transforms.RandomApply([transforms.RandomAffine(
        degrees=0, translate=(0.06, 0.06), scale=(0.92, 1.08), fill=0)], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25]),
    AddGaussianNoise(std=0.02, p=0.25),
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

# MIXUP / CUTMIX
def rand_bbox(size, lam):
    _, _, H, W = size
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = max(cx - cut_w // 2, 0)
    bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, W)
    bby2 = min(cy + cut_h // 2, H)
    return bbx1, bby1, bbx2, bby2


def mixup_or_cutmix(x, y):
    if random.random() < 0.5:
        lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
        idx = torch.randperm(x.size(0), device=x.device)
        x_mix = lam * x + (1 - lam) * x[idx]
        return x_mix, y, y[idx], lam
    else:
        lam = float(np.random.beta(CUTMIX_ALPHA, CUTMIX_ALPHA))
        idx = torch.randperm(x.size(0), device=x.device)
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x_mix = x.clone()
        x_mix[:, :, bby1:bby2, bbx1:bbx2] = x[idx, :, bby1:bby2, bbx1:bbx2]
        lam = 1.0 - ((bbx2-bbx1)*(bby2-bby1) / float(x.size(2)*x.size(3)))
        return x_mix, y, y[idx], lam


def mixup_loss(criterion, out, y_a, y_b, lam):
    return lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)

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

        use_mix = random.random() < MIXUP_PROB
        if use_mix:
            images, y_a, y_b, lam = mixup_or_cutmix(images, labels)

        optimizer.zero_grad()
        outputs = model(images)

        if use_mix:
            loss = mixup_loss(criterion, outputs, y_a, y_b, lam)
        else:
            loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * images.size(0)
        with torch.no_grad():
            _, preds = outputs.max(1)
            target = (y_a if lam >= 0.5 else y_b) if use_mix else labels
            correct += (preds == target).sum().item()
            total += images.size(0)

    return total_loss/total, 100.0 * correct/total


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
            per_class_total[c] += mask.sum()
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
    val_acc = float(lines[0].strip().split('=')[1])
    ema_acc = float(lines[1].strip().split('=')[1])
    return val_acc, ema_acc


#function2 for training the model
def train_one_model(seed, train_df, val_df):
    print(f"\n{'='*60}")
    print(f"Training model with SEED={seed}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_set = SignalDataset(train_df, '/kaggle/input/datasets/iuliamariapopescu/signal-object-detection/train', transform=train_transform)
    val_set = SignalDataset(val_df,   '/kaggle/input/datasets/iuliamariapopescu/signal-object-detection/train', transform=val_transform)

    use_pin = (device.type == 'cuda')
    loader_kw = dict(
        num_workers=NUM_WORKERS,
        pin_memory=use_pin,
        persistent_workers=(NUM_WORKERS > 0),
    )
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,     shuffle=True,  **loader_kw)
    val_loader = DataLoader(val_set,   batch_size=BATCH_SIZE * 2, shuffle=False, **loader_kw)

    model = SignalResNet().to(device)
    ema = EMA(model, decay=EMA_DECAY)

    cw = torch.tensor(CLASS_WEIGHT_VALUES, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=LABEL_SMOOTHING)
    eval_criterion = nn.CrossEntropyLoss()

    param_grps = get_param_groups(model, WEIGHT_DECAY)
    optimizer = torch.optim.AdamW(param_grps, lr=BASE_LR, betas=(0.9, 0.999))
    scheduler = WarmupCosineLR(optimizer, WARMUP_EPOCHS, NUM_EPOCHS, BASE_LR)

    best_val_acc = 0.0
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

    test_df  = pd.read_csv('/kaggle/input/datasets/iuliamariapopescu/signal-object-detection/test.csv')
    test_set = SignalDataset(test_df, '/kaggle/input/datasets/iuliamariapopescu/signal-object-detection/test', transform=val_transform, is_test=True)
    use_pin  = (device.type == 'cuda')
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=use_pin)

    def load_model(path):
        m = SignalResNet().to(device)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        m.eval()
        return m

    # looking for all existing models and reading accuracy from meta.txt
    candidates = []
    for seed in ALL_SEEDS:
        model_path = f'model_seed{seed}.pth'
        ema_path   = f'model_seed{seed}_ema.pth'
        val_acc, ema_acc = load_meta(seed)

        if val_acc is None:
            print(f"Seed {seed}: meta.txt doesn't exist, pass.")
            continue

        if os.path.exists(model_path):
            candidates.append((model_path, val_acc))
            print(f"Found: {model_path} (val_acc={val_acc:.2f}%)")
        if os.path.exists(ema_path):
            candidates.append((ema_path, ema_acc))
            print(f"Found: {ema_path} (ema_acc={ema_acc:.2f}%)")

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


def main():
    if SEED_TO_TRAIN is None:
        print("SEED_TO_TRAIN = None -> generating submission with existing models")
        generate_submission()
        return

    full_train_df = pd.read_csv('/kaggle/input/datasets/iuliamariapopescu/signal-object-detection/train.csv')
    train_df, val_df = train_test_split(full_train_df, test_size=0.1, random_state=42, stratify=full_train_df['label'])
    print(f"Train: {len(train_df)} | Valida: tion{len(val_df)}")
    print(f"Class weights: {CLASS_WEIGHT_VALUES}")

    train_one_model(SEED_TO_TRAIN, train_df, val_df)

    # verify how many models are ready
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
