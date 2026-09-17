import math
import copy
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
from sklearn.model_selection import train_test_split

#configuration
SEED = 42
NUM_CLASSES = 5
NUM_EPOCHS = 100
BATCH_SIZE = 64
BASE_LR = 3e-4
WEIGHT_DECAY = 5e-4         
WARMUP_EPOCHS = 5
LABEL_SMOOTHING = 0.1
MIXUP_PROB  = 0.5          # probability for MixUp sau CutMix
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0
EMA_DECAY = 0.999
GRAD_CLIP = 1.0
BN_MOMENTUM = 0.05         
DROPOUT = 0.3
NUM_WORKERS = 2        
BASE_CHANNELS = 64
BLOCKS_PER_STG = 2

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

BASE = "/kaggle/input/datasets/iuliamariapopescu/signal-object-detection"

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
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


# data augmentation
class AddGaussianNoise:
    """Zgomot gaussian aplicat dupa Normalize."""
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
    transforms.RandomApply( [transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.92, 1.08), fill=0)], p=0.5,),
    transforms.ToTensor(), 
    transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25]),
    AddGaussianNoise(std=0.02, p=0.25), # 25% chance
    transforms.RandomErasing(p=0.20, scale=(0.02, 0.15), value=0.0),
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

        stage_channels = [c, c*2, c*4, c*8]
        stages = []
        in_c = c
        for stage_idx, out_c in enumerate(stage_channels):
            stride = 1 if stage_idx == 0 else 2
            blocks = []
            for block_idx in range(blocks_per_stage):
                s = stride if block_idx == 0 else 1
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


def mixup_or_cutmix(x, y, mixup_alpha=MIXUP_ALPHA, cutmix_alpha=CUTMIX_ALPHA):
    if random.random() < 0.5:
        # MIXUP
        lam = float(np.random.beta(mixup_alpha, mixup_alpha))
        idx = torch.randperm(x.size(0), device=x.device)
        x_mix = lam * x + (1 - lam) * x[idx]
        return x_mix, y, y[idx], lam
    else:
        # CUTMIX
        lam = float(np.random.beta(cutmix_alpha, cutmix_alpha))
        idx = torch.randperm(x.size(0), device=x.device)
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x_mix = x.clone()
        x_mix[:, :, bby1:bby2, bbx1:bbx2] = x[idx, :, bby1:bby2, bbx1:bbx2]
        # ajustam lam la aria reala
        lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / float(x.size(2) * x.size(3)))
        return x_mix, y, y[idx], lam


def mixup_loss(criterion, out, y_a, y_b, lam):
    return lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)


# using EMA
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
        # buffer-ele (BN running mean/var) — copiem direct
        for eb, b in zip(self.module.buffers(), model.buffers()):
            eb.copy_(b)


# WARMUP + COSINE LR
class WarmupCosineLR:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.opt = optimizer
        self.we  = warmup_epochs
        self.te  = total_epochs
        self.bl  = base_lr
        self.ml  = min_lr
        self.e   = 0

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
        # without weight decay
        if p.ndim <= 1 or name.endswith('.bias'):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]


# function for training the model
def train_one_epoch(model, loader, criterion, optimizer, ema, device,
                    mixup_prob=MIXUP_PROB, grad_clip=GRAD_CLIP):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        use_mix = random.random() < mixup_prob
        if use_mix:
            images, y_a, y_b, lam = mixup_or_cutmix(images, labels)

        optimizer.zero_grad()
        outputs = model(images)

        if use_mix:
            loss = mixup_loss(criterion, outputs, y_a, y_b, lam)
        else:
            loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * images.size(0)
        with torch.no_grad():
            _, preds = outputs.max(1)
            if use_mix:
                # we only measure against the dominant target
                target = y_a if lam >= 0.5 else y_b
            else:
                target = labels
            correct += (preds == target).sum().item()
            total   += images.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)
    return total_loss / total, 100.0 * correct / total


def main():
    # data
    train_df = pd.read_csv(f'{BASE}/train.csv')
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.1,
        random_state=SEED,
        stratify=train_df['label'],
    )
    print(f"Train: {len(train_df)} imagini | Validation: {len(val_df)} imagini")

    train_set = SignalDataset(train_df, f'{BASE}/train', transform=train_transform)
    val_set = SignalDataset(val_df, f'{BASE}/train', transform=val_transform)

    use_pin = (device.type == 'cuda')
    loader_kwargs = dict(
        num_workers=NUM_WORKERS,
        pin_memory=use_pin,
        persistent_workers=(NUM_WORKERS > 0),
    )
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,     shuffle=True,  **loader_kwargs)
    val_loader = DataLoader(val_set,   batch_size=BATCH_SIZE * 2, shuffle=False, **loader_kwargs)

    # model
    model = SignalResNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.2f}M parametri")

    # EMA
    ema = EMA(model, decay=EMA_DECAY)

    # loss/optimizer/scheduler
    criterion  = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    param_grps = get_param_groups(model, WEIGHT_DECAY)
    optimizer = torch.optim.AdamW(param_grps, lr=BASE_LR, betas=(0.9, 0.999))
    scheduler = WarmupCosineLR(optimizer, WARMUP_EPOCHS, NUM_EPOCHS, BASE_LR)

    # train
    print("Start training...")
    best_val_acc     = 0.0
    best_val_acc_ema = 0.0
    t0 = time.time()

    for epoch in range(NUM_EPOCHS):
        lr = scheduler.step()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, ema, device)
        val_loss, val_acc = evaluate(model,     val_loader, criterion, device)
        val_loss_ema, val_acc_ema = evaluate(ema.module, val_loader, criterion, device)
        elapsed = (time.time() - t0) / 60.0
        print(
            f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | LR {lr:.5f} | "
            f"Train L {tr_loss:.4f} A {tr_acc:5.2f}% | "
            f"Val L {val_loss:.4f} A {val_acc:5.2f}% | "
            f"EMA L {val_loss_ema:.4f} A {val_acc_ema:5.2f}% | "
            f"t={elapsed:.1f}min"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Best model saved! Val Acc: {best_val_acc:.2f}%")
        if val_acc_ema > best_val_acc_ema:
            best_val_acc_ema = val_acc_ema
            torch.save(ema.module.state_dict(), 'best_model_ema.pth')
            print(f"Best EMA saved! EMA Val Acc: {best_val_acc_ema:.2f}%")

    print(f"Training finished in {(time.time()-t0)/60:.1f} min.")
    print(f"Best Val Acc: {best_val_acc:.2f}% | Best EMA Val Acc: {best_val_acc_ema:.2f}%")

    # submission cu TTA + ensemble (model + EMA)
    print("Generating submission with TTA + ensemble (best + best EMA)...")

    test_df = pd.read_csv(f'{BASE}/test.csv')
    test_set = SignalDataset(test_df, f'{BASE}/test', transform=val_transform, is_test=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE * 2, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=use_pin)

    def load_model(path):
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
            p += F.softmax(m(torch.flip(images, dims=[3])), dim=1)   # H-flip
            p += F.softmax(m(torch.flip(images, dims=[2])), dim=1)   # V-flip
            p += F.softmax(m(torch.flip(images, dims=[2, 3])), dim=1)  # H+V
            p /= 4.0
            all_probs.append(p.cpu())
        return torch.cat(all_probs, dim=0)

    m1 = load_model('best_model.pth')
    m2 = load_model('best_model_ema.pth')

    probs = (predict_tta(m1, test_loader) + predict_tta(m2, test_loader)) / 2.0
    preds = probs.argmax(dim=1).numpy() + 1  

    out = pd.DataFrame({'id': test_df['id'], 'label': preds})
    out.to_csv('sample_submission.csv', index=False)

    print("\nPredictions dstribution for test set:")
    print(pd.Series(preds).value_counts().sort_index())
    print("sample_submission has been successfully created!")


if __name__ == "__main__":
    main()
