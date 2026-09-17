import os
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
from sklearn.model_selection import StratifiedKFold


SEED = 42
NUM_CLASSES = 5
EPOCHS = 100
BATCH_SIZE = 32
LR = 3e-3
WEIGHT_DECAY = 3e-4
DROPOUT = 0.35
LABEL_SMOOTHING = 0.05
EMA_DECAY = 0.999
GRAD_CLIP = 2.0
PATIENCE = 25
N_FOLDS = 5

IMG_H = 128
IMG_W = 64

BASE_PATH = '/kaggle/input/datasets/iuliamariapopescu/signal-object-detection'
OUT_PATH = os.getcwd()

# system setup
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
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
train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_H, IMG_W)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=8),
    transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.95, 1.05)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.25]),
    transforms.RandomErasing(p=0.10, scale=(0.01, 0.08), value=0.0),
])

val_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.25]),
])

#dataset
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
            
        img = Image.open(img_path).convert('RGB')
        if self.transform: 
            img = self.transform(img)
            
        if self.is_test: 
            return img
            
        return img, int(self.df['label'].iloc[idx]) - 1


#arhitecture
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2),
    )

class GrayCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=DROPOUT):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 2))
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
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
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class FocalLoss(nn.Module):
    def __init__(self, class_weights=None, gamma=1.5):
        super().__init__()
        self.register_buffer('w', class_weights.float() if class_weights is not None else None)
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.w, reduction='none', label_smoothing=LABEL_SMOOTHING)
        pt = torch.exp(-ce).clamp(1e-6, 1.0)
        return (((1 - pt) ** self.gamma) * ce).mean()

# EMA
class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.d = decay
        self.m = copy.deepcopy(model).eval()
        for p in self.m.parameters(): 
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ep, p in zip(self.m.parameters(), model.parameters()):
            ep.mul_(self.d).add_(p.detach(), alpha=1-self.d)
        for eb, b in zip(self.m.buffers(), model.buffers()):
            eb.copy_(b)

@torch.no_grad()
def predict_tta(model, loader):
    model.eval()
    out = []
    for x in loader:
        x = x.to(device)
        p  = F.softmax(model(x), dim=1)
        p += F.softmax(model(torch.flip(x, [3])), dim=1)
        p += F.softmax(model(torch.flip(x, [2])), dim=1)
        p += F.softmax(model(torch.flip(x, [2, 3])), dim=1)
        out.append((p / 4.0).cpu().numpy())
    return np.concatenate(out, axis=0)

# function for training the model
def train_epoch(model, loader, crit, opt, ema, scaler):
    model.train()
    tl, tc, tt = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad()
        
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
    return tl/tt, 100.0*tc/tt

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
    return vl/vt, 100.0*vc/vt


def main():
    train_df = pd.read_csv(os.path.join(BASE_PATH, 'train.csv'))
    test_df  = pd.read_csv(os.path.join(BASE_PATH, 'test.csv'))
    print(f"Data loaded. Train: {len(train_df)} | Test: {len(test_df)}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    test_set = SignalDataset(test_df, os.path.join(BASE_PATH, 'test'), val_tf, is_test=True)
    test_loader = DataLoader(test_set, BATCH_SIZE*2, shuffle=False, num_workers=2, pin_memory=True)
    
    all_fold_test_probs = np.zeros((len(test_df), NUM_CLASSES))
    oof_acc_list = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, train_df['label'])):
        print(f"\nFOLD {fold+1}/{N_FOLDS}")

        tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
        va_df = train_df.iloc[va_idx].reset_index(drop=True)

        counts = tr_df['label'].value_counts().sort_index().values.astype(np.float64)
        cls_w = (1.0 / np.maximum(counts, 1.0))
        cls_w = cls_w / cls_w.mean()
        cw_t = torch.tensor(cls_w, dtype=torch.float32, device=device)

        tr_set = SignalDataset(tr_df, os.path.join(BASE_PATH, 'train'), train_tf)
        va_set = SignalDataset(va_df, os.path.join(BASE_PATH, 'train'), val_tf)

        tr_loader = DataLoader(tr_set, BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
        va_loader = DataLoader(va_set, BATCH_SIZE*2, shuffle=False, num_workers=2, pin_memory=True)

        model = GrayCNN().to(device)
        ema = EMA(model)
        crit = FocalLoss(class_weights=cw_t, gamma=1.5)
        
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
        scaler = torch.amp.GradScaler(device.type)

        best_e = 0.0
        best_wts = copy.deepcopy(ema.m.state_dict())
        no_impr = 0
        t0 = time.time()

        for ep in range(EPOCHS):
            tl, ta = train_epoch(model, tr_loader, crit, opt, ema, scaler)
            va_loss, va_acc = evaluate(model, va_loader, crit)
            ea_loss, ea_acc = evaluate(ema.m, va_loader, crit)
            sch.step()

            print(
                f"  Ep {ep+1:3d}/{EPOCHS} | LR: {opt.param_groups[0]['lr']:.5f} | "
                f"Tr L: {tl:.4f} Tr A: {ta:5.2f}% | "
                f"Val L: {va_loss:.4f} Val A: {va_acc:5.2f}% | "
                f"EMA L: {ea_loss:.4f} EMA A: {ea_acc:5.2f}% | "
                f"Time: {(time.time()-t0)/60:.1f}m"
            )

            if ea_acc > best_e:
                best_e = ea_acc
                best_wts = copy.deepcopy(ema.m.state_dict())
                no_impr = 0
            else:
                no_impr += 1

            if no_impr >= PATIENCE:
                print(f"Early stopping la ep {ep+1}. Best EMA Val Acc: {best_e:.2f}%")
                break

        print(f"Fold {fold+1} Best Val Acc: {best_e:.2f}%")
        oof_acc_list.append(best_e)

        ema.m.load_state_dict(best_wts)
        fold_test_probs = predict_tta(ema.m, test_loader)
        
        all_fold_test_probs += fold_test_probs / N_FOLDS

        del model, ema, opt, sch, tr_loader, va_loader
        torch.cuda.empty_cache()

    print(f"Overall OOF Accuracy: {np.mean(oof_acc_list):.2f}%")

    final_preds = all_fold_test_probs.argmax(axis=1) + 1 

    sub = pd.DataFrame({'id': test_df['id'], 'label': final_preds})
    out_file = os.path.join(OUT_PATH, 'sample_submission.csv')
    sub.to_csv(out_file, index=False)

    print("\n[INFO] Final Test Set Prediction Distribution:")
    counts = sub['label'].value_counts().sort_index()
    for c, count in counts.items():
        print(f"  Class {c}: {count:4d} samples ({(count/len(sub)*100):.1f}%)")
        
    print(f"\nSubmission file saved: {out_file}")

if __name__ == '__main__':
    main()