import math, copy, time, random, os, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

NCLS = 5
EPOCHS = 70                  
BS = 64
LR = 3e-4
WD = 5e-4
WARMUP = 5
LS = 0.1
EMA_DEC = 0.999
GCLIP = 1.0
BN_MOM = 0.05
DO = 0.3
CH = 64
BLOCKS = 2
MIX_P = 0.5 # mixup/cutmix on half of the batches
NUM_W = 2

# each model has its own seed for init + its own val split because different val split = diversity in the ensemble
RUNS = [
    {'seed': 2024, 'val_seed': 2024, 'tag': 'a'},
    {'seed': 314,  'val_seed': 314,  'tag': 'b'},
]

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")

ROOT = os.path.dirname(os.path.abspath(__file__))

# dataset 
class SignalDataset(Dataset):
    def __init__(self, df, image_directory, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.image_directory = image_directory
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = os.path.join(self.image_directory, self.df['id'].iloc[idx])
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.is_test:
            return img
        label = int(self.df['label'].iloc[idx]) - 1
        return img, label


# data augmentation
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
    transforms.RandomErasing(p=0.20, scale=(0.02, 0.15), value=0.0),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25]),
])

#arhitecture
class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, bn_momentum=BN_MOM):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_c, momentum=bn_momentum)
        self.act   = nn.SiLU()
        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c, momentum=bn_momentum),
            )

    def forward(self, x):
        idn = x if self.downsample is None else self.downsample(x)
        o = self.act(self.bn1(self.conv1(x)))
        o = self.bn2(self.conv2(o))
        return self.act(o + idn)


class Net(nn.Module):
    def __init__(self, num_classes=NCLS, base_channels=CH,
                 blocks_per_stage=BLOCKS, dropout=DO, bn_momentum=BN_MOM):
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
        self.stages    = nn.Sequential(*stages)
        self.head_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(in_c, num_classes)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.head_pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


# MIXUP/CUTMIX
def rand_bbox(size, lam):
    _, _, H, W = size
    cut_rat = math.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    return (max(cx - cut_w // 2, 0), max(cy - cut_h // 2, 0),
            min(cx + cut_w // 2, W), min(cy + cut_h // 2, H))


def mixup_or_cutmix(x, y, mu_alpha=0.2, cm_alpha=1.0):
    if random.random() < 0.5:
         #mixup
        lam = float(np.random.beta(mu_alpha, mu_alpha))
        idx = torch.randperm(x.size(0), device=x.device)
        return lam * x + (1 - lam) * x[idx], y, y[idx], lam
    else:
        #cutmix
        lam = float(np.random.beta(cm_alpha, cm_alpha))
        idx = torch.randperm(x.size(0), device=x.device)
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        xm = x.clone()
        xm[:, :, bby1:bby2, bbx1:bbx2] = x[idx, :, bby1:bby2, bbx1:bbx2]
        lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / float(x.size(2) * x.size(3)))
        return xm, y, y[idx], lam

# EMA
class EMA:
    def __init__(self, m, decay=EMA_DEC):
        self.d = decay
        self.m = copy.deepcopy(m).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, m):
        for ep, p in zip(self.m.parameters(), m.parameters()):
            ep.mul_(self.d).add_(p.detach(), alpha=1 - self.d)
        for eb, b in zip(self.m.buffers(), m.buffers()):
            eb.copy_(b)

# LR scheduler (warmup + cosine) 
class WarmupCosineLR:
    def __init__(self, opt, warmup, total, base_lr, min_lr=1e-6):
        self.opt = opt
        self.we = warmup
        self.te = total
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


def get_param_groups(m, wd):
    decay, no_decay = [], []
    for nm, p in m.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or nm.endswith('.bias'):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{'params': decay, 'weight_decay': wd},
            {'params': no_decay, 'weight_decay': 0.0}]


# function for training the model
def train_one_epoch(model, loader, crit, opt, ema, device, mix_p=MIX_P):
    model.train()
    tl, tc, tt = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        use_mix = random.random() < mix_p
        if use_mix:
            x, ya, yb, lam = mixup_or_cutmix(x, y)
        opt.zero_grad()
        o = model(x)
        if use_mix:
            loss = lam * crit(o, ya) + (1 - lam) * crit(o, yb)
        else:
            loss = crit(o, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GCLIP)
        opt.step()
        ema.update(model)
        tl += loss.item() * x.size(0)
        with torch.no_grad():
            _, pr = o.max(1)
            tgt = (ya if (use_mix and lam >= 0.5) else (yb if use_mix else y))
            tc += (pr == tgt).sum().item()
            tt += x.size(0)
    return tl / tt, 100.0 * tc / tt


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    vc, vt = 0, 0
    per_c = np.zeros(NCLS); per_t = np.zeros(NCLS)
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        o = model(x)
        _, pr = o.max(1)
        vc += (pr == y).sum().item()
        vt += x.size(0)
        yn = y.cpu().numpy(); pn = pr.cpu().numpy()
        for c in range(NCLS):
            m = yn == c
            per_t[c] += m.sum()
            per_c[c] += ((pn == c) & m).sum()
    return 100.0 * vc / vt, per_c / np.maximum(per_t, 1) * 100


# helpers
def find_file(name, base=ROOT):
    if os.path.exists(name):
        return os.path.abspath(name)
    for r, _, fs in os.walk(base):
        if name in fs:
            return os.path.join(r, name)
    return None


def load_model(path):
    m = Net().to(device)
    m.load_state_dict(torch.load(path, map_location = device, weights_only = True))
    m.eval()
    return m


@torch.no_grad()
def predict_tta(model, loader):
    model.eval()
    out = []
    for x in loader:
        if isinstance(x, (list, tuple)): x = x[0]
        x = x.to(device)
        p  = F.softmax(model(x), dim=1)
        p += F.softmax(model(torch.flip(x, [3])), dim=1)
        p += F.softmax(model(torch.flip(x, [2])), dim=1)
        p += F.softmax(model(torch.flip(x, [2, 3])), dim=1)
        p /= 4.0
        out.append(p.cpu())
    return torch.cat(out, 0)


# function2 for training the model
def train_one_model(run_cfg, full_train_df):
    seed = run_cfg['seed']
    val_seed = run_cfg['val_seed']
    tag = run_cfg['tag']
    out_ckpt = os.path.join(ROOT, f'best_v7_{tag}.pth')
    out_ema  = os.path.join(ROOT, f'best_v7_{tag}_ema.pth')

    print(f'\nModel {tag.upper()} (seed={seed}, val_seed={val_seed}) ====')

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    tr_df, va_df = train_test_split(
        full_train_df, test_size=0.1,
        random_state=val_seed, stratify=full_train_df['label'])
    print(f'Train={len(tr_df)} Val={len(va_df)}')

    tr_set = SignalDataset(tr_df, os.path.join(ROOT, 'train'), train_transform)
    va_set = SignalDataset(va_df, os.path.join(ROOT, 'train'), val_transform)

    tr_loader = DataLoader(tr_set, batch_size=BS, shuffle=True,
                           num_workers=NUM_W, persistent_workers=(NUM_W > 0))
    va_loader = DataLoader(va_set, batch_size=BS * 2, shuffle=False,
                           num_workers=NUM_W, persistent_workers=(NUM_W > 0))

    net = Net().to(device)
    n_p = sum(p.numel() for p in net.parameters())
    print(f'Net: {n_p/1e6:.2f}M params')
    ema = EMA(net)

    crit = nn.CrossEntropyLoss(label_smoothing=LS)
    opt = torch.optim.AdamW(get_param_groups(net, WD), lr=LR)
    sch = WarmupCosineLR(opt, WARMUP, EPOCHS, LR)

    best_v, best_e = 0.0, 0.0
    t0 = time.time()
    for ep in range(EPOCHS):
        lr = sch.step()
        tl, ta = train_one_epoch(net, tr_loader, crit, opt, ema, device)
        va, per_c   = evaluate(net, va_loader, device)
        ea, per_c_e = evaluate(ema.m, va_loader, device)
        el = (time.time() - t0) / 60
        print(f'Ep {ep+1:2d}/{EPOCHS} lr={lr:.5f} '
              f'tl={tl:.3f} ta={ta:5.2f} va={va:5.2f} ema={ea:5.2f} '
              f't={el:.1f}min')
        if (ep + 1) % 15 == 0:
            print('Per-class:', ' '.join([f'C{i+1}:{per_c[i]:.1f}' for i in range(NCLS)]))

        if va > best_v:
            best_v = va
            torch.save(net.state_dict(), out_ckpt)
        if ea > best_e:
            best_e = ea
            torch.save(ema.m.state_dict(), out_ema)

    print(f'Best val={best_v:.2f}% | Best ema={best_e:.2f}%')

# final ensemble + submission 
def make_submission(train_df):
    print('\nEnsemble final + submission')

    test_df = pd.read_csv(os.path.join(ROOT, 'test.csv'))
    test_set = SignalDataset(test_df, os.path.join(ROOT, 'test'),transform=val_transform, is_test=True)
    test_loader = DataLoader(test_set, batch_size=BS * 2, shuffle=False, num_workers=NUM_W, persistent_workers=(NUM_W > 0))

    # logit adjustment - corrects the class bias with log(prior)
    cts = train_df['label'].value_counts().sort_index().values.astype(float)
    priors = cts / cts.sum()
    log_priors = torch.tensor(np.log(priors), dtype=torch.float32)
    print(f'priors: {priors.round(3).tolist()}')

    def adj(p, tau=1.0):
        lp = torch.log(p + 1e-9) - tau * log_priors.unsqueeze(0)
        return F.softmax(lp, dim=1)

    # candidates for ensemble: the 4 existing + 2 new + EMAs
    candidates = [
        # existing (from var4 and var5_ft)
        ('best_model.pth',         1.0),
        ('best_model_ema.pth',     1.0),
        ('best_model_ft.pth',      1.2),     # the best individual (73.10% val)
        ('best_model_ft_ema.pth',  1.2),
        # the new ones from var8
        ('best_v7_a.pth',          1.3),
        ('best_v7_a_ema.pth',      1.3),
        ('best_v7_b.pth',          1.3),
        ('best_v7_b_ema.pth',      1.3),
    ]

    parts, ws = [], 0.0
    for fname, w in candidates:
        p = find_file(fname)
        if not p:
            print(f'  missing: {fname}')
            continue
        m = load_model(p)
        pr = predict_tta(m, test_loader)
        parts.append(adj(pr) * w)
        ws += w
        print(f'  {fname:<28} w={w}')
        del m
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()

    if not parts:
        print('No model available!')
        return

    final = sum(parts) / ws
    preds = final.argmax(1).numpy() + 1

    out_path = os.path.join(ROOT, 'sample_submission.csv')
    pd.DataFrame({'id': test_df['id'], 'label': preds}).to_csv(out_path, index=False)

    dist = pd.Series(preds).value_counts(normalize=True).sort_index() * 100
    exp = [22.6, 19.4, 19.4, 19.4, 19.2]
    print('\n  final distribution:')
    for c in range(1, 6):
        a = dist.get(c, 0.0)
        e = exp[c - 1]
        print(f'Class{c}: {a:5.1f}% (expected ~{e}%, diff {a-e:+.1f})')
    print(f'\n -> {out_path}')


if __name__ == '__main__':
    t_total = time.time()
    train_df = pd.read_csv(os.path.join(ROOT, 'train.csv'))

    for run in RUNS:
        train_one_model(run, train_df)

    make_submission(train_df)
    print(f'\nTotal: {(time.time() - t_total) / 60:.1f} min')