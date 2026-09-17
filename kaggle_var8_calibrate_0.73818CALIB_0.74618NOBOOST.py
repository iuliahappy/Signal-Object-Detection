import os, math, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

NCLS, BS, BN_MOM, DO, CH, BLOCKS = 5, 128, 0.05, 0.3, 64, 2

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")

ROOT = os.path.dirname(os.path.abspath(__file__))

# arhitecture
class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c, momentum=BN_MOM)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_c, momentum=BN_MOM)
        self.act   = nn.SiLU()
        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c, momentum=BN_MOM),
            )
    def forward(self, x):
        idn = x if self.downsample is None else self.downsample(x)
        o = self.act(self.bn1(self.conv1(x)))
        o = self.bn2(self.conv2(o))
        return self.act(o + idn)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        c = CH
        self.stem = nn.Sequential(
            nn.Conv2d(3, c, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c, momentum=BN_MOM),
            nn.SiLU(),
        )
        stages = []
        ic = c
        for i, oc in enumerate([c, c*2, c*4, c*8]):
            s = 1 if i == 0 else 2
            blocks = []
            for b in range(BLOCKS):
                ss = s if b == 0 else 1
                blocks.append(BasicBlock(ic, oc, ss))
                ic = oc
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.head_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(DO)
        self.fc = nn.Linear(ic, NCLS)
    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.head_pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

val_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.25]*3),
])


class Dset(Dataset):
    def __init__(self, df, d, tf, is_test=False):
        self.df = df.reset_index(drop=True)
        self.d = d
        self.tf = tf
        self.is_test = is_test
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        img = Image.open(os.path.join(self.d, self.df['id'].iloc[i])).convert('RGB')
        img = self.tf(img)
        if self.is_test:
            return img
        return img, int(self.df['label'].iloc[i]) - 1


def find_file(name):
    if os.path.exists(name):
        return os.path.abspath(name)
    for r, _, fs in os.walk(ROOT):
        if name in fs:
            return os.path.join(r, name)
    return None


def load_model(p):
    m = Net().to(device)
    m.load_state_dict(torch.load(p, map_location = device, weights_only = True))
    m.eval()
    return m


@torch.no_grad()
def predict_tta(m, loader):
    m.eval()
    out = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device)
        p  = F.softmax(m(x), dim=1)
        p += F.softmax(m(torch.flip(x, [3])), dim=1)
        p += F.softmax(m(torch.flip(x, [2])), dim=1)
        p += F.softmax(m(torch.flip(x, [2, 3])), dim=1)
        out.append((p / 4.0).cpu().numpy())
    return np.concatenate(out, 0)


#data
train_df = pd.read_csv(os.path.join(ROOT, 'train.csv'))
test_df  = pd.read_csv(os.path.join(ROOT, 'test.csv'))

_, va_df = train_test_split(
    train_df, test_size=0.1, random_state=2024, stratify=train_df['label'])
print(f'Val set: {len(va_df)} (split - model A din var8)')

va_set = Dset(va_df, os.path.join(ROOT, 'train'), val_tf)
te_set = Dset(test_df, os.path.join(ROOT, 'test'), val_tf, is_test=True)
va_loader = DataLoader(va_set, batch_size=BS, shuffle=False, num_workers=0)
te_loader = DataLoader(te_set, batch_size=BS, shuffle=False, num_workers=0)

# TTA for all models
candidates = [
    'best_model.pth',
    'best_model_ema.pth',
    'best_model_ft.pth',
    'best_model_ft_ema.pth',
    'best_v7_a.pth',
    'best_v7_a_ema.pth',
    'best_v7_b.pth',
    'best_v7_b_ema.pth',
]

va_probs, te_probs, names = [], [], []
t0 = time.time()
for fn in candidates:
    p = find_file(fn)
    if not p:
        print(f'  LIPSA {fn}')
        continue
    print(f'  TTA: {fn} ...', end=' ', flush=True)
    m = load_model(p)
    va_probs.append(predict_tta(m, va_loader))
    te_probs.append(predict_tta(m, te_loader))
    names.append(fn)
    del m
    if hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    print(f'done ({(time.time()-t0):.0f}s total)')

print(f'\nEnsemble: {len(names)} modele')

va_arr = np.stack(va_probs, 0)
te_arr = np.stack(te_probs, 0)

# saving probabilities
np.save(os.path.join(ROOT, 'va_probs.npy'), va_arr)
np.save(os.path.join(ROOT, 'te_probs.npy'), te_arr)
with open(os.path.join(ROOT, 'va_names.txt'), 'w') as f:
    f.write('\n'.join(names))


# calibration
va_y = va_df['label'].values - 1

va_avg = va_arr.mean(0)
te_avg = te_arr.mean(0)

def acc(p, y):
    return (p.argmax(1) == y).mean() * 100

def c_dist(p):
    pr = p.argmax(1) + 1
    d = pd.Series(pr).value_counts(normalize=True).sort_index() * 100
    return [d.get(c, 0.0) for c in range(1, 6)]

base_acc = acc(va_avg, va_y)
base_d   = c_dist(va_avg)
print(f'\nBaseline (no boost): val={base_acc:.2f}%')
print(f'Dist: C1:{base_d[0]:.1f} C2:{base_d[1]:.1f} C3:{base_d[2]:.1f} C4:{base_d[3]:.1f} C5:{base_d[4]:.1f}')


# Grid coarse: b1 vs b_other 
print('\nGrid coarse: b1 vs b_other')
best_acc, best_b = base_acc, np.ones(5)
for b1 in np.arange(0.55, 1.01, 0.05):
    for bo in np.arange(1.00, 1.21, 0.025):
        boost = np.array([b1, bo, bo, bo, bo])
        p = va_avg * boost
        p = p / p.sum(1, keepdims=True)
        a = acc(p, va_y)
        if a > best_acc:
            best_acc = a
            best_b = boost
            print(f'b1={b1:.2f} bo={bo:.3f} -> {a:.2f}%')

print(f'\nCoarse best: {best_b.round(3)} acc={best_acc:.2f}%')


# fine grid
print('\nFine grid: per-class')
b1_b = best_b[0]
bo_b = best_b[1]
for b2 in np.arange(bo_b-0.06, bo_b+0.07, 0.03):
    for b3 in np.arange(bo_b-0.06, bo_b+0.07, 0.03):
        for b4 in np.arange(bo_b-0.06, bo_b+0.07, 0.03):
            for b5 in np.arange(bo_b-0.06, bo_b+0.07, 0.03):
                boost = np.array([b1_b, b2, b3, b4, b5])
                p = va_avg * boost
                p = p / p.sum(1, keepdims=True)
                a = acc(p, va_y)
                if a > best_acc:
                    best_acc = a
                    best_b = boost
                    print(f'  {boost.round(3)} -> {a:.2f}%')


# fine grid
print('\nFine grid : b1')
for b1 in np.arange(best_b[0]-0.05, best_b[0]+0.06, 0.02):
    boost = best_b.copy()
    boost[0] = b1
    p = va_avg * boost
    p = p / p.sum(1, keepdims=True)
    a = acc(p, va_y)
    if a > best_acc:
        best_acc = a
        best_b = boost
        print(f'  b1={b1:.3f} -> {a:.2f}%')


# comparison with logit adjustment standard 
print('\nLogit adjustment with tau')
priors = train_df['label'].value_counts(normalize=True).sort_index().values
lp = np.log(priors)
for tau in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
    z = np.log(va_avg + 1e-9) - tau * lp
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    p = p / p.sum(1, keepdims=True)
    a = acc(p, va_y)
    d = c_dist(p)
    print(f'  tau={tau:.2f} val={a:.2f}% dist=[{d[0]:.1f},{d[1]:.1f},{d[2]:.1f},{d[3]:.1f},{d[4]:.1f}]')


# Weighted ensemble + best boost 
print('\nUneven weights on models')
# proposed weights: the new var8 should weigh more because they have a higher acc wave
ws_list = [
    np.array([1.0,1.0,1.2,1.2,1.3,1.3,1.3,1.3]),  # strong var7
    np.array([0.8,0.8,1.0,1.0,1.3,1.3,1.5,1.5]),  # accent pe va7 b
    np.array([1.0]*8),                              # uniform
    np.array([0.7,0.7,1.0,1.0,1.4,1.4,1.4,1.4]),  # more for var7
]

for ws in ws_list:
    ws_n = ws[:len(names)]
    ws_norm = ws_n / ws_n.sum() * len(ws_n)
    va_w = (va_arr * ws_norm[:, None, None]).mean(0)
    p = va_w * best_b
    p = p / p.sum(1, keepdims=True)
    a = acc(p, va_y)
    print(f'  ws={list(ws_n)} -> val={a:.2f}%')


# final pe test cu best_b + weighted ensemble 
# using weights uniforme + boost optim 
print(f'\nFinal')
print(f'Boost: {best_b.round(3)}')
print(f'Val acc: {best_acc:.2f}%')

te_avg = te_arr.mean(0)
te_adj = te_avg * best_b
te_adj = te_adj / te_adj.sum(1, keepdims=True)
preds = te_adj.argmax(1) + 1

d = pd.Series(preds).value_counts(normalize=True).sort_index() * 100
exp = [22.6, 19.4, 19.4, 19.4, 19.2]
print('\nTest distribution after calibration:')
for c in range(1, 6):
    a = d.get(c, 0.0)
    e = exp[c-1]
    print(f'Class{c}: {a:5.1f}% (waited ~{e}, diff {a-e:+.1f})')

out = os.path.join(ROOT, 'sample_submission_calib.csv')
pd.DataFrame({'id': test_df['id'], 'label': preds}).to_csv(out, index=False)
print(f'\n  -> {out}')

# saving  also no-boost
preds_nb = te_avg.argmax(1) + 1
pd.DataFrame({'id': test_df['id'], 'label': preds_nb}).to_csv(os.path.join(ROOT, 'sample_submission_noboost.csv'), index=False)
print(f'sample_submission_noboost.csv (ensemble fara boost)')

print(f'\nTotal: {(time.time()-t0)/60:.1f} min')
