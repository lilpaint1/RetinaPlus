# %% [markdown]
# # AI Diabetic Retinopathy Screening — **v3 (ดันสุด)**
#
# v3 = Ordinal Regression + OptimizedRounder, EfficientNet-B4@380, domain augmentation หนัก,
# EMA weights, grad-accumulation, TTA(hflip+vflip), disk-cache, IDRiD external auto-detect
#
# เป้า: external QWK 0.74 -> ~0.85, กู้ Mild/Severe, binary AUC คงระดับ ~0.95
#
# ก่อนรัน: Settings -> Accelerator -> GPU T4 ; Add Data: eyepacs-aptos-messidor + IDRiD

# %%
# ============================================================
# 0. SETUP — installs, imports, GPU check, seed, config
# ============================================================
import subprocess, sys
def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)
_pip("grad-cam", "timm", "albumentations==1.3.1")   # pin: กัน API เปลี่ยนชื่อ arg ใน 1.4+

import os, json, math, random, time, copy, hashlib, warnings
from collections import deque, Counter
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                             cohen_kappa_score, fbeta_score, roc_auc_score,
                             roc_curve, precision_score, recall_score)
warnings.filterwarnings("ignore")
cv2.setNumThreads(0)

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

assert torch.cuda.is_available(), "GPU ไม่เจอ! Settings -> Accelerator -> GPU"
DEVICE = "cuda"
_gpu = torch.cuda.get_device_properties(0)
print(f"[GPU] {_gpu.name} | VRAM {_gpu.total_memory/1e9:.1f} GB")

# ---- config ----
ON_KAGGLE   = os.path.isdir("/kaggle/input")
INPUT_BASE  = "/kaggle/input" if ON_KAGGLE else os.environ.get("DR_INPUT", "./data/raw")
OUT         = Path("/kaggle/working" if ON_KAGGLE else "./outputs"); OUT.mkdir(parents=True, exist_ok=True)

BACKBONE        = "efficientnet_b4"   # ดันสุด; ถ้าช้า/OOM เปลี่ยนเป็น "efficientnet_b3"
IMG_SIZE        = 380                 # B4 native; B3 ใช้ 320-352 ก็พอ
BATCH           = 16                  # B4@380 บน T4; auto-halve ถ้า OOM
ACCUM           = 2                   # grad accumulation -> effective batch 32
EPOCHS_STAGE1   = 1                   # freeze backbone (regression head warmup)
EPOCHS_STAGE2   = 12                  # unfreeze + cosine
PATIENCE        = 4                   # early stop ที่ val QWK
LR_HEAD         = 1e-3
LR_FULL         = 2e-4
DROP_RATE       = 0.3                 # กัน overfit
DROP_PATH       = 0.2
EMA_DECAY       = 0.999
NUM_CLASSES     = 5
TRAIN_FRAC      = 1.0                 # << แนะนำตั้ง 0.1 ทดสอบก่อน 1 รอบ แล้วค่อยเป็น 1.0
NUM_WORKERS     = os.cpu_count() or 4

CACHE_DIR = Path(("/kaggle/temp" if ON_KAGGLE else ".") + f"/dr_cache_{IMG_SIZE}")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
print(f"[cfg-v3] {BACKBONE}@{IMG_SIZE} batch={BATCH}x{ACCUM} workers={NUM_WORKERS} cache={CACHE_DIR}")

# %%
# ============================================================
# 1. DATASET DISCOVERY — auto-detect combined + IDRiD
# ============================================================
IMG_EXT = {".jpg", ".jpeg", ".png"}
GRADE_MAP = {
    "0": 0, "no_dr": 0, "nodr": 0, "normal": 0,
    "1": 1, "mild": 1,
    "2": 2, "moderate": 2,
    "3": 3, "severe": 3,
    "4": 4, "proliferate_dr": 4, "proliferative_dr": 4, "proliferate": 4,
    "proliferative": 4, "pdr": 4,
}
def norm_label(name):
    k = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    if k in GRADE_MAP: return GRADE_MAP[k]
    return GRADE_MAP.get(k.replace("_", ""), None)

def subdirs(p):
    try: return [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]
    except Exception: return []

def list_images(d):
    out = []
    for r, _, fs in os.walk(d):
        for f in fs:
            if os.path.splitext(f)[1].lower() in IMG_EXT:
                out.append(os.path.join(r, f))
    return out

def find_dir(root, *names):
    m = {s.lower(): s for s in subdirs(root)}
    for n in names:
        if n in m: return os.path.join(root, m[n])
    return None

def collect_class_folders(split_dir):
    rows = []
    for sub in subdirs(split_dir):
        lab = norm_label(sub)
        if lab is None: continue
        for f in list_images(os.path.join(split_dir, sub)):
            rows.append((f, lab))
    return rows

def discover_combined(base):
    q = deque([(base, 0)]); flat = None
    while q:
        d, depth = q.popleft()
        if depth > 6: continue
        subs = subdirs(d); low = [s.lower() for s in subs]
        if "train" in low and any(x in low for x in ("test", "val", "valid", "validation")):
            return ("split", d)
        labs = [s for s in subs if norm_label(s) is not None]
        if len(labs) >= 3 and flat is None:
            flat = ("flat", d)
        for s in subs:
            q.append((os.path.join(d, s), depth + 1))
    return flat

def discover_idrid(base):
    csvs = []
    for r, _, fs in os.walk(base):
        if "idrid" not in r.lower(): continue
        for f in fs:
            if f.lower().endswith(".csv"):
                csvs.append(os.path.join(r, f))
    if not csvs: return None
    csv_path = csvs[0]; root = os.path.dirname(csv_path)
    img_idx = {}
    for p in list_images(root):
        img_idx[os.path.splitext(os.path.basename(p))[0].lower()] = p
    return ([csv_path], img_idx)

comb = discover_combined(INPUT_BASE)
assert comb is not None, f"หาชุด combined ไม่เจอใน {INPUT_BASE}"
print(f"[combined] kind={comb[0]} root={comb[1]}")
idr = discover_idrid(INPUT_BASE)
if idr is None:
    print("[IDRiD] ⚠️ ไม่เจอ — จะข้าม external test")

# %%
# ============================================================
# 2. BUILD INDEX
# ============================================================
def stratified_split(rows, fracs=(0.70, 0.15, 0.15)):
    paths = [r[0] for r in rows]; labs = [r[1] for r in rows]
    p_tr, p_tmp, y_tr, y_tmp = train_test_split(
        paths, labs, train_size=fracs[0], stratify=labs, random_state=SEED)
    rel = fracs[1] / (fracs[1] + fracs[2])
    p_va, p_te, y_va, y_te = train_test_split(
        p_tmp, y_tmp, train_size=rel, stratify=y_tmp, random_state=SEED)
    mk = lambda P, Y: pd.DataFrame({"path": P, "label": Y})
    return mk(p_tr, y_tr), mk(p_va, y_va), mk(p_te, y_te)

if comb[0] == "split":
    root = comb[1]
    d_tr = find_dir(root, "train"); d_va = find_dir(root, "val", "valid", "validation"); d_te = find_dir(root, "test")
    df_train = pd.DataFrame(collect_class_folders(d_tr), columns=["path", "label"])
    df_test  = pd.DataFrame(collect_class_folders(d_te), columns=["path", "label"]) if d_te else None
    if d_va:
        df_val = pd.DataFrame(collect_class_folders(d_va), columns=["path", "label"])
    else:
        df_train, df_val, _ = stratified_split(list(df_train.itertuples(index=False, name=None)), (0.85, 0.15, 0.0001))
else:
    df_train, df_val, df_test = stratified_split(collect_class_folders(comb[1]))

if TRAIN_FRAC < 1.0:
    df_train = df_train.groupby("label", group_keys=False).apply(
        lambda g: g.sample(frac=TRAIN_FRAC, random_state=SEED)).reset_index(drop=True)

df_idrid = None
if idr is not None:
    csvs, img_idx = idr; parts = []
    for c in csvs:
        t = pd.read_csv(c); t.columns = [str(x).strip() for x in t.columns]
        col_img = next((x for x in t.columns if x.lower() in ("id_code", "image", "id") or "image" in x.lower()), t.columns[0])
        col_grd = next((x for x in t.columns if x.lower() in ("diagnosis", "retinopathy grade") or "grade" in x.lower() or "retin" in x.lower() or "diagnos" in x.lower()), t.columns[1])
        for _, r in t.iterrows():
            p = img_idx.get(str(r[col_img]).strip().lower())
            if p is None: continue
            try: lab = int(r[col_grd])
            except Exception: continue
            if 0 <= lab <= 4: parts.append((p, lab))
    if parts:
        df_idrid = pd.DataFrame(parts, columns=["path", "label"]).drop_duplicates("path")

def dist(df): return dict(sorted(Counter(df["label"]).items()))
print(f"[train] {len(df_train)}  dist={dist(df_train)}")
print(f"[val]   {len(df_val)}  dist={dist(df_val)}")
if df_test  is not None: print(f"[internal_test] {len(df_test)}  dist={dist(df_test)}")
if df_idrid is not None: print(f"[external IDRiD] {len(df_idrid)}  dist={dist(df_idrid)}")
json.dump({i: n for i, n in enumerate(CLASS_NAMES)}, open(OUT / "classes.json", "w"), ensure_ascii=False, indent=2)

# %%
# ============================================================
# 3. PREPROCESS (crop + CLAHE) + disk cache
# ============================================================
def _crop_fundus(img, tol=7):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); mask = gray > tol
    if mask.sum() == 0: return img
    c = np.argwhere(mask); y0, x0 = c.min(0); y1, x1 = c.max(0) + 1
    return img[y0:y1, x0:x1]

def _clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB); l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def preprocess_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(path)
    img = _crop_fundus(img)
    if img.size == 0: img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = _clahe(img)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def cached_preprocess(path):
    fp = CACHE_DIR / (hashlib.md5(str(path).encode()).hexdigest() + ".png")
    if fp.exists():
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is not None: return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = preprocess_image(path)
    # tmp ต่อ pid -> หลาย worker ไม่ชนไฟล์เดียวกัน (กัน race FileNotFoundError)
    tmp = CACHE_DIR / (hashlib.md5(str(path).encode()).hexdigest() + f".{os.getpid()}.tmp.png")
    cv2.imwrite(str(tmp), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    try: os.replace(tmp, fp)
    except OSError: pass
    return rgb

# %%
# ============================================================
# 4. DATASET — domain augmentation หนัก (albumentations) ; label = float (regression)
# ============================================================
_MEAN = (0.485, 0.456, 0.406); _STD = (0.229, 0.224, 0.225)
train_aug = A.Compose([
    A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=180, border_mode=cv2.BORDER_CONSTANT, p=0.8),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    A.OneOf([A.GaussianBlur(blur_limit=(3, 5)), A.MotionBlur(blur_limit=5)], p=0.2),
    A.CoarseDropout(max_holes=8, max_height=int(0.08 * IMG_SIZE), max_width=int(0.08 * IMG_SIZE), p=0.3),
    A.Normalize(mean=_MEAN, std=_STD), ToTensorV2(),
])
eval_aug = A.Compose([A.Normalize(mean=_MEAN, std=_STD), ToTensorV2()])

class RetinaDS(Dataset):
    def __init__(self, df, training):
        self.df = df.reset_index(drop=True); self.training = training
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cached_preprocess(r["path"])
        img = (train_aug if self.training else eval_aug)(image=img)["image"]
        return img, float(r["label"])      # << float สำหรับ regression

def make_loader(df, training, batch, sampler=None):
    return DataLoader(RetinaDS(df, training), batch_size=batch,
                      shuffle=(training and sampler is None), sampler=sampler,
                      num_workers=NUM_WORKERS, pin_memory=True, drop_last=training,
                      persistent_workers=(NUM_WORKERS > 0))

def balanced_sampler(df):
    counts = df["label"].value_counts().to_dict()
    w = df["label"].map(lambda c: 1.0 / math.sqrt(counts[c])).values
    return WeightedRandomSampler(torch.DoubleTensor(w), len(w), replacement=True)

# %%
# ============================================================
# 5. MODEL — regression (1 output) + EMA
# ============================================================
def build_model():
    m = timm.create_model(BACKBONE, pretrained=True, num_classes=1,
                          drop_rate=DROP_RATE, drop_path_rate=DROP_PATH)
    return m.to(DEVICE)

def set_trainable(model, backbone_trainable):
    for n, p in model.named_parameters():
        p.requires_grad = True if "classifier" in n else backbone_trainable

class EMA:
    """exponential moving average ของ weights -> นิ่งขึ้น generalize ดีขึ้น"""
    def __init__(self, model, decay):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        for e, m in zip(self.model.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point: e.mul_(self.decay).add_(m.detach(), alpha=1 - self.decay)
            else: e.copy_(m)

# %%
# ============================================================
# 6. METRICS — QWK + OptimizedRounder (หา threshold ปัดเศษที่ดัน QWK สูงสุด)
# ============================================================
def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")

class OptimizedRounder:
    """หาเส้นแบ่ง 4 เส้นบนค่าต่อเนื่อง 0-4 ให้ QWK สูงสุด (สูตรชนะ APTOS)"""
    def __init__(self): self.coef_ = [0.5, 1.5, 2.5, 3.5]
    def fit(self, X, y):
        def loss(c): return -qwk(y, np.digitize(X, sorted(c)))
        r = minimize(loss, self.coef_, method="Nelder-Mead",
                     options={"maxiter": 1000, "xatol": 1e-3, "fatol": 1e-4})
        self.coef_ = sorted(r.x)
    def predict(self, X):
        return np.clip(np.digitize(X, self.coef_), 0, 4).astype(int)

def youden_threshold(y_bin, score):
    fpr, tpr, thr = roc_curve(y_bin, score)
    return thr[np.argmax(tpr - fpr)]

# %%
# ============================================================
# 7. TRAIN — regression(SmoothL1) + grad-accum + EMA + cosine + early-stop(val QWK)
# ============================================================
from tqdm.auto import tqdm

@torch.no_grad()
def infer(model, loader, desc="infer", tta=False):
    """คืนค่าทำนายต่อเนื่อง (regression) + label"""
    model.eval(); P, Y = [], []
    for x, y in tqdm(loader, desc=desc, leave=False):
        x = x.to(DEVICE, non_blocking=True)
        with torch.autocast("cuda"):
            o = model(x).float().squeeze(1)
            if tta:
                o = o + model(torch.flip(x, [3])).float().squeeze(1) + model(torch.flip(x, [2])).float().squeeze(1)
                o = o / 3.0
        o = torch.nan_to_num(o, nan=2.0)
        P.append(o.cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)

def train_one_epoch(model, ema, loader, opt, scaler, crit, sched, ep, total):
    model.train(); run = 0.0; seen = 0
    opt.zero_grad(set_to_none=True)
    bar = tqdm(loader, desc=f"epoch {ep}/{total}", leave=False)
    for i, (x, y) in enumerate(bar):
        x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True).float()
        with torch.autocast("cuda"):
            out = model(x).squeeze(1)
            loss = crit(out, y) / ACCUM
        scaler.scale(loss).backward()
        if (i + 1) % ACCUM == 0:
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            if sched is not None: sched.step()
            ema.update(model)
        run += loss.item() * ACCUM * x.size(0); seen += x.size(0)
        mem = torch.cuda.memory_reserved() / 1e9
        bar.set_postfix(loss=f"{run/seen:.3f}", lr=f"{opt.param_groups[0]['lr']:.1e}", gpu=f"{mem:.1f}G")
    return run / seen

def run_training(batch):
    model = build_model(); ema = EMA(model, EMA_DECAY)
    crit = nn.SmoothL1Loss()
    scaler = torch.cuda.amp.GradScaler()
    tr_loader = make_loader(df_train, True, batch, sampler=balanced_sampler(df_train))
    va_loader = make_loader(df_val, False, batch)
    hist = {k: [] for k in ["tr_loss", "va_qwk", "va_acc", "va_prec", "va_rec"]}
    best_qwk = -1; bad = 0; best_path = OUT / "best.pt"; total = EPOCHS_STAGE1 + EPOCHS_STAGE2

    def monitor(ep, trl, ref):
        vp, vy = infer(ema.model, va_loader, "val")
        rnd = OptimizedRounder(); rnd.fit(vp, vy); pred = rnd.predict(vp)
        vq = qwk(vy, pred); va = float((pred == vy).mean())
        vpr = precision_score(vy, pred, average="macro", zero_division=0)
        vre = recall_score(vy, pred, average="macro", zero_division=0)
        for k, v in zip(hist, [trl, vq, va, vpr, vre]): hist[k].append(float(v))
        print(f"  [{ref}] train_loss={trl:.3f} | val_qwk={vq:.4f} val_acc={va:.3f}")
        return vq

    # STAGE 1: freeze
    set_trainable(model, False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR_HEAD)
    for ep in range(1, EPOCHS_STAGE1 + 1):
        trl = train_one_epoch(model, ema, tr_loader, opt, scaler, crit, None, ep, total)
        monitor(ep, trl, "s1")

    # STAGE 2: unfreeze + cosine
    set_trainable(model, True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_FULL, weight_decay=1e-4)
    steps = max(1, len(tr_loader) // ACCUM) * EPOCHS_STAGE2
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for ep in range(1, EPOCHS_STAGE2 + 1):
        trl = train_one_epoch(model, ema, tr_loader, opt, scaler, crit, sched, EPOCHS_STAGE1 + ep, total)
        vq = monitor(EPOCHS_STAGE1 + ep, trl, "s2")
        if vq > best_qwk:
            best_qwk = vq; bad = 0; torch.save(ema.model.state_dict(), best_path)
            print(f"  ✓ saved best EMA (val QWK={vq:.4f})")
        else:
            bad += 1
            if bad >= PATIENCE: print(f"  early stop (no improve {PATIENCE})"); break
    json.dump(hist, open(OUT / "history.json", "w"), indent=2)
    return model, ema, hist, va_loader, best_path

model = ema = hist = va_loader = best_path = None
for b in [BATCH, BATCH // 2, max(1, BATCH // 4)]:
    try:
        print(f"\n===== START TRAINING ({BACKBONE}@{IMG_SIZE} batch={b}x{ACCUM}) =====")
        model, ema, hist, va_loader, best_path = run_training(b)
        break
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"⚠️ OOM ที่ batch={b} -> ลดครึ่ง"); torch.cuda.empty_cache()
        else: raise
assert model is not None, "เทรนไม่สำเร็จแม้ลด batch แล้ว"

# %%
# ============================================================
# 8. EVALUATE — fit rounder บน val แล้ว apply ลง test (internal + external IDRiD)
# ============================================================
model.load_state_dict(torch.load(best_path, map_location=DEVICE)); model.eval()

vp, vy = infer(model, va_loader, "val(rounder)", tta=True)
ROUNDER = OptimizedRounder(); ROUNDER.fit(vp, vy)
THR = float(ROUNDER.coef_[1])     # เส้นแบ่ง grade1|grade2 = เส้น referable
print(f"[rounder coefs] {[round(c,3) for c in ROUNDER.coef_]}  referable_thr={THR:.3f}")

def plot_confmat(cm, title, fname):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cmn = cm.astype(float) / cm.sum(1, keepdims=True).clip(min=1)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right"); ax.set_yticklabels(CLASS_NAMES)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.colorbar(im, fraction=0.046); fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)

def evaluate(name, df):
    cont, y = infer(model, make_loader(df, False, BATCH), name, tta=True)
    pred = ROUNDER.predict(cont)
    acc = float((pred == y).mean()); kappa = float(qwk(y, pred))
    f2 = float(fbeta_score(y, pred, beta=2, average="macro", zero_division=0))
    rep = classification_report(y, pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0, labels=list(range(5)))
    cm = confusion_matrix(y, pred, labels=list(range(5)))
    plot_confmat(cm, f"Confusion Matrix — {name}", OUT / f"confusion_{name}.png")
    ybin = (y >= 2).astype(int)                  # referable = ใช้ค่าต่อเนื่องตรงๆ
    try: ref_auc = float(roc_auc_score(ybin, cont))
    except Exception: ref_auc = float("nan")
    pos = cont >= THR
    tp = int((pos & (ybin == 1)).sum()); tn = int((~pos & (ybin == 0)).sum())
    fp = int((pos & (ybin == 0)).sum()); fn = int((~pos & (ybin == 1)).sum())
    sens = tp / max(1, tp + fn); spec = tn / max(1, tn + fp)
    fpr, tpr, _ = roc_curve(ybin, cont)
    res = {"n": int(len(y)), "accuracy": acc, "qwk": kappa, "f2_macro": f2,
           "referable": {"auc": ref_auc, "sensitivity": sens, "specificity": spec, "threshold": THR},
           "per_class": {CLASS_NAMES[i]: rep.get(CLASS_NAMES[i], {}) for i in range(5)}}
    print(f"\n=== {name} ===  acc={acc:.3f} QWK={kappa:.4f} F2={f2:.3f} refAUC={ref_auc:.3f} sens={sens:.3f} spec={spec:.3f}")
    return res, (fpr, tpr, ref_auc)

all_res = {}; roc_data = {}
if df_test is not None:
    all_res["internal_test"], roc_data["internal_test (optimistic)"] = evaluate("internal_test", df_test)
if df_idrid is not None:
    all_res["external_idrid"], roc_data["external IDRiD (trustworthy)"] = evaluate("external_idrid", df_idrid)

fig, ax = plt.subplots(figsize=(6.5, 6))
for label, (fpr, tpr, auc) in roc_data.items():
    ax.plot(fpr, tpr, lw=2, label=f"{label}  AUC={auc:.3f}")
ax.plot([0, 1], [0, 1], "--", color="gray")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC — Referable DR (grade ≥ 2)"); ax.legend(loc="lower right"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "roc_curve.png", dpi=130); plt.close(fig)

all_res["config"] = {"backbone": BACKBONE, "img_size": IMG_SIZE, "rounder_coefs": list(map(float, ROUNDER.coef_)),
                     "train_n": len(df_train), "val_n": len(df_val)}
json.dump(all_res, open(OUT / "metrics.json", "w"), ensure_ascii=False, indent=2)
print("\n[saved] metrics.json, confusion_*.png, roc_curve.png")

# %%
# ============================================================
# 9. TRAINING CURVES
# ============================================================
ep = range(1, len(hist["va_qwk"]) + 1)
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs[0, 0].plot(ep, hist["va_qwk"], color="purple"); axs[0, 0].set_title("Validation QWK")
axs[0, 1].plot(ep, hist["tr_loss"], color="blue"); axs[0, 1].set_title("Train Loss (SmoothL1)")
axs[1, 0].plot(ep, hist["va_prec"], color="green"); axs[1, 0].set_title("Validation Precision (macro)")
axs[1, 1].plot(ep, hist["va_rec"], color="orange"); axs[1, 1].set_title("Validation Recall (macro)")
for a in axs.ravel(): a.set_xlabel("Epoch"); a.grid(alpha=0.3)
fig.suptitle("Training & Validation Curves (v3)", fontsize=14)
fig.tight_layout(); fig.savefig(OUT / "training_curves.png", dpi=130); plt.close(fig)
print("[saved] training_curves.png")

# %%
# ============================================================
# 10. GRAD-CAM — ภาพเปรียบเทียบดวงตา (regression target)
# ============================================================
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

class RegTarget:
    def __call__(self, model_output): return model_output[0]

target_layer = model.conv_head if hasattr(model, "conv_head") else model.blocks[-1]
cam = GradCAM(model=model, target_layers=[target_layer])

src_df = df_idrid if df_idrid is not None else df_test
samples = []
for g in range(5):
    sub = src_df[src_df["label"] == g]
    if len(sub): samples.append((sub.iloc[0]["path"], g))

n = len(samples); fig, axs = plt.subplots(n, 3, figsize=(10, 3.2 * n))
if n == 1: axs = axs[None, :]
for row, (path, g) in enumerate(samples):
    rgb = preprocess_image(path); rgbf = rgb.astype(np.float32) / 255.0
    x = eval_aug(image=rgb)["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        cont = float(model(x).item())
    pg = int(ROUNDER.predict(np.array([cont]))[0])
    gray = cam(input_tensor=x, targets=[RegTarget()])[0]
    overlay = show_cam_on_image(rgbf, gray, use_rgb=True)
    orig = cv2.cvtColor(cv2.resize(cv2.imread(str(path)), (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
    axs[row, 0].imshow(orig); axs[row, 0].set_ylabel(f"True: {CLASS_NAMES[g]}", fontsize=11)
    axs[row, 1].imshow(rgb); axs[row, 2].imshow(overlay)
    if row == 0:
        axs[row, 0].set_title("Original"); axs[row, 1].set_title("Preprocessed"); axs[row, 2].set_title("Grad-CAM")
    axs[row, 2].set_xlabel(f"Pred: {CLASS_NAMES[pg]} (score={cont:.2f})")
    for c in range(3): axs[row, c].set_xticks([]); axs[row, c].set_yticks([])
fig.suptitle("Grad-CAM — Eye Comparison across DR Severity (v3)", fontsize=14)
fig.tight_layout(); fig.savefig(OUT / "gradcam_eyes.png", dpi=130); plt.close(fig)
print("[saved] gradcam_eyes.png")

# %%
# ============================================================
# 11. PREDICT — inference รูปเดียว (เป็น/ไม่เป็น + ระดับ)
# ============================================================
@torch.no_grad()
def predict_image(path, show=True):
    rgb = preprocess_image(path)
    x = eval_aug(image=rgb)["image"].unsqueeze(0).to(DEVICE)
    cont = float(model(x).item())
    grade = int(ROUNDER.predict(np.array([cont]))[0]); referable = cont >= THR
    if show:
        verdict = "🔴 พบความผิดปกติ — ควรพบจักษุแพทย์" if referable else "🟢 ปกติ — ไม่ต้องส่งต่อ"
        print(f"\n{os.path.basename(str(path))}")
        print(f"  score={cont:.2f} -> ระดับ: {CLASS_NAMES[grade]} (grade {grade})")
        print(f"  ผล: {verdict}")
    return {"grade": grade, "grade_name": CLASS_NAMES[grade], "referable": bool(referable), "score": cont}

demo_df = df_idrid if df_idrid is not None else df_test
if demo_df is not None:
    print("\n========== DEMO PREDICTIONS ==========")
    for p in demo_df.sample(min(6, len(demo_df)), random_state=SEED)["path"]:
        predict_image(p)

print("\n✅ เสร็จสิ้น (v3) — ผลใน:", OUT)
