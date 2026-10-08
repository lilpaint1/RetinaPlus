# %% [markdown]
# # AI Diabetic Retinopathy Screening — EfficientNet-B3 (PyTorch)
#
# End-to-end DR screening จากภาพ fundus — รันบน **Kaggle GPU** (P100/T4)
#
# **โมเดลตัวเดียว ออก 2 ผลลัพธ์:**
# - **Binary "Referable DR"**: grade 0-1 = NEGATIVE (ไม่ต้องส่งต่อ), 2-4 = POSITIVE (พบจักษุแพทย์)
# - **Severity** ระดับ 0-4 เต็ม
#
# **การแบ่งข้อมูล (กัน leakage):**
# - TRAIN + VAL = ชุด combined (EyePACS+APTOS+Messidor)
# - **EXTERNAL TEST = IDRiD** (โมเดลไม่เคยเห็น → คะแนนน่าเชื่อถือ)
#
# **ก่อนรัน:** Settings → Accelerator → **GPU** , แล้ว Add Data 2 ชุด:
# `eyepacs-aptos-messidor-diabetic-retinopathy` + `IDRiD`

# %%
# ============================================================
# 0. SETUP — installs, imports, GPU check, seed, config
# ============================================================
import subprocess, sys
def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)
_pip("grad-cam", "timm")

import os, json, math, random, time, warnings
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
import torchvision.transforms as T
from PIL import Image
import timm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                             cohen_kappa_score, fbeta_score, roc_auc_score,
                             roc_curve, precision_score, recall_score)
warnings.filterwarnings("ignore")

# ---- reproducibility ----
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

# ---- GPU บังคับ (ห้าม fallback CPU เงียบๆ) ----
assert torch.cuda.is_available(), (
    "GPU ไม่เจอ! บน Kaggle ไปที่ Settings → Accelerator → GPU (P100/T4) แล้วรันใหม่")
DEVICE = "cuda"
_gpu = torch.cuda.get_device_properties(0)
print(f"[GPU] {_gpu.name} | VRAM {_gpu.total_memory/1e9:.1f} GB")

# ---- config (แก้ตรงนี้ที่เดียว) ----
ON_KAGGLE   = os.path.isdir("/kaggle/input")
INPUT_BASE  = "/kaggle/input" if ON_KAGGLE else os.environ.get("DR_INPUT", "./data/raw")
OUT         = Path("/kaggle/working" if ON_KAGGLE else "./outputs"); OUT.mkdir(parents=True, exist_ok=True)

IMG_SIZE        = 300          # EfficientNet-B3 native
BATCH           = 24           # ปลอดภัยบน P100/T4 16GB @300px (มี auto-halve กัน OOM)
EPOCHS_STAGE1   = 2            # freeze backbone, เทรนแต่ head
EPOCHS_STAGE2   = 12           # unfreeze ทั้งตัว + cosine LR
PATIENCE        = 4            # early stopping (วัดที่ val QWK)
LR_HEAD         = 1e-3
LR_FULL         = 3e-4
LABEL_SMOOTH    = 0.1
NUM_CLASSES     = 5
TRAIN_FRAC      = 1.0          # ลดเหลือ 0.15 เพื่อทดสอบเร็วๆ ได้
NUM_WORKERS     = min(4, os.cpu_count() or 2)

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
print(f"[cfg] img={IMG_SIZE} batch={BATCH} workers={NUM_WORKERS} out={OUT}")

# %%
# ============================================================
# 1. DATASET DISCOVERY — auto-detect combined + IDRiD
#    (ไม่ hardcode path — สแกนหาเอง กัน mismatch)
# ============================================================
IMG_EXT = {".jpg", ".jpeg", ".png"}

# map ชื่อโฟลเดอร์/label หลายแบบ -> เกรด 0-4
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
    k2 = k.replace("_", "")
    return GRADE_MAP.get(k2, None)

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
    """รวม (path,label) จากโฟลเดอร์คลาสภายใต้ split_dir"""
    rows = []
    for sub in subdirs(split_dir):
        lab = norm_label(sub)
        if lab is None: continue
        for f in list_images(os.path.join(split_dir, sub)):
            rows.append((f, lab))
    return rows

def discover_combined(base):
    """หา root ของชุด combined: คืน ('split',root) ถ้ามี train/val/test, ('flat',root) ถ้าเป็นโฟลเดอร์คลาสล้วน"""
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
    """หา IDRiD: คืน (csv_list, image_index) — จับ csv ที่อยู่ใน path ที่มีคำว่า 'idrid'"""
    csvs = []
    for r, _, fs in os.walk(base):
        if "idrid" not in r.lower(): continue
        for f in fs:
            if f.lower().endswith(".csv"):
                csvs.append(os.path.join(r, f))
    if not csvs: return None
    csv_path = csvs[0]
    # index รูป "เฉพาะใน root ของ IDRiD" (ไม่ไล่ขึ้นไปแตะ combined)
    root = os.path.dirname(csv_path)
    img_idx = {}
    for p in list_images(root):
        img_idx[os.path.splitext(os.path.basename(p))[0].lower()] = p
    return ([csv_path], img_idx)

# ---- หา combined (เลือก root ที่มีรูปเยอะสุด) ----
comb = discover_combined(INPUT_BASE)
assert comb is not None, f"หาชุด combined ไม่เจอใน {INPUT_BASE} — เช็คว่า Add Data แล้วหรือยัง"
print(f"[combined] kind={comb[0]} root={comb[1]}")

# ---- หา IDRiD (external test) ----
idr = discover_idrid(INPUT_BASE)
if idr is None:
    print("[IDRiD] ⚠️ ไม่เจอ — จะข้าม external test (เพิ่ม Add Data 'IDRiD' เพื่อให้ครบ)")

# %%
# ============================================================
# 2. BUILD INDEX — สร้าง train / val / internal_test / external(IDRiD)
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
    d_tr = find_dir(root, "train")
    d_va = find_dir(root, "val", "valid", "validation")
    d_te = find_dir(root, "test")
    df_train = pd.DataFrame(collect_class_folders(d_tr), columns=["path", "label"])
    df_test  = pd.DataFrame(collect_class_folders(d_te), columns=["path", "label"]) if d_te else None
    if d_va:
        df_val = pd.DataFrame(collect_class_folders(d_va), columns=["path", "label"])
    else:  # ไม่มี val -> เจียดจาก train 85/15
        df_train, df_val, _ = stratified_split(list(df_train.itertuples(index=False, name=None)),
                                               (0.85, 0.15, 0.0001))
else:  # flat -> แบ่งเอง 70/15/15
    rows = collect_class_folders(comb[1])
    df_train, df_val, df_test = stratified_split(rows)

# subsample เพื่อทดสอบเร็ว (ถ้า TRAIN_FRAC<1)
if TRAIN_FRAC < 1.0:
    df_train = df_train.groupby("label", group_keys=False).apply(
        lambda g: g.sample(frac=TRAIN_FRAC, random_state=SEED)).reset_index(drop=True)

# ---- IDRiD external test ----
df_idrid = None
if idr is not None:
    csvs, img_idx = idr
    parts = []
    for c in csvs:
        t = pd.read_csv(c)
        t.columns = [str(x).strip() for x in t.columns]
        col_img = next((x for x in t.columns if x.lower() in ("id_code", "image", "id") or "image" in x.lower()), t.columns[0])
        col_grd = next((x for x in t.columns if x.lower() in ("diagnosis", "retinopathy grade") or "grade" in x.lower() or "retin" in x.lower() or "diagnos" in x.lower()), t.columns[1])
        for _, r in t.iterrows():
            key = str(r[col_img]).strip().lower()
            p = img_idx.get(key)
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
# 3. PREPROCESS — circle-crop + Ben Graham + CLAHE  (ฟังก์ชันเดียว ใช้ทุกที่)
# ============================================================
def _crop_fundus(img, tol=7):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > tol
    if mask.sum() == 0: return img
    c = np.argwhere(mask)
    y0, x0 = c.min(0); y1, x1 = c.max(0) + 1
    return img[y0:y1, x0:x1]

def _ben_graham(img):
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=IMG_SIZE / 30.0)
    return cv2.addWeighted(img, 4, blur, -4, 128)

def _clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def preprocess_image(path):
    """อ่านรูป -> RGB uint8 (IMG_SIZE,IMG_SIZE,3) — ใช้ทั้ง train/test/predict ให้เหมือนกัน"""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(path)
    img = _crop_fundus(img)
    if img.size == 0: img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = _ben_graham(img)
    img = _clahe(img)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# %%
# ============================================================
# 4. DATASET / DATALOADER — augment เฉพาะ train + balanced sampler
# ============================================================
_MEAN = [0.485, 0.456, 0.406]; _STD = [0.229, 0.224, 0.225]
train_tf = T.Compose([
    T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
    T.RandomRotation(20), T.ColorJitter(0.1, 0.1, 0.1),
    T.ToTensor(), T.Normalize(_MEAN, _STD),
])
eval_tf = T.Compose([T.ToTensor(), T.Normalize(_MEAN, _STD)])

class RetinaDS(Dataset):
    def __init__(self, df, training):
        self.df = df.reset_index(drop=True); self.training = training
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.fromarray(preprocess_image(r["path"]))
        img = train_tf(img) if self.training else eval_tf(img)
        return img, int(r["label"])

def make_loader(df, training, batch, sampler=None):
    return DataLoader(RetinaDS(df, training), batch_size=batch,
                      shuffle=(training and sampler is None), sampler=sampler,
                      num_workers=NUM_WORKERS, pin_memory=True, drop_last=training)

def balanced_sampler(df):
    counts = df["label"].value_counts().to_dict()
    w = df["label"].map(lambda c: 1.0 / counts[c]).values
    return WeightedRandomSampler(torch.DoubleTensor(w), len(w), replacement=True)

# %%
# ============================================================
# 5. MODEL — EfficientNet-B3 (timm, pretrained)
# ============================================================
def build_model():
    m = timm.create_model("efficientnet_b3", pretrained=True, num_classes=NUM_CLASSES)
    return m.to(DEVICE)

def set_trainable(model, backbone_trainable):
    for n, p in model.named_parameters():
        p.requires_grad = True if "classifier" in n else backbone_trainable

# %%
# ============================================================
# 6. METRICS helpers
# ============================================================
def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")

def referable_score(probs):       # P(grade>=2)
    return probs[:, 2:].sum(axis=1)

def youden_threshold(y_bin, score):
    fpr, tpr, thr = roc_curve(y_bin, score)
    return thr[np.argmax(tpr - fpr)]

# %%
# ============================================================
# 7. TRAIN — 2-stage, AMP, tqdm progress bar, checkpoint+early-stop
# ============================================================
from tqdm.auto import tqdm

@torch.no_grad()
def infer(model, loader, desc="infer"):
    model.eval(); P, Y = [], []
    for x, y in tqdm(loader, desc=desc, leave=False):
        x = x.to(DEVICE, non_blocking=True)
        with torch.autocast("cuda"):
            out = model(x)
        P.append(torch.softmax(out.float(), 1).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)

def train_one_epoch(model, loader, opt, scaler, crit, sched, ep, total_ep):
    model.train(); run = 0.0; seen = 0; correct = 0
    bar = tqdm(loader, desc=f"epoch {ep}/{total_ep}", leave=False)
    for x, y in bar:
        x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda"):
            out = model(x); loss = crit(out, y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if sched is not None: sched.step()
        run += loss.item() * x.size(0); seen += x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        mem = torch.cuda.memory_reserved() / 1e9
        bar.set_postfix(loss=f"{run/seen:.3f}", acc=f"{correct/seen:.3f}",
                        lr=f"{opt.param_groups[0]['lr']:.1e}", gpu=f"{mem:.1f}G")
    return run / seen, correct / seen

def run_training(batch):
    model = build_model()
    crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    scaler = torch.cuda.amp.GradScaler()
    tr_loader = make_loader(df_train, True, batch, sampler=balanced_sampler(df_train))
    va_loader = make_loader(df_val, False, batch)

    hist = {k: [] for k in ["tr_loss", "tr_acc", "va_loss", "va_acc", "va_qwk", "va_prec", "va_rec"]}
    best_qwk = -1; bad = 0; best_path = OUT / "best.pt"

    # ---------- STAGE 1: freeze backbone ----------
    set_trainable(model, False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR_HEAD)
    for ep in range(1, EPOCHS_STAGE1 + 1):
        trl, tra = train_one_epoch(model, tr_loader, opt, scaler, crit, None, ep, EPOCHS_STAGE1 + EPOCHS_STAGE2)
        vp, vy = infer(model, va_loader, "val")
        _log_epoch(hist, model, best_path, trl, tra, vp, vy, ref="s1")

    # ---------- STAGE 2: unfreeze all + cosine ----------
    set_trainable(model, True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_FULL, weight_decay=1e-4)
    steps = max(1, len(tr_loader)) * EPOCHS_STAGE2
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for ep in range(1, EPOCHS_STAGE2 + 1):
        gep = EPOCHS_STAGE1 + ep
        trl, tra = train_one_epoch(model, tr_loader, opt, scaler, crit, sched, gep, EPOCHS_STAGE1 + EPOCHS_STAGE2)
        vp, vy = infer(model, va_loader, "val")
        cur = _log_epoch(hist, model, best_path, trl, tra, vp, vy, ref="s2")
        if cur > best_qwk:
            best_qwk = cur; bad = 0; torch.save(model.state_dict(), best_path)
            print(f"  ✓ saved best (val QWK={cur:.4f})")
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"  early stop (no QWK improve {PATIENCE} epochs)"); break
    json.dump(hist, open(OUT / "history.json", "w"), indent=2)
    return model, hist, va_loader, best_path

def _log_epoch(hist, model, best_path, trl, tra, vp, vy, ref):
    vpred = vp.argmax(1)
    vloss = float(nn.functional.cross_entropy(torch.tensor(np.log(vp + 1e-9)), torch.tensor(vy)).item())
    vq = qwk(vy, vpred); vpr = precision_score(vy, vpred, average="macro", zero_division=0)
    vre = recall_score(vy, vpred, average="macro", zero_division=0)
    va = (vpred == vy).mean()
    for k, v in zip(["tr_loss", "tr_acc", "va_loss", "va_acc", "va_qwk", "va_prec", "va_rec"],
                    [trl, tra, vloss, va, vq, vpr, vre]):
        hist[k].append(float(v))
    print(f"  [{ref}] train_loss={trl:.3f} train_acc={tra:.3f} | "
          f"val_acc={va:.3f} val_qwk={vq:.4f} val_loss={vloss:.3f}")
    return vq

# ---- รัน (auto-halve batch ถ้า OOM) ----
model = hist = va_loader = best_path = None
for b in [BATCH, BATCH // 2, BATCH // 4]:
    try:
        print(f"\n===== START TRAINING (batch={b}) =====")
        model, hist, va_loader, best_path = run_training(b)
        break
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"⚠️ OOM ที่ batch={b} -> ลดครึ่งแล้วลองใหม่"); torch.cuda.empty_cache()
        else:
            raise
assert model is not None, "เทรนไม่สำเร็จแม้ลด batch แล้ว"

# %%
# ============================================================
# 8. EVALUATE — internal test + external IDRiD  (binary + 5-class)
# ============================================================
model.load_state_dict(torch.load(best_path, map_location=DEVICE)); model.eval()

# threshold สำหรับ referable เลือกจาก VAL (Youden) แล้วเอาไปใช้กับ test
vp, vy = infer(model, va_loader, "val(thr)")
THR = float(youden_threshold((vy >= 2).astype(int), referable_score(vp)))
print(f"[referable threshold @val] = {THR:.3f}")

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
    loader = make_loader(df, False, BATCH)
    probs, y = infer(model, loader, name)
    pred = probs.argmax(1)
    # ----- 5-class -----
    acc = float((pred == y).mean()); kappa = float(qwk(y, pred))
    f2 = float(fbeta_score(y, pred, beta=2, average="macro", zero_division=0))
    rep = classification_report(y, pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    present = sorted(set(y))
    try:
        macro_auc = float(roc_auc_score(y, probs, multi_class="ovr", average="macro",
                                        labels=list(range(5))))
    except Exception:
        macro_auc = float("nan")
    cm = confusion_matrix(y, pred, labels=list(range(5)))
    plot_confmat(cm, f"Confusion Matrix — {name}", OUT / f"confusion_{name}.png")
    # ----- binary referable -----
    ybin = (y >= 2).astype(int); sc = referable_score(probs)
    try: ref_auc = float(roc_auc_score(ybin, sc))
    except Exception: ref_auc = float("nan")
    pos = sc >= THR
    tp = int(((pos == 1) & (ybin == 1)).sum()); tn = int(((pos == 0) & (ybin == 0)).sum())
    fp = int(((pos == 1) & (ybin == 0)).sum()); fn = int(((pos == 0) & (ybin == 1)).sum())
    sens = tp / max(1, tp + fn); spec = tn / max(1, tn + fp)
    res = {"n": int(len(y)), "accuracy": acc, "qwk": kappa, "f2_macro": f2,
           "macro_auc_ovr": macro_auc,
           "referable": {"auc": ref_auc, "sensitivity": sens, "specificity": spec, "threshold": THR},
           "per_class": {CLASS_NAMES[i]: rep.get(CLASS_NAMES[i], {}) for i in range(5)}}
    fpr, tpr, _ = roc_curve(ybin, sc)
    print(f"\n=== {name} ===  acc={acc:.3f}  QWK={kappa:.4f}  F2={f2:.3f}  "
          f"refAUC={ref_auc:.3f}  sens={sens:.3f}  spec={spec:.3f}")
    return res, (fpr, tpr, ref_auc)

all_res = {}; roc_data = {}
if df_test is not None:
    all_res["internal_test"], roc_data["internal_test (optimistic)"] = evaluate("internal_test", df_test)
if df_idrid is not None:
    all_res["external_idrid"], roc_data["external IDRiD (trustworthy)"] = evaluate("external_idrid", df_idrid)

# ----- ROC curve รวม -----
fig, ax = plt.subplots(figsize=(6.5, 6))
for label, (fpr, tpr, auc) in roc_data.items():
    ax.plot(fpr, tpr, lw=2, label=f"{label}  AUC={auc:.3f}")
ax.plot([0, 1], [0, 1], "--", color="gray")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC — Referable DR (grade ≥ 2)"); ax.legend(loc="lower right"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "roc_curve.png", dpi=130); plt.close(fig)

all_res["config"] = {"img_size": IMG_SIZE, "backbone": "efficientnet_b3",
                     "train_n": len(df_train), "val_n": len(df_val), "referable_threshold": THR}
json.dump(all_res, open(OUT / "metrics.json", "w"), ensure_ascii=False, indent=2)
print("\n[saved] metrics.json, confusion_*.png, roc_curve.png")

# %%
# ============================================================
# 9. TRAINING CURVES — accuracy / loss / precision / recall (แบบสไลด์)
# ============================================================
ep = range(1, len(hist["tr_acc"]) + 1)
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs[0, 0].plot(ep, hist["tr_acc"], label="Train"); axs[0, 0].plot(ep, hist["va_acc"], label="Val")
axs[0, 0].set_title("Accuracy"); axs[0, 0].legend()
axs[0, 1].plot(ep, hist["tr_loss"], label="Train"); axs[0, 1].plot(ep, hist["va_loss"], label="Val")
axs[0, 1].set_title("Loss"); axs[0, 1].legend()
axs[1, 0].plot(ep, hist["va_prec"], color="green"); axs[1, 0].set_title("Validation Precision (macro)")
axs[1, 1].plot(ep, hist["va_rec"], color="orange"); axs[1, 1].set_title("Validation Recall (macro)")
for a in axs.ravel(): a.set_xlabel("Epoch"); a.grid(alpha=0.3)
fig.suptitle("Training & Validation Curves", fontsize=14)
fig.tight_layout(); fig.savefig(OUT / "training_curves.png", dpi=130); plt.close(fig)
print("[saved] training_curves.png")

# %%
# ============================================================
# 10. GRAD-CAM — ภาพเปรียบเทียบดวงตา (original | preprocessed | heatmap) ต่อเกรด
# ============================================================
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

target_layer = model.conv_head if hasattr(model, "conv_head") else model.blocks[-1]
cam = GradCAM(model=model, target_layers=[target_layer])

# เลือกตัวอย่าง 1 รูป/เกรด จากชุด test ที่มี (external ก่อน ไม่งั้น internal)
src_df = df_idrid if df_idrid is not None else df_test
samples = []
for g in range(5):
    sub = src_df[src_df["label"] == g]
    if len(sub): samples.append((sub.iloc[0]["path"], g))

n = len(samples)
fig, axs = plt.subplots(n, 3, figsize=(10, 3.2 * n))
if n == 1: axs = axs[None, :]
for row, (path, g) in enumerate(samples):
    rgb = preprocess_image(path)
    rgbf = rgb.astype(np.float32) / 255.0
    x = eval_tf(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = torch.softmax(model(x).float(), 1)[0].cpu().numpy()
    pred = int(prob.argmax())
    gray = cam(input_tensor=x, targets=[ClassifierOutputTarget(g)])[0]
    overlay = show_cam_on_image(rgbf, gray, use_rgb=True)
    orig = cv2.cvtColor(cv2.resize(cv2.imread(str(path)), (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
    axs[row, 0].imshow(orig);    axs[row, 0].set_ylabel(f"True: {CLASS_NAMES[g]}", fontsize=11)
    axs[row, 1].imshow(rgb);     axs[row, 1].set_title("Preprocessed" if row == 0 else "")
    axs[row, 2].imshow(overlay); axs[row, 2].set_title("Grad-CAM" if row == 0 else "")
    if row == 0: axs[row, 0].set_title("Original")
    axs[row, 2].set_xlabel(f"Pred: {CLASS_NAMES[pred]} ({prob[pred]*100:.0f}%)")
    for c in range(3): axs[row, c].set_xticks([]); axs[row, c].set_yticks([])
fig.suptitle("Grad-CAM — Eye Comparison across DR Severity", fontsize=14)
fig.tight_layout(); fig.savefig(OUT / "gradcam_eyes.png", dpi=130); plt.close(fig)
print("[saved] gradcam_eyes.png")

# %%
# ============================================================
# 11. PREDICT — inference รูปเดียว (เป็น/ไม่เป็น + ระดับ)
# ============================================================
@torch.no_grad()
def predict_image(path, show=True):
    rgb = preprocess_image(path)
    x = eval_tf(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
    prob = torch.softmax(model(x).float(), 1)[0].cpu().numpy()
    grade = int(prob.argmax()); ref_p = float(prob[2:].sum())
    referable = ref_p >= THR
    out = {"grade": grade, "grade_name": CLASS_NAMES[grade],
           "referable": bool(referable), "referable_prob": ref_p,
           "probs": {CLASS_NAMES[i]: float(prob[i]) for i in range(5)}}
    if show:
        verdict = "🔴 พบความผิดปกติ — ควรพบจักษุแพทย์" if referable else "🟢 ปกติ — ไม่ต้องส่งต่อ"
        print(f"\n{os.path.basename(str(path))}")
        print(f"  ระดับ: {CLASS_NAMES[grade]} (grade {grade})  |  P(referable)={ref_p:.2f}")
        print(f"  ผล: {verdict}")
    return out

# ---- เดโม่บนรูปจาก test set ----
demo_df = df_idrid if df_idrid is not None else df_test
if demo_df is not None:
    print("\n========== DEMO PREDICTIONS ==========")
    for p in demo_df.sample(min(6, len(demo_df)), random_state=SEED)["path"]:
        predict_image(p)

print("\n✅ เสร็จสิ้น — ผลทั้งหมดอยู่ใน:", OUT)
print("   models/best.pt, metrics.json, confusion_*.png, roc_curve.png, training_curves.png, gradcam_eyes.png")
