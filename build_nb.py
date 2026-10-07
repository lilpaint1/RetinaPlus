import json, io

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})

def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})

md(r"""# RetinaPlus — External Validation + Screening Calibration (ฉบับเต็ม)

ensemble **v3+v4 EfficientNet-B4 (ordinal regression)** | **inference อย่างเดียว ไม่เทรน**

ทำ 3 อย่างในไฟล์เดียว:
1. inference ทุก set (**val / test / DDR / IDRiD**) แล้ว **เซฟ CSV ครบ** (รวม `val_predictions.csv` ที่เคยหาย)
2. **Isotonic calibration** (score → P(referable)) fit บน val → `calibrator.pkl`
3. **Abstain band** (clear-normal / abstain / clear-refer) → `thresholds.json` + รายงาน sens/spec/automation-rate

> ก่อนรัน: Settings → Accelerator → **GPU** และ Add Input:
> (ก) weights v3/v4 (ข) `ddrdataset` (ค) `idrid-dataset` (ง) `eyepacs-aptos-messidor` (มี `val/` และ `test/`)
""")

code(r"""
# ============================ CELL 1 — setup ============================
import sys, subprocess
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "timm"], check=False)

import os, glob, cv2, json, collections, numpy as np, pandas as pd
import torch, timm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (cohen_kappa_score, accuracy_score, confusion_matrix,
                             classification_report, roc_auc_score, roc_curve)
from sklearn.isotonic import IsotonicRegression
import joblib
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| device:", DEVICE)
""")

code(r"""
# ============================ CELL 2 — config ============================
IMG_SIZE   = 380
BACKBONE   = "efficientnet_b4"
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

# OptimizedRounder ตอนเทรน — ห้าม fit ใหม่บน external (ใช้บอก "ระดับ" เท่านั้น)
ROUNDER_COEFS = [0.515, 1.264, 2.376, 3.079]
REFERABLE_THR = ROUNDER_COEFS[1]          # grade >= 2 = referable DR

# เป้าหมาย screening (ใช้สร้าง abstain band บน val)
SENS_TARGET = 0.95   # โซน auto-clear: ต่ำกว่า P_LO พลาด referable <= 5%
SPEC_TARGET = 0.95   # โซน auto-refer: สูงกว่า P_HI false-positive <= 5%

# --- weights v3/v4 (auto-หา) ---
WEIGHT_PATHS = sorted(glob.glob("/kaggle/input/**/best*.pt", recursive=True))
if not WEIGHT_PATHS:
    WEIGHT_PATHS = sorted(glob.glob("/kaggle/input/**/*.pt", recursive=True))
print("WEIGHT_PATHS:", WEIGHT_PATHS)
assert WEIGHT_PATHS, "ไม่เจอ weight .pt — Add Input weights ก่อน"

# --- dataset paths (auto-หา) ---
def _find(*pats):
    for pat in pats:
        h = sorted(glob.glob(pat, recursive=True))
        if h: return h[0]
    return None

DDR_CSV   = _find("/kaggle/input/**/DR_grading.csv")
IDRID_CSV = next((p for p in glob.glob("/kaggle/input/**/*.csv", recursive=True)
                  if "idrid" in p.lower()), None)
UNIFIED   = _find("/kaggle/input/**/dr_unified_v2/dr_unified_v2",
                  "/kaggle/input/**/dr_unified_v2")
VAL_DIR   = os.path.join(UNIFIED, "val")  if UNIFIED else None
TEST_DIR  = os.path.join(UNIFIED, "test") if UNIFIED else None
print("DDR_CSV  :", DDR_CSV)
print("IDRID_CSV:", IDRID_CSV)
print("VAL_DIR  :", VAL_DIR)
print("TEST_DIR :", TEST_DIR)

BATCH, WORKERS, USE_FP16 = 64, 4, True
OUT = "/kaggle/working"
""")

code(r"""
# ============================ CELL 3 — load ensemble (v3 + v4) ============================
def build_model():
    m = timm.create_model(BACKBONE, pretrained=False, num_classes=1,
                          drop_rate=0.3, drop_path_rate=0.2)
    return m.to(DEVICE).eval()

MEMBERS = []
for p in WEIGHT_PATHS:
    m = build_model()
    sd = torch.load(p, map_location=DEVICE)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    m.load_state_dict(sd)
    MEMBERS.append(m)
    print("loaded", p)
print("ensemble size =", len(MEMBERS))
""")

code(r"""
# ============================ CELL 4 — preprocess (เหมือน app.py เป๊ะ) ============================
_MEAN = np.array([0.485, 0.456, 0.406]); _STD = np.array([0.229, 0.224, 0.225])

def _crop_fundus(img, tol=7):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); mask = g > tol
    if mask.sum() == 0: return img
    c = np.argwhere(mask); y0, x0 = c.min(0); y1, x1 = c.max(0) + 1
    return img[y0:y1, x0:x1]

def _clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB); l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(2.0, (8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def preprocess(bgr):
    img = _crop_fundus(bgr)
    if img.size == 0: img = bgr
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = _clahe(img)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def to_chw(rgb):
    return ((rgb / 255.0 - _MEAN) / _STD).transpose(2, 0, 1).astype(np.float32)
""")

code(r"""
# ============================ CELL 5 — dataset + builders + inference ============================
class DRDataset(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        path, lab = self.items[i]
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            bgr = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
        return torch.from_numpy(to_chw(preprocess(bgr))), lab

def items_from_folder(split_dir):
    items = []
    if not split_dir or not os.path.isdir(split_dir): return items
    for c in ["0", "1", "2", "3", "4"]:
        for p in glob.glob(os.path.join(split_dir, c, "*")):
            if p.lower().endswith((".jpg", ".jpeg", ".png")):
                items.append((p, int(c)))
    return items

def items_from_csv(csv_path, name_col="id_code", label_col="diagnosis", drop=frozenset({5})):
    if not csv_path or not os.path.exists(csv_path): return []
    df = pd.read_csv(csv_path); df.columns = [str(c).strip() for c in df.columns]
    root = os.path.dirname(csv_path)
    if name_col not in df.columns:
        name_col = next((c for c in df.columns if "image" in c.lower()
                         or c.lower() in ("id_code", "id")), df.columns[0])
    if label_col not in df.columns:
        label_col = next((c for c in df.columns if "grade" in c.lower()
                          or "retin" in c.lower() or "diagnos" in c.lower()), df.columns[1])
    idx = {}
    for r, _, fs in os.walk(root):
        for fn in fs:
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                full = os.path.join(r, fn)
                idx[fn] = full; idx[os.path.splitext(fn)[0]] = full
    items = []
    for _, row in df.iterrows():
        try: lab = int(float(row[label_col]))
        except Exception: continue
        if lab in drop or not (0 <= lab <= 4): continue
        key = str(row[name_col]).strip()
        p = idx.get(key) or idx.get(os.path.splitext(key)[0])
        if p: items.append((p, lab))
    return list(dict(items).items())   # ตัดซ้ำตาม path

@torch.no_grad()
def ensemble_batch(x):
    x = x.to(DEVICE, non_blocking=True)
    total = torch.zeros(x.size(0), device=DEVICE)
    with torch.amp.autocast("cuda", enabled=USE_FP16):     # API ใหม่ (ไม่ deprecated)
        for m in MEMBERS:
            o = m(x).float().squeeze(1)
            o = o + m(torch.flip(x, [3])).float().squeeze(1) + m(torch.flip(x, [2])).float().squeeze(1)
            total += o / 3
    return (total / len(MEMBERS)).cpu().numpy()

def run_infer(items, desc=""):
    if not items: return np.array([]), np.array([])
    loader = DataLoader(DRDataset(items), batch_size=BATCH, shuffle=False,
                        num_workers=WORKERS, pin_memory=True)
    scores, ys = [], []
    for bi, (x, lab) in enumerate(loader):
        scores.append(ensemble_batch(x)); ys.append(lab.numpy())
        if bi % 25 == 0: print(f"  [{desc}] batch {bi}/{len(loader)}")
    return np.concatenate(scores), np.concatenate(ys)
""")

code(r"""
# ============================ CELL 6 — inference ทุก set + เซฟ CSV ครบ ============================
SETS = {
    "val":   items_from_folder(VAL_DIR),
    "test":  items_from_folder(TEST_DIR),
    "ddr":   items_from_csv(DDR_CSV),
    "idrid": items_from_csv(IDRID_CSV),
}
for k, v in SETS.items():
    dist = dict(sorted(collections.Counter(l for _, l in v).items()))
    print(f"{k:6s}: {len(v):6d} images  {dist}")

PRED = {}
for name, items in SETS.items():
    if not items:
        print(f"[skip] {name} ว่าง (เช็ค path/Add Input)"); continue
    s, y = run_infer(items, name)
    pred = np.clip(np.digitize(s, ROUNDER_COEFS), 0, 4)
    df = pd.DataFrame({"path": [p for p, _ in items], "true": y,
                       "pred": pred, "score": np.round(s, 4)})
    csv = f"{OUT}/{name}_predictions.csv"
    df.to_csv(csv, index=False)
    PRED[name] = df
    print(f"[saved] {csv}  ({len(df)} rows)")

assert "val" in PRED, "ไม่มี val — calibration ทำไม่ได้ เช็ค VAL_DIR ใน CELL 2"
""")

code(r"""
# ============================ CELL 7 — metrics + confusion matrix ต่อ set ============================
def eval_set(name, df):
    y, pred, s = df["true"].values, df["pred"].values, df["score"].values
    qwk = cohen_kappa_score(y, pred, weights="quadratic")
    acc = accuracy_score(y, pred)
    ref = (y >= 2).astype(int)
    auc = roc_auc_score(ref, s) if len(set(ref)) > 1 else float("nan")
    print(f"\n===== {name} (n={len(df)}) =====")
    print(f"QWK {qwk:.4f} | Acc {acc:.4f} | Referable AUC {auc:.4f}")
    print(classification_report(y, pred, target_names=CLASS_NAMES, digits=3,
                                zero_division=0, labels=list(range(5))))
    cm = confusion_matrix(y, pred, labels=[0, 1, 2, 3, 4])
    plt.figure(figsize=(6, 5)); plt.imshow(cm, cmap="Blues")
    plt.title(f"{name}  QWK={qwk:.3f} Acc={acc:.3f} n={len(df)}")
    plt.xticks(range(5), CLASS_NAMES, rotation=40, ha="right"); plt.yticks(range(5), CLASS_NAMES)
    th = cm.max() / 2
    for i in range(5):
        for j in range(5):
            plt.text(j, i, cm[i, j], ha="center",
                     color="white" if cm[i, j] > th else "black", fontsize=10)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.colorbar(fraction=0.046)
    plt.tight_layout(); plt.savefig(f"{OUT}/confusion_{name}.png", dpi=140); plt.show()
    return {"dataset": name, "n": int(len(df)), "qwk": float(qwk),
            "accuracy": float(acc), "referable_auc": float(auc)}

summ = [eval_set(n, PRED[n]) for n in PRED]
pd.DataFrame(summ).to_csv(f"{OUT}/summary_metrics.csv", index=False)
print("\n[saved] summary_metrics.csv")
""")

code(r"""
# ============================ CELL 8 — Isotonic calibration (fit บน val) ============================
v = PRED["val"]
iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
iso.fit(v["score"].values.astype(float), (v["true"].values >= 2).astype(int))
joblib.dump(iso, f"{OUT}/calibrator.pkl")
print("[saved] calibrator.pkl")

def to_prob(scores):
    return iso.predict(np.asarray(scores, dtype=float))

# sanity: score เดิม -> ความน่าจะเป็น referable
for sc in [0.1, 0.5, 1.0, 1.264, 2.0, 3.0]:
    print(f"  score {sc:5.2f} -> P(referable) {float(to_prob([sc])[0]):.3f}")
""")

code(r"""
# ============================ CELL 9 — Abstain band + รายงาน screening ============================
# หา P_LO (sens>=SENS_TARGET) และ P_HI (spec>=SPEC_TARGET) บน prob ที่ calibrate แล้ว จาก val
vy = (PRED["val"]["true"].values >= 2).astype(int)
vp = to_prob(PRED["val"]["score"].values)
fpr, tpr, thr = roc_curve(vy, vp)

i_lo = int(np.argmax(tpr >= SENS_TARGET))           # threshold สูงสุดที่ยังได้ sens>=target
P_LO = float(thr[i_lo])
ok   = np.where(fpr <= (1 - SPEC_TARGET))[0]         # spec>=target  <=>  fpr<=1-target
P_HI = float(thr[ok[-1]]) if len(ok) else float(thr[0])
P_LO, P_HI = (min(P_LO, P_HI), max(P_LO, P_HI))
print(f"Abstain band (P referable):  clear-NORMAL < {P_LO:.3f}  |  ABSTAIN  |  >= {P_HI:.3f} clear-REFER\n")

def screen_report(name, df):
    ref = (df["true"].values >= 2).astype(int)
    p = to_prob(df["score"].values)
    refer  = p >= P_HI
    normal = p <  P_LO
    abstain = ~(refer | normal)
    auto = ~abstain
    yp = refer[auto].astype(int); yt = ref[auto]
    tp = int(((yp == 1) & (yt == 1)).sum()); fn = int(((yp == 0) & (yt == 1)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum()); fp = int(((yp == 1) & (yt == 0)).sum())
    sens = tp / max(1, tp + fn); spec = tn / max(1, tn + fp)
    print(f"{name:6s} n={len(df):6d} | auto={auto.mean()*100:5.1f}%  abstain={abstain.mean()*100:5.1f}% "
          f"| (auto-zone) Sens={sens:.3f}  Spec={spec:.3f}")
    return {"dataset": name, "n": int(len(df)),
            "automation_rate": float(auto.mean()), "abstain_rate": float(abstain.mean()),
            "auto_sens": float(sens), "auto_spec": float(spec)}

print("=== Screening + abstain (operating point จาก val — ถูกหลัก) ===")
rep = [screen_report(n, PRED[n]) for n in ["ddr", "idrid", "test"] if n in PRED]

json.dump({"P_LO": P_LO, "P_HI": P_HI,
           "sens_target": SENS_TARGET, "spec_target": SPEC_TARGET,
           "rounder_coefs": ROUNDER_COEFS, "referable_thr": REFERABLE_THR,
           "report": rep},
          open(f"{OUT}/thresholds.json", "w"), ensure_ascii=False, indent=2)
print("\n[saved] thresholds.json  (เอา calibrator.pkl + thresholds.json ไปใส่ app.py)")
""")

code(r"""
# ============================ CELL 10 — threshold sweep (อ้างอิง, 0 GPU) ============================
def sweep(name, df):
    s, ref = df["score"].values, (df["true"].values >= 2).astype(int)
    print(f"\n### {name} (n={len(df)}) ###")
    for t in [1.288, 1.0, 0.8, 0.6, 0.4]:
        p = (s >= t).astype(int)
        tp = ((p == 1) & (ref == 1)).sum(); fn = ((p == 0) & (ref == 1)).sum()
        tn = ((p == 0) & (ref == 0)).sum(); fp = ((p == 1) & (ref == 0)).sum()
        print(f"thr={t:.2f}  Sens={tp/max(1,tp+fn):.3f}  Spec={tn/max(1,tn+fp):.3f}")
for n in ["ddr", "idrid"]:
    if n in PRED: sweep(n, PRED[n])
""")

code(r"""
# ============================ CELL 11 — ROC curves ============================
plt.figure(figsize=(6.5, 6))
for n, lbl in [("test", "internal test (optimistic)"),
               ("idrid", "external IDRiD"),
               ("ddr", "external DDR (trustworthy)")]:
    if n not in PRED: continue
    d = PRED[n]; ref = (d["true"].values >= 2).astype(int)
    if len(set(ref)) < 2: continue
    fpr, tpr, _ = roc_curve(ref, d["score"].values)
    a = roc_auc_score(ref, d["score"].values)
    plt.plot(fpr, tpr, lw=2, label=f"{lbl}  AUC={a:.3f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC — Referable DR (grade >= 2)"); plt.legend(loc="lower right"); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{OUT}/roc_curve.png", dpi=140); plt.show()
""")

code(r"""
# ============================ CELL 12 — t-SNE (referable vs non, IDRiD) ============================
from sklearn.manifold import TSNE
SRC = SETS["idrid"]
m0 = MEMBERS[0]
fl = DataLoader(DRDataset(SRC), batch_size=BATCH, shuffle=False, num_workers=WORKERS, pin_memory=True)

feats, labs, scrs = [], [], []
for x, lab in fl:
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=USE_FP16):
        xd = x.to(DEVICE, non_blocking=True)
        f = m0.forward_features(xd)
        f = m0.global_pool(f) if hasattr(m0, "global_pool") else f.mean((2, 3))
        if f.ndim > 2: f = f.flatten(1)
    feats.append(f.float().cpu().numpy()); scrs.append(ensemble_batch(x)); labs.append(lab.numpy())
F = np.concatenate(feats); L = np.concatenate(labs); S = np.concatenate(scrs)

emb = TSNE(n_components=2, init="pca", perplexity=30, random_state=2024).fit_transform(F)
ybin = (L >= 2).astype(int)
wrong = (S >= REFERABLE_THR).astype(int) != ybin
plt.figure(figsize=(9, 8))
plt.scatter(emb[ybin == 0, 0], emb[ybin == 0, 1], s=16, color="blue", label="non-referable (No/Mild)")
plt.scatter(emb[ybin == 1, 0], emb[ybin == 1, 1], s=16, color="red", label="referable (grade >= 2)")
plt.scatter(emb[wrong, 0], emb[wrong, 1], s=130, facecolors="none",
            edgecolors="green", linewidths=1.6, label="misclassified")
plt.xticks([]); plt.yticks([]); plt.title("t-SNE — Referable vs Non-referable (external IDRiD)")
plt.legend(loc="best"); plt.tight_layout(); plt.savefig(f"{OUT}/tsne_referable_idrid.png", dpi=140); plt.show()
print("misclassified", int(wrong.sum()), "/", len(L))
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = r"C:\Users\Acer\Downloads\Retina Classifier\RetinaPlus_External_Validation_full.ipynb"
with io.open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("WROTE", out, "| cells:", len(cells))
