import json, io

cells = []
def md(s):  cells.append({"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)})
def code(s):cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],
                          "source":s.strip("\n").splitlines(keepends=True)})

md(r"""# RetinaPlus — NB2 Quality Gate (heuristic + EyeQ model)

ด่านตรวจคุณภาพรูปก่อนทำนาย DR ตามผัง:

```
รูป → [heuristic] เป็น fundus + ไม่พังชัด ๆ ? ──ไม่ผ่าน→ "ไม่ใช่รูปจอประสาทตา/ถ่ายใหม่"
       ↓ ผ่าน
     [EyeQ model] Good / Usable / Reject ──Reject→ "ภาพไม่ชัดพอ ถ่ายใหม่"
       ↓ Good/Usable
     ทำนาย DR ต่อ
```

**Output:** `quality_gate.pt` (EfficientNet-B0 3 คลาส) + `quality_gate_config.json` (cutoffs heuristic + reject threshold) → เอาไปใส่ `app.py`

---
### ⚙️ Settings (สำคัญ)
- Accelerator → **GPU**
- **Internet → ON** (ดาวน์โหลด weight pretrained ของ B0 ~20MB). ถ้าเปิด Internet ไม่ได้ ให้ Add Input โมเดล `timm/efficientnet_b0.ra_in1k` แทน — โค้ดมี fallback ให้

### 📥 Add Input ที่ต้องมี
1. **ชุด EyeQ** ที่มีทั้ง
   - label CSV: `Label_EyeQ_train.csv` / `Label_EyeQ_test.csv` (คอลัมน์ `image`, `quality` โดย 0=Good 1=Usable 2=Reject)
   - **รูป fundus** ที่ CSV อ้างถึง (ถ้าชุด EyeQ มีแต่ label ให้ add ชุดรูป EyePACS เพิ่ม เพราะ EyeQ ใช้ชื่อไฟล์จาก EyePACS เช่น `10_left.jpeg`)

> CELL 2 มี auto-detect ให้ — แค่ add แล้วดู `quality CSVs:` กับ `indexed images:` ว่าไม่ว่าง
""")

code(r"""
# ============================ CELL 1 — setup ============================
import sys, subprocess
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "timm"], check=False)

import os, glob, cv2, json, math, random, collections, numpy as np, pandas as pd
import torch, torch.nn as nn, timm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, roc_auc_score)
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(2024); np.random.seed(2024); torch.manual_seed(2024)
print("torch", torch.__version__, "| device:", DEVICE)
""")

code(r"""
# ============================ CELL 2 — config + auto-detect ============================
QIMG     = 384
QCLASSES = ["Good", "Usable", "Reject"]     # EyeQ: 0=Good 1=Usable 2=Reject
OUT      = "/kaggle/working"
EPOCHS_Q = 5

# --- หา label CSV ของ EyeQ (ชื่อไฟล์มี eyeq / quality) ---
def _is_quality_csv(p):
    n = os.path.basename(p).lower()
    return ("eyeq" in n) or ("quality" in n)
QCSV = sorted([p for p in glob.glob("/kaggle/input/**/*.csv", recursive=True) if _is_quality_csv(p)])
print("quality CSVs:", QCSV)

# --- index รูปทั้งหมด (basename -> path) เพื่อจับคู่กับ CSV ---
print("building image index ... (อาจใช้เวลาสักครู่ถ้ารูปเยอะ)")
IMG_INDEX = {}
for r, _, fs in os.walk("/kaggle/input"):
    for fn in fs:
        if fn.lower().endswith((".jpg", ".jpeg", ".png")):
            full = os.path.join(r, fn)
            IMG_INDEX.setdefault(fn.lower(), full)
            IMG_INDEX.setdefault(os.path.splitext(fn)[0].lower(), full)
print("indexed images:", len(IMG_INDEX))
assert QCSV, "ไม่เจอ label CSV ของ EyeQ — Add Input ชุด EyeQ ก่อน"
""")

code(r"""
# ============================ CELL 3 — preprocess + heuristic metrics ============================
_MEAN = np.array([0.485, 0.456, 0.406]); _STD = np.array([0.229, 0.224, 0.225])

def crop_fundus_bbox(bgr, tol=7):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY); mask = g > tol
    if mask.sum() == 0: return bgr, 0.0
    c = np.argwhere(mask); y0, x0 = c.min(0); y1, x1 = c.max(0) + 1
    area_ratio = float(mask.sum()) / (bgr.shape[0] * bgr.shape[1])   # สัดส่วนพื้นที่จอในเฟรม
    return bgr[y0:y1, x0:x1], area_ratio

def fundus_metrics(bgr):
    crop, area_ratio = crop_fundus_bbox(bgr)
    if crop.size == 0: crop = bgr
    g = cv2.cvtColor(cv2.resize(crop, (QIMG, QIMG)), cv2.COLOR_BGR2GRAY)
    blur   = float(cv2.Laplacian(g, cv2.CV_64F).var())   # สูง=คม / ต่ำ=เบลอ
    bright = float(g.mean())                              # 0-255
    return {"area_ratio": area_ratio, "blur": blur, "bright": bright}

def q_preprocess(bgr):
    crop, _ = crop_fundus_bbox(bgr)
    if crop.size == 0: crop = bgr
    img = cv2.resize(crop, (QIMG, QIMG), interpolation=cv2.INTER_AREA)
    # หมายเหตุ: ไม่ใช้ CLAHE ที่นี่ — ตั้งใจเก็บ brightness/contrast ไว้เป็น "สัญญาณคุณภาพ"
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def q_chw(rgb):
    return ((rgb / 255.0 - _MEAN) / _STD).transpose(2, 0, 1).astype(np.float32)
""")

code(r"""
# ============================ CELL 4 — โหลด EyeQ + จับคู่รูป (robust) ============================
import re
IMG_PAT = re.compile(r'(\.jpe?g$|\.png$|_(left|right)\b)', re.I)

def pick_image_col(t):
    # 1) เลือกคอลัมน์จาก "ค่า" ที่หน้าตาเป็นชื่อรูป (ลงท้าย .jpg/.png หรือมี _left/_right)
    best, best_score = None, 0.0
    for c in t.columns:
        s = t[c].dropna().astype(str).head(100)
        if len(s) == 0: continue
        score = float(s.str.contains(IMG_PAT).mean())
        if score > best_score: best, best_score = c, score
    if best is not None and best_score > 0.5: return best
    # 2) fallback: เดาจากชื่อคอลัมน์
    for c in t.columns:
        if "image" in c.lower() or "name" in c.lower() or c.lower() in ("id", "id_code", "img"):
            return c
    return t.columns[0]

def load_eyeq(csvs):
    rows = []
    for cp in csvs:
        t = pd.read_csv(cp); t.columns = [str(c).strip() for c in t.columns]
        col_img = pick_image_col(t)
        col_q = next((c for c in t.columns if "quality" in c.lower()
                      or c.lower() in ("label", "q")), None)
        if col_q is None:    # เดาจากค่า: คอลัมน์ที่ค่าทั้งหมดอยู่ใน {0,1,2}
            for c in t.columns:
                if c == col_img: continue
                vals = pd.to_numeric(t[c], errors="coerce").dropna().unique()
                if len(vals) and set(vals).issubset({0, 1, 2}):
                    col_q = c; break
        print(f"  {os.path.basename(cp)} | columns={list(t.columns)} | img_col={col_img} q_col={col_q}")
        print("    sample:", t[col_img].head(3).tolist())
        if col_q is None:
            print("  ข้าม (หา quality column ไม่เจอ):", cp); continue
        split = "test" if "test" in os.path.basename(cp).lower() else "train"
        hit = 0
        for _, r in t.iterrows():
            try: q = int(r[col_q])
            except Exception: continue
            if q not in (0, 1, 2): continue
            key = os.path.basename(str(r[col_img]).strip()).lower()   # ตัด path นำหน้าออก
            p = IMG_INDEX.get(key) or IMG_INDEX.get(os.path.splitext(key)[0])
            if p: rows.append((p, q, split)); hit += 1
        print(f"    -> [{split}] จับคู่ได้ {hit}")
    return rows

ROWS = load_eyeq(QCSV)
assert ROWS, "จับคู่ EyeQ ไม่ได้เลย — เช็คว่า add 'รูป' ที่ CSV อ้างถึงด้วย (ไม่ใช่แค่ label)"

tr = [(p, q) for p, q, s in ROWS if s == "train"]
te = [(p, q) for p, q, s in ROWS if s == "test"]
if not te:                                   # ไม่มี split test แยก -> สุ่มแบ่ง 15%
    random.shuffle(tr); k = int(len(tr) * 0.15); te, tr = tr[:k], tr[k:]
print("train", len(tr), dict(sorted(collections.Counter(q for _, q in tr).items())))
print("test ", len(te), dict(sorted(collections.Counter(q for _, q in te).items())))
""")

code(r"""
# ============================ CELL 5 — heuristic analysis + เสนอ cutoff ============================
samp = random.sample(tr, min(1500, len(tr)))
M = {0: [], 1: [], 2: []}
for p, q in samp:
    bgr = cv2.imread(p, cv2.IMREAD_COLOR)
    if bgr is None: continue
    m = fundus_metrics(bgr); M[q].append((m["area_ratio"], m["blur"], m["bright"]))

def arr(q, i): return np.array([t[i] for t in M[q]]) if M[q] else np.array([])
names = ["area_ratio", "blur", "bright"]
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for i in range(3):
    for q, col in [(0, "green"), (2, "red")]:
        a = arr(q, i)
        if len(a): ax[i].hist(a, bins=40, alpha=0.5, label=QCLASSES[q], color=col, density=True)
    ax[i].set_title(names[i]); ax[i].legend()
plt.tight_layout(); plt.savefig(f"{OUT}/heuristic_hist.png", dpi=130); plt.show()

# cutoff = เปอร์เซ็นไทล์ต่ำ ๆ ของรูป Good -> reject Good แทบไม่โดน แต่จับขยะชัด ๆ ได้
ga, gb, gr = arr(0, 0), arr(0, 1), arr(0, 2)
AREA_MIN = float(np.percentile(ga, 2))  if len(ga) else 0.05
BLUR_MIN = float(np.percentile(gb, 2))  if len(gb) else 50.0
BRI_LO   = float(np.percentile(gr, 1))  if len(gr) else 25.0
BRI_HI   = float(np.percentile(gr, 99)) if len(gr) else 230.0
print(f"เสนอ cutoff -> AREA_MIN={AREA_MIN:.3f}  BLUR_MIN={BLUR_MIN:.1f}  BRIGHT=[{BRI_LO:.0f},{BRI_HI:.0f}]")

# เช็คเร็ว ๆ: heuristic จับ Reject ได้กี่ % (โดยไม่เผลอตัด Good)
def _flag(t): return (t[0] < AREA_MIN) or (t[1] < BLUR_MIN) or not (BRI_LO <= t[2] <= BRI_HI)
for q in (0, 2):
    if M[q]:
        rate = np.mean([_flag(t) for t in M[q]])
        print(f"  heuristic flag {QCLASSES[q]:6s}: {rate*100:.1f}%")
""")

code(r"""
# ============================ CELL 6 — dataset + loaders ============================
train_aug = A.Compose([A.HorizontalFlip(0.5), A.VerticalFlip(0.5), A.RandomRotate90(0.5),
                       A.RandomBrightnessContrast(0.1, 0.1, p=0.3), A.Normalize(_MEAN, _STD)])
eval_aug  = A.Compose([A.Normalize(_MEAN, _STD)])   # ไม่ aug blur/contrast แรง — กันลบสัญญาณคุณภาพ

class QDS(Dataset):
    def __init__(s, items, training): s.items = items; s.training = training
    def __len__(s): return len(s.items)
    def __getitem__(s, i):
        p, q = s.items[i]
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None: bgr = np.zeros((QIMG, QIMG, 3), np.uint8)
        rgb = q_preprocess(bgr)
        a = (train_aug if s.training else eval_aug)(image=rgb)["image"]
        return torch.from_numpy(a.transpose(2, 0, 1).astype(np.float32)), q

cnt = collections.Counter(q for _, q in tr)
w = [1.0 / math.sqrt(cnt[q]) for _, q in tr]            # weighted sampler แก้ class imbalance
sampler = WeightedRandomSampler(torch.DoubleTensor(w), len(w), replacement=True)
trl = DataLoader(QDS(tr, True),  batch_size=32, sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)
tel = DataLoader(QDS(te, False), batch_size=64, shuffle=False,   num_workers=4, pin_memory=True)
print("batches/epoch:", len(trl))
""")

code(r"""
# ============================ CELL 7 — model (B0) + train ============================
def build_q():
    try:
        m = timm.create_model("efficientnet_b0", pretrained=True, num_classes=3, drop_rate=0.2)
    except Exception as e:
        print("⚠️ โหลด pretrained ไม่ได้ (เปิด Internet หรือ add weight B0) ->", type(e).__name__)
        m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=3, drop_rate=0.2)
        for pt in glob.glob("/kaggle/input/**/*efficientnet_b0*", recursive=True):
            if pt.endswith((".pt", ".pth", ".safetensors")):
                try:
                    sd = torch.load(pt, map_location="cpu")
                    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
                    m.load_state_dict(sd, strict=False); print("  loaded backbone:", pt); break
                except Exception: pass
    return m.to(DEVICE)

qmodel = build_q()
opt    = torch.optim.AdamW(qmodel.parameters(), lr=3e-4, weight_decay=1e-4)
sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_Q)
crit   = nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler("cuda")

for ep in range(1, EPOCHS_Q + 1):
    qmodel.train(); run = 0.0
    for x, y in trl:
        x = x.to(DEVICE); y = y.to(DEVICE, dtype=torch.long); opt.zero_grad()
        with torch.amp.autocast("cuda"):
            loss = crit(qmodel(x), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); run += loss.item()
    sched.step(); print(f"ep{ep}/{EPOCHS_Q} loss={run/len(trl):.4f}")

torch.save(qmodel.state_dict(), f"{OUT}/quality_gate.pt")
print("[saved] quality_gate.pt")
""")

code(r"""
# ============================ CELL 8 — evaluate ============================
qmodel.eval(); P, Y = [], []
with torch.no_grad():
    for x, y in tel:
        with torch.amp.autocast("cuda"):
            o = qmodel(x.to(DEVICE)).softmax(1).float().cpu().numpy()
        P.append(o); Y.append(y.numpy())
P = np.concatenate(P); Y = np.concatenate(Y); pred = P.argmax(1)

print("Accuracy:", round(accuracy_score(Y, pred), 4))
print(classification_report(Y, pred, target_names=QCLASSES, digits=3, zero_division=0, labels=[0, 1, 2]))
rej_true = (Y == 2).astype(int)
if len(set(rej_true)) > 1:
    print("Reject-vs-rest AUC:", round(roc_auc_score(rej_true, P[:, 2]), 4))

cm = confusion_matrix(Y, pred, labels=[0, 1, 2])
plt.figure(figsize=(5, 4)); plt.imshow(cm, cmap="Blues")
plt.xticks(range(3), QCLASSES); plt.yticks(range(3), QCLASSES)
for i in range(3):
    for j in range(3):
        plt.text(j, i, cm[i, j], ha="center", color="white" if cm[i, j] > cm.max()/2 else "black")
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("EyeQ quality"); plt.colorbar(fraction=0.046)
plt.tight_layout(); plt.savefig(f"{OUT}/confusion_quality.png", dpi=130); plt.show()
""")

code(r"""
# ============================ CELL 9 — check_quality() + เซฟ config ============================
REJECT_THR = 0.50    # P(Reject) เกินนี้ = ปฏิเสธ (ปรับได้)

@torch.no_grad()
def check_quality(bgr):
    m = fundus_metrics(bgr)
    # ---- ด่าน 1: heuristic (กัน non-fundus / พังชัด ๆ) ----
    if m["area_ratio"] < AREA_MIN:
        return {"ok": False, "stage": "heuristic", "reason": "ไม่ใช่รูปจอประสาทตา/ถ่ายไม่ติดจอ", "metrics": m}
    if m["blur"] < BLUR_MIN:
        return {"ok": False, "stage": "heuristic", "reason": "ภาพเบลอเกินไป ถ่ายใหม่", "metrics": m}
    if not (BRI_LO <= m["bright"] <= BRI_HI):
        return {"ok": False, "stage": "heuristic", "reason": "ภาพมืด/สว่างเกินไป ถ่ายใหม่", "metrics": m}
    # ---- ด่าน 2: EyeQ model ----
    rgb = q_preprocess(bgr)
    x = torch.from_numpy(q_chw(rgb)).unsqueeze(0).to(DEVICE)
    with torch.amp.autocast("cuda"):
        prob = qmodel(x).softmax(1)[0].float().cpu().numpy()
    qlab = int(prob.argmax())
    if prob[2] >= REJECT_THR or qlab == 2:
        return {"ok": False, "stage": "model", "reason": "ภาพไม่ชัดพอสำหรับวินิจฉัย ถ่ายใหม่",
                "quality": "Reject", "probs": prob.round(3).tolist(), "metrics": m}
    return {"ok": True, "stage": "model", "quality": QCLASSES[qlab],
            "probs": prob.round(3).tolist(), "metrics": m}

cfg = {"AREA_MIN": AREA_MIN, "BLUR_MIN": BLUR_MIN, "BRI_LO": BRI_LO, "BRI_HI": BRI_HI,
       "REJECT_THR": REJECT_THR, "QIMG": QIMG, "classes": QCLASSES}
json.dump(cfg, open(f"{OUT}/quality_gate_config.json", "w"), ensure_ascii=False, indent=2)
print("[saved] quality_gate_config.json\n", cfg)
""")

code(r"""
# ============================ CELL 10 — demo ============================
print("ลอง check_quality บนรูป test:")
for p, q in random.sample(te, min(8, len(te))):
    bgr = cv2.imread(p, cv2.IMREAD_COLOR)
    if bgr is None: continue
    r = check_quality(bgr)
    verdict = "PASS ✅" if r["ok"] else "REJECT ❌"
    detail = r.get("reason", r.get("quality", ""))
    print(f"  {os.path.basename(p):24s} | true={QCLASSES[q]:6s} -> {verdict}  ({r['stage']}: {detail})")
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}

out = r"C:\Users\Acer\Downloads\Retina Classifier\RetinaPlus_NB2_QualityGate.ipynb"
with io.open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("WROTE", out, "| cells:", len(cells))
