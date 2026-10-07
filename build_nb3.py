import json, io

cells = []
def md(s):  cells.append({"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)})
def code(s):cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],
                          "source":s.strip("\n").splitlines(keepends=True)})

md(r"""# RetinaPlus — NB3 Lesion → Rationale (แทน GradCAM)

เอา **UNet++ (`unet_lesion_v2.pt`)** มาทำมากกว่าแค่ระบายสี:
1. segment รอยโรค MA/HE/EX/SE (sliding window)
2. **แปลง mask → ตัวเลข** (จำนวน MA, จำนวน/พื้นที่ HE, EX ใกล้กลางจอไหม, SE, การกระจาย quadrant)
3. **สร้างประโยคเหตุผล** ผูกกับ grade จาก classifier → "ทำไมถึงเป็นระดับนี้"
4. วาด overlay + ได้ฟังก์ชัน **`explain(bgr)`** ไปเสียบ `app.py` แทน GradCAM

---
### ⚙️ Settings: GPU + Internet ON (smp ดาวน์โหลด encoder) — หรือใช้ weight ที่โหลดจาก unet เอง
### 📥 Add Input
1. **`unet_lesion_v2.pt`** (UNet++ encoder b4, 4 คลาส) — notebook output หรือ dataset
2. *(ไม่บังคับ)* **`best.pt` ทั้ง v3 + v4** (notebook output 2 ตัว) — ใช้เป็น ensemble + TTA เหมือน app.py เพื่อให้ grade ตรงกับที่ deploy
3. รูป fundus สำหรับ demo (IDRiD/DDR/อะไรก็ได้ที่มีรอยโรค)
""")

code(r"""
# ============================ CELL 1 — setup ============================
import sys, subprocess
subprocess.run([sys.executable, "-m", "pip", "-q", "install",
                "segmentation-models-pytorch", "timm"], check=False)

import os, glob, cv2, json, random, numpy as np
import torch, timm
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| smp", smp.__version__, "| device:", DEVICE)
""")

code(r"""
# ============================ CELL 2 — config + auto-detect ============================
LESIONS  = ["MA", "HE", "EX", "SE"]
PALETTE  = {"MA": (255, 0, 0), "HE": (0, 0, 255), "EX": (0, 255, 0), "SE": (255, 255, 0)}
LES_NAME = {"MA": "microaneurysm", "HE": "hemorrhage", "EX": "hard exudate", "SE": "soft exudate/cotton-wool"}
NCLASS   = 4
ENCODER  = "efficientnet-b4"

PATCH    = 768
WORK     = 1536          # < 2048 ตอนเทรน เพื่อเร็วขึ้น (Dice ตกเล็กน้อย) — ปรับได้
THR      = 0.5
MIN_AREA = 4             # ก้อนเล็กกว่านี้ไม่นับ (กัน noise)
_MEAN = np.array([0.485, 0.456, 0.406]); _STD = np.array([0.229, 0.224, 0.225])

CLASS_NAMES   = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
ROUNDER_COEFS = [0.515, 1.264, 2.376, 3.079]
OUT = "/kaggle/working"

def _find(*pats):
    for pat in pats:
        h = sorted(glob.glob(pat, recursive=True))
        if h: return h[0]
    return None

UNET_PATH = _find("/kaggle/input/**/unet*lesion*.pt", "/kaggle/input/**/*unet*.pt", "/kaggle/input/**/*lesion*.pt")
DR_PATHS  = sorted(glob.glob("/kaggle/input/**/best*.pt", recursive=True))   # ensemble v3+v4 (เหมือน app.py)
print("UNET_PATH:", UNET_PATH)
print("DR_PATHS :", DR_PATHS, "(ensemble — ไม่บังคับ)")
assert UNET_PATH, "ไม่เจอ unet_lesion_v2.pt — Add Input ก่อน"

# รูป demo: เลือก fundus สัก 4 รูป (เลี่ยงไฟล์ที่ดูเป็น mask)
all_imgs = [p for p in glob.glob("/kaggle/input/**/*.jp*g", recursive=True)
            if "mask" not in p.lower() and "label" not in p.lower()]
pref = [p for p in all_imgs if ("idrid" in p.lower() or "grading" in p.lower())]
pool = pref if pref else all_imgs
random.seed(7); SAMPLES = random.sample(pool, min(4, len(pool))) if pool else []
print("demo images:", len(SAMPLES))
""")

code(r"""
# ============================ CELL 3 — preprocess + load models ============================
def crop_clahe(bgr, work):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY); fg = g > 7
    if fg.sum() > 0:
        c = np.argwhere(fg); y0, x0 = c.min(0); y1, x1 = c.max(0) + 1; bgr = bgr[y0:y1, x0:x1]
    h, w = bgr.shape[:2]; s = work / max(h, w)
    bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB); l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(2.0, (8, 8)).apply(l)
    bgr = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def dr_preprocess(bgr, size=380):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY); m = g > 7
    if m.sum() > 0:
        c = np.argwhere(m); y0, x0 = c.min(0); y1, x1 = c.max(0) + 1; bgr = bgr[y0:y1, x0:x1]
    bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB); l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(2.0, (8, 8)).apply(l)
    bgr = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# --- UNet++ ---
unet = smp.UnetPlusPlus(ENCODER, encoder_weights=None, in_channels=3, classes=NCLASS).to(DEVICE)
sd = torch.load(UNET_PATH, map_location=DEVICE)
if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
unet.load_state_dict(sd); unet.eval()
print("loaded UNet:", UNET_PATH)

# --- classifier DR ensemble v3+v4 (optional, เหมือน app.py) ---
DR_MODELS = []
for dp in DR_PATHS:
    try:
        m = timm.create_model("efficientnet_b4", pretrained=False, num_classes=1,
                              drop_rate=0.3, drop_path_rate=0.2).to(DEVICE)
        sd2 = torch.load(dp, map_location=DEVICE)
        if isinstance(sd2, dict) and "state_dict" in sd2: sd2 = sd2["state_dict"]
        m.load_state_dict(sd2); m.eval(); DR_MODELS.append(m)
        print("loaded DR member:", dp)
    except Exception as e:
        print("ข้าม", dp, ":", type(e).__name__)
print("DR ensemble size:", len(DR_MODELS))
""")

code(r"""
# ============================ CELL 4 — predict_full (sliding window) ============================
@torch.no_grad()
def predict_full(rgb):
    H, W = rgb.shape[:2]; st = PATCH // 2
    prob = np.zeros((NCLASS, H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    ys = list(range(0, max(1, H - PATCH + 1), st)); xs = list(range(0, max(1, W - PATCH + 1), st))
    if H > PATCH and ys[-1] != H - PATCH: ys.append(H - PATCH)
    if W > PATCH and xs[-1] != W - PATCH: xs.append(W - PATCH)
    for y0 in ys:
        for x0 in xs:
            p = rgb[y0:y0 + PATCH, x0:x0 + PATCH]
            xt = torch.from_numpy(((p / 255.0 - _MEAN) / _STD).transpose(2, 0, 1).astype(np.float32)).unsqueeze(0).to(DEVICE)
            with torch.amp.autocast("cuda"):
                pr = torch.sigmoid(unet(xt))[0].float().cpu().numpy()
            prob[:, y0:y0 + PATCH, x0:x0 + PATCH] += pr
            cnt[y0:y0 + PATCH, x0:x0 + PATCH] += 1
    return prob / np.maximum(cnt, 1e-6)

@torch.no_grad()
def dr_grade(bgr):
    if not DR_MODELS: return None, None
    rgb = dr_preprocess(bgr)
    x = torch.from_numpy(((rgb / 255.0 - _MEAN) / _STD).transpose(2, 0, 1).astype(np.float32)).unsqueeze(0).to(DEVICE)
    with torch.amp.autocast("cuda"):
        outs = []
        for m in DR_MODELS:                       # ensemble + TTA hflip/vflip (เหมือน app.py)
            o = m(x).float().squeeze(1)
            o = o + m(torch.flip(x, [3])).float().squeeze(1) + m(torch.flip(x, [2])).float().squeeze(1)
            outs.append(o / 3)
        s = float((sum(outs) / len(outs)).item())
    g = int(np.clip(np.digitize(s, ROUNDER_COEFS), 0, 4))
    return CLASS_NAMES[g], s
""")

code(r"""
# ============================ CELL 5 — mask -> lesion features ============================
def lesion_features(prob, thr=THR, min_area=MIN_AREA):
    C, H, W = prob.shape
    feats, masks = {}, {}
    for ci, les in enumerate(LESIONS):
        m = (prob[ci] > thr).astype(np.uint8); masks[les] = m
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        comps = [int(stats[k, cv2.CC_STAT_AREA]) for k in range(1, n)
                 if stats[k, cv2.CC_STAT_AREA] >= min_area]
        area = int(sum(comps))
        feats[les] = {"count": len(comps), "area_px": area, "area_pct": 100.0 * area / (H * W)}
    return feats, masks

def quadrant_spread(mask, min_area=MIN_AREA):
    H, W = mask.shape; my, mx = H // 2, W // 2
    quads = {"บนซ้าย": mask[:my, :mx], "บนขวา": mask[:my, mx:],
             "ล่างซ้าย": mask[my:, :mx], "ล่างขวา": mask[my:, mx:]}
    return [k for k, v in quads.items() if v.sum() >= min_area]

def in_central(mask, frac=0.25):
    H, W = mask.shape; cy, cx = H // 2, W // 2; r = int(frac * min(H, W))
    yy, xx = np.ogrid[:H, :W]; circ = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r
    return int((mask * circ).sum())
""")

code(r"""
# ============================ CELL 5.5 — optic disc + fovea + ETDRS (classical CV) ============================
# heuristic ไม่ต้องเทรน — disc = บริเวณสว่างสุด, fovea = บริเวณมืดสุดห่าง disc ~2.5 DD
def detect_disc(rgb):
    H, W = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    fg = gray > 10
    blur = cv2.GaussianBlur(gray, (0, 0), max(3.0, W * 0.015))
    blur_m = np.where(fg, blur, -1.0)
    _, _, _, (cx, cy) = cv2.minMaxLoc(blur_m)
    r = int(0.05 * W)                                   # default
    try:
        t = np.percentile(blur[fg], 99.0)
        bright = ((blur >= t) & fg).astype(np.uint8)
        n, lbl, stats, cent = cv2.connectedComponentsWithStats(bright, 8)
        if n > 1:
            lab = lbl[cy, cx]
            if lab == 0:
                lab = min(range(1, n), key=lambda i: (cent[i][0]-cx)**2 + (cent[i][1]-cy)**2)
            area = int(stats[lab, cv2.CC_STAT_AREA])
            r = int(np.clip(np.sqrt(area / np.pi), 0.03 * W, 0.12 * W))
    except Exception:
        pass
    return int(cx), int(cy), int(r)

def detect_fovea(rgb, disc):
    H, W = rgb.shape[:2]; cx, cy, r = disc; DD = 2 * r
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    fg = (gray > 10).astype(np.uint8)
    ys0, xs0 = np.where(fg > 0)
    gx = xs0.mean() if len(xs0) else W / 2                       # centroid ของ fundus
    sidex = 1.0 if (gx - cx) >= 0 else -1.0                      # ฝั่งที่ disc ชี้ไปกลางจอ = ฝั่ง macula
    er = max(3, int(0.6 * DD))                                   # หด fundus กัน vignette ขอบ
    fg_e = cv2.erode(fg, np.ones((er, er), np.uint8), 1).astype(bool)
    blur = cv2.GaussianBlur(gray, (0, 0), max(3.0, DD * 0.35))
    yy, xx = np.mgrid[0:H, 0:W]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    horiz = np.abs(yy - cy) < 1.3 * DD
    side = (np.sign(xx - cx) == sidex)
    band = fg_e & (dist > 1.5 * DD) & (dist < 3.5 * DD) & horiz & side
    if band.sum() < 50: band = fg_e & (dist > 1.2 * DD) & (dist < 4.0 * DD) & side
    if band.sum() < 50: band = fg_e & (dist > 1.0 * DD)
    if band.sum() < 50: band = fg_e
    cand = np.where(band, blur, 1e9)
    fy, fx = np.unravel_index(int(np.argmin(cand)), cand.shape)
    return int(fx), int(fy)

def fovea_metrics(masks, fovea, DD):
    fx, fy = fovea; out = {}
    for les in LESIONS:
        ys, xs = np.where(masks[les] > 0)
        if len(xs) == 0:
            out[les] = {"min_dd": None, "px_within_1dd": 0}; continue
        d = np.sqrt((xs - fx) ** 2 + (ys - fy) ** 2) / max(DD, 1)
        out[les] = {"min_dd": float(d.min()), "px_within_1dd": int((d <= 1.0).sum())}
    return out

def draw_landmarks(ov, disc, fovea):
    cx, cy, r = disc; fx, fy = fovea; DD = 2 * r; W = ov.shape[1]
    cv2.circle(ov, (cx, cy), r, (0, 255, 255), 2)                       # disc = cyan
    s = max(6, int(0.03 * W))
    cv2.drawMarker(ov, (fx, fy), (255, 0, 255), cv2.MARKER_CROSS, s, 2)  # fovea = magenta
    for k in (1, 2):                                                    # ETDRS rings 1DD, 2DD
        cv2.circle(ov, (fx, fy), int(k * DD), (255, 255, 255), 1)
    return ov
""")

code(r"""
# ============================ CELL 6 — rationale + overlay (fovea-aware) ============================
def build_rationale(feats, masks, fov, grade_name=None):
    bits = []
    if feats["MA"]["count"] > 0:
        bits.append(f"microaneurysm ~{feats['MA']['count']} จุด")
    he = feats["HE"]
    if he["area_px"] > 0:
        q = quadrant_spread(masks["HE"])
        bits.append(f"hemorrhage {he['count']} ก้อน กระจาย {len(q)} quadrant ({', '.join(q)}) ~{he['area_pct']:.2f}% ของจอ")
    ex = feats["EX"]
    if ex["area_px"] > 0:
        d = fov["EX"]["min_dd"]
        if d is not None and d <= 1.0:
            loc = f"ใกล้ fovea ~{d:.1f} DD — อยู่ภายใน 1 DD เสี่ยง CSME สูง"
        elif d is not None:
            loc = f"ห่าง fovea ~{d:.1f} DD"
        else:
            loc = ""
        bits.append(f"hard exudate ~{ex['area_pct']:.2f}% {loc}".strip())
    if feats["SE"]["area_px"] > 0:
        bits.append("soft exudate / cotton-wool spot")
    if not bits:
        return "segmentation ไม่พบรอยโรคชัดเจน (ไม่ได้แปลว่าปกติ — รอยเล็กมากอาจตรวจไม่พบ)"
    head = f"เหตุผลประกอบ (โมเดลจัดเป็น {grade_name}): " if grade_name else "รอยโรคที่ตรวจพบ: "
    return head + "; ".join(bits)

def render_overlay(rgb, masks, alpha=0.5):
    ov = rgb.copy()
    for les in LESIONS:
        color = np.array(PALETTE[les], dtype=np.float32)
        m = masks[les].astype(bool)
        ov[m] = (alpha * color + (1 - alpha) * ov[m]).astype(np.uint8)
    return ov

def explain(bgr):
    rgb = crop_clahe(bgr, WORK)
    prob = predict_full(rgb)
    feats, masks = lesion_features(prob)
    disc = detect_disc(rgb); fovea = detect_fovea(rgb, disc); DD = 2 * disc[2]
    fov = fovea_metrics(masks, fovea, DD)
    grade, score = dr_grade(bgr)
    ov = draw_landmarks(render_overlay(rgb, masks), disc, fovea)
    return {"grade": grade, "score": score,
            "rationale": build_rationale(feats, masks, fov, grade),
            "features": feats, "fovea_metrics": fov, "disc": disc, "fovea": fovea,
            "overlay": ov, "preprocessed": rgb}
""")

code(r"""
# ============================ CELL 7 — demo: overlay + rationale ============================
assert SAMPLES, "ไม่มีรูป demo — Add Input รูป fundus"
n = len(SAMPLES)
fig, axs = plt.subplots(n, 2, figsize=(11, 5.2 * n))
if n == 1: axs = axs[None, :]
for row, path in enumerate(SAMPLES):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None: continue
    r = explain(bgr)
    axs[row, 0].imshow(r["preprocessed"]); axs[row, 0].set_title("input (crop+CLAHE)")
    axs[row, 1].imshow(r["overlay"]);      axs[row, 1].set_title("MA-red HE-blue EX-green SE-yellow")
    for c in (0, 1): axs[row, c].set_xticks([]); axs[row, c].set_yticks([])
    title = r["grade"] if r["grade"] else "(ไม่มี classifier)"
    axs[row, 0].set_ylabel(f"{os.path.basename(path)}\ngrade: {title}", fontsize=9)
    print(f"\n[{os.path.basename(path)}]  grade={title}  score={r['score']}")
    print("  ", r["rationale"])
    print(f"   disc={r['disc'][:2]} r={r['disc'][2]} | fovea={r['fovea']} | EX min_dd={r['fovea_metrics']['EX']['min_dd']}")
    print("  features:", {k: {kk: round(vv, 2) if isinstance(vv, float) else vv
                              for kk, vv in v.items()} for k, v in r["features"].items()})
fig.tight_layout(); fig.savefig(f"{OUT}/lesion_rationale_demo.png", dpi=130); plt.show()
print("\n[saved] lesion_rationale_demo.png")
""")

code(r"""
# ============================ CELL 8 — Dice ต่อรอยโรค (IDRiD Segmentation) ============================
# IDRiD เก็บ mask แยกไฟล์ต่อรอยโรค: IDRiD_01_MA.tif / _HE / _EX / _SE
import re
def crop_bbox(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY); fg = g > 7
    if fg.sum() > 0:
        c = np.argwhere(fg); y0, x0 = c.min(0); y1, x1 = c.max(0) + 1
        return int(y0), int(x0), int(y1), int(x1)
    return 0, 0, bgr.shape[0], bgr.shape[1]

# index mask แยกตามรอยโรค (จาก suffix ในชื่อไฟล์)
mask_re = re.compile(r'(idrid_\d+).*?_(ma|he|ex|se)\b', re.I)
masks_by_les = {l: {} for l in LESIONS}
for mp in glob.glob("/kaggle/input/**/*.*", recursive=True):
    if os.path.splitext(mp)[1].lower() not in (".tif", ".tiff", ".png", ".bmp"): continue
    m = mask_re.search(os.path.basename(mp))
    if m: masks_by_les[m.group(2).upper()][m.group(1).lower()] = mp

# index รูปต้นฉบับ IDRiD_XX.jpg
img_by_id = {}
for ip in glob.glob("/kaggle/input/**/*.jp*g", recursive=True):
    name = os.path.splitext(os.path.basename(ip))[0]
    if re.fullmatch(r'idrid_\d+', name, re.I): img_by_id[name.lower()] = ip

ids = sorted(set().union(*[set(d) for d in masks_by_les.values()]) & set(img_by_id)) if any(masks_by_les.values()) else []
test_ids = [i for i in ids if "test" in img_by_id[i].lower()]
use_ids = test_ids if test_ids else ids
print(f"IDRiD seg: {len(ids)} ภาพมี mask | ใช้ {'TEST ' if test_ids else 'ALL '}{len(use_ids)} ภาพ")
if ids and not test_ids:
    print("  ⚠️ ไม่เจอ test set แยก -> ใช้ทั้งหมด (อาจ optimistic ถ้า UNet เทรนบน IDRiD train)")

if not use_ids:
    print("ข้าม Dice — ไม่เจอ IDRiD seg masks. add 'IDRiD Segmentation Dataset' แล้วเช็คโครงไฟล์ IDRiD_XX_MA.tif")
else:
    inter = np.zeros(NCLASS); uni = np.zeros(NCLASS); cov = np.zeros(NCLASS, int)
    for k, iid in enumerate(use_ids[:40]):
        bgr = cv2.imread(img_by_id[iid]); rgb = crop_clahe(bgr, WORK)
        H, W = rgb.shape[:2]; y0, x0, y1, x1 = crop_bbox(bgr)
        pred = (predict_full(rgb) > THR).astype(np.float32)
        gt = np.zeros((NCLASS, H, W), np.float32)
        for ci, les in enumerate(LESIONS):
            mp = masks_by_les[les].get(iid)
            if mp is None: continue
            mm = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if mm is None: continue
            mm = mm[y0:y1, x0:x1]; mm = cv2.resize(mm, (W, H), interpolation=cv2.INTER_NEAREST)
            gt[ci] = (mm > 127).astype(np.float32); cov[ci] += 1
        for c in range(NCLASS):
            inter[c] += (pred[c] * gt[c]).sum(); uni[c] += pred[c].sum() + gt[c].sum()
        if k % 10 == 0: print(f"  {k}/{min(40,len(use_ids))}")
    dice = (2 * inter) / (uni + 1e-6)
    print(f"\nDice (n={min(40, len(use_ids))}):")
    for c, les in enumerate(LESIONS):
        print(f"  {les} ({LES_NAME[les]}): {dice[c]:.3f}   [มี GT {cov[c]} ภาพ]")
    print(f"  mean: {dice.mean():.3f}")
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
out = r"C:\Users\Acer\Downloads\Retina Classifier\RetinaPlus_NB3_LesionRationale.ipynb"
with io.open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("WROTE", out, "| cells:", len(cells))
