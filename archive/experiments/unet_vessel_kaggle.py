# %% [markdown]
# # Phase 2 — U-Net Retinal Vessel Segmentation (DRIVE)
#
# แยกเส้นเลือดจอตาเป็นภาพขาว — สำหรับโชว์ "AI วิเคราะห์เส้นเลือด" (แบบ KidneyLife+)
# ก่อนรัน: GPU T4 + Add Data: **DRIVE** (ค้น "DRIVE retinal" บน Kaggle)
#
# ⚠️ DRIVE บน Kaggle มีหลายโครงสร้าง — รัน cell 1-2 ก่อน ดู print ว่าจับคู่ภาพ/mask ได้กี่คู่
#    ถ้า 0 คู่ ส่ง log มา เดี๋ยวปรับ loader (เหมือนตอน IDRiD)

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch"], check=False)

import os, glob, re, random, warnings
from pathlib import Path
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from PIL import Image
warnings.filterwarnings("ignore")

SEED = 42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
assert torch.cuda.is_available(), "GPU ไม่เจอ! Settings -> Accelerator -> GPU"
DEVICE = "cuda"; OUT = Path("/kaggle/working"); OUT.mkdir(exist_ok=True)
SIZE = 512; EPOCHS = 40; BATCH = 4

# %%
# ---- discover DRIVE: จับคู่ภาพ <-> vessel mask ด้วยเลขนำหน้าชื่อไฟล์ ----
IMG_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".gif", ".ppm", ".bmp"}
def walk_imgs(filterfn):
    out = []
    for r, _, fs in os.walk("/kaggle/input"):
        for f in fs:
            p = os.path.join(r, f)
            if os.path.splitext(f)[1].lower() in IMG_EXT and filterfn(r.lower(), f.lower()):
                out.append(p)
    return out

def num_key(path):
    m = re.search(r"(\d+)", os.path.basename(path)); return m.group(1) if m else None

# ภาพ = อยู่ในโฟลเดอร์ 'image'; mask เส้นเลือด = โฟลเดอร์/ชื่อมี 'manual' หรือ 'vessel' หรือ '1st'
imgs  = walk_imgs(lambda d, f: "image" in d and "manual" not in d and "mask" not in d)
masks = walk_imgs(lambda d, f: ("manual" in d or "manual" in f or "vessel" in d or "1st" in d or "1st" in f))
# จับคู่ด้วยเลข + แยก train/test ด้วยคำว่า 'train'/'test' ใน path
def build(split):
    mk = {num_key(m): m for m in masks if split in m.lower()}
    pairs = [(i, mk[num_key(i)]) for i in imgs if split in i.lower() and num_key(i) in mk]
    return pairs
train_pairs = build("train"); test_pairs = build("test")
if not train_pairs:  # fallback: ไม่มีคำ train/test -> จับคู่ทั้งหมดแล้วแบ่งเอง
    mk = {num_key(m): m for m in masks}
    allp = [(i, mk[num_key(i)]) for i in imgs if num_key(i) in mk]
    random.shuffle(allp); k = int(len(allp) * 0.8)
    train_pairs, test_pairs = allp[:k], allp[k:]
print(f"[DRIVE] train pairs={len(train_pairs)}  test pairs={len(test_pairs)}")
assert len(train_pairs) > 0, "จับคู่ภาพ/mask ไม่ได้ — ส่ง log มาปรับ loader"
print("ตัวอย่าง:", os.path.basename(train_pairs[0][0]), "<->", os.path.basename(train_pairs[0][1]))

# %%
# ---- dataset ----
def read_img(p):
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None: img = np.array(Image.open(p).convert("RGB"))[:, :, ::-1]
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def read_mask(p):
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if m is None: m = np.array(Image.open(p).convert("L"))
    return (m > 127).astype(np.float32)

_M = np.array([0.485, 0.456, 0.406]); _S = np.array([0.229, 0.224, 0.225])
class VesselDS(Dataset):
    def __init__(self, pairs, train): self.pairs = pairs; self.train = train
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        img = cv2.resize(read_img(ip), (SIZE, SIZE))
        msk = cv2.resize(read_mask(mp), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        if self.train and random.random() < 0.5:
            img = img[:, ::-1].copy(); msk = msk[:, ::-1].copy()
        img = ((img / 255.0 - _M) / _S).transpose(2, 0, 1).astype(np.float32)
        return torch.from_numpy(img), torch.from_numpy(msk[None])
tr = DataLoader(VesselDS(train_pairs, True), batch_size=BATCH, shuffle=True, num_workers=2)
te = DataLoader(VesselDS(test_pairs, False), batch_size=1, shuffle=False, num_workers=2)

# %%
# ---- U-Net (smp, resnet34 encoder) + Dice+BCE ----
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
bce = nn.BCEWithLogitsLoss()
def dice_loss(p, y, e=1.):
    p = torch.sigmoid(p); inter = (p * y).sum((2, 3))
    return (1 - (2 * inter + e) / (p.sum((2, 3)) + y.sum((2, 3)) + e)).mean()
scaler = torch.cuda.amp.GradScaler()
from tqdm.auto import tqdm

def dice_score(p, y, e=1.):
    p = (torch.sigmoid(p) > 0.5).float(); inter = (p * y).sum((2, 3))
    return ((2 * inter + e) / (p.sum((2, 3)) + y.sum((2, 3)) + e)).mean().item()

best = 0
for ep in range(1, EPOCHS + 1):
    model.train()
    for x, y in tqdm(tr, desc=f"epoch {ep}/{EPOCHS}", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE); opt.zero_grad()
        with torch.autocast("cuda"):
            o = model(x); loss = bce(o, y) + dice_loss(o, y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    model.eval(); ds = []
    with torch.no_grad():
        for x, y in te:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.autocast("cuda"): ds.append(dice_score(model(x), y))
    d = float(np.mean(ds)); print(f"epoch {ep}: test Dice={d:.4f}")
    if d > best: best = d; torch.save(model.state_dict(), OUT / "unet_vessel.pt")
print(f"[best Dice] {best:.4f}  -> unet_vessel.pt")

# %%
# ---- โชว์ภาพ: original | ground truth | predicted vessel ----
model.load_state_dict(torch.load(OUT / "unet_vessel.pt")); model.eval()
n = min(4, len(test_pairs)); fig, axs = plt.subplots(n, 3, figsize=(10, 3.2 * n))
if n == 1: axs = axs[None, :]
for r in range(n):
    ip, mp = test_pairs[r]
    img = cv2.resize(read_img(ip), (SIZE, SIZE)); gt = cv2.resize(read_mask(mp), (SIZE, SIZE))
    x = torch.from_numpy(((img / 255.0 - _M) / _S).transpose(2, 0, 1).astype(np.float32))[None].to(DEVICE)
    with torch.no_grad(), torch.autocast("cuda"):
        pr = (torch.sigmoid(model(x))[0, 0].float().cpu().numpy() > 0.5)
    axs[r, 0].imshow(img); axs[r, 1].imshow(gt, cmap="gray"); axs[r, 2].imshow(pr, cmap="gray")
    if r == 0:
        axs[r, 0].set_title("Original"); axs[r, 1].set_title("Ground truth"); axs[r, 2].set_title("U-Net predicted")
    for c in range(3): axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
fig.suptitle(f"Retinal Vessel Segmentation — U-Net (Dice={best:.3f})", fontsize=14)
fig.tight_layout(); fig.savefig(OUT / "unet_vessel_results.png", dpi=130); plt.close(fig)
print("✅ เสร็จ — unet_vessel.pt + unet_vessel_results.png อยู่ใน /kaggle/working")
