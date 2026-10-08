# %% [markdown]
# # Phase 2 — YOLOv8 Lesion Detection (blood spots / exudates)
#
# ตรวจจับรอยโรค (microaneurysm, hemorrhage, exudate) เป็นกล่อง — โชว์ "AI ชี้รอยโรค"
# ก่อนรัน: GPU T4 + Add Data: **DR lesion dataset แบบ YOLO format** (มี data.yaml)
#   ค้น Kaggle: "diabetic retinopathy lesion yolo" หรือ "DDR yolo" หรือ Roboflow DR detection
#
# ⚠️ ต้องเป็น YOLO format (มีไฟล์ data.yaml + images/ + labels/) — รัน cell 2 เช็คก่อน
#    ถ้าไม่เจอ data.yaml ส่ง log มา เดี๋ยวเขียนตัวแปลง mask->bbox จาก IDRiD-Segmentation ให้

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=False)
import os, glob, shutil, yaml
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
import torch
assert torch.cuda.is_available(), "GPU ไม่เจอ! Settings -> Accelerator -> GPU"
OUT = Path("/kaggle/working"); OUT.mkdir(exist_ok=True)

# %%
# ---- หา data.yaml ของ dataset YOLO ----
yamls = glob.glob("/kaggle/input/**/data.yaml", recursive=True) + glob.glob("/kaggle/input/**/*.yaml", recursive=True)
yamls = [y for y in yamls if "data" in os.path.basename(y).lower() or "dataset" in os.path.basename(y).lower()]
assert yamls, ("ไม่เจอ data.yaml — เพิ่ม DR lesion dataset แบบ YOLO format\n"
               "(ค้น Kaggle: diabetic retinopathy lesion yolo) แล้วรันใหม่")
DATA = yamls[0]
print(f"[yolo] data.yaml: {DATA}")
with open(DATA) as f: cfg = yaml.safe_load(f)
print("classes:", cfg.get("names"))

# Kaggle input อ่านอย่างเดียว -> ถ้า path ใน yaml ชี้ relative ให้ ultralytics จัดการ; ก๊อป yaml มา working กันแก้ path
local_yaml = OUT / "data.yaml"
# แก้ path ให้ absolute ถ้าจำเป็น
root = os.path.dirname(DATA)
for k in ("train", "val", "test"):
    if k in cfg and isinstance(cfg[k], str) and not os.path.isabs(cfg[k]):
        cand = os.path.normpath(os.path.join(root, cfg[k]))
        if os.path.exists(cand): cfg[k] = cand
cfg.setdefault("path", root)
yaml.safe_dump(cfg, open(local_yaml, "w"))
print(f"[yolo] ใช้ {local_yaml}")

# %%
# ---- เทรน YOLOv8n ----
model = YOLO("yolov8n.pt")
model.train(data=str(local_yaml), epochs=60, imgsz=640, batch=16,
            project=str(OUT), name="dr_yolo", seed=42, patience=15, verbose=True)
# ผลเทรน (curves, metrics) ถูกเซฟใน OUT/dr_yolo/ อัตโนมัติ (results.png, confusion ฯลฯ)
print("[yolo] เทรนเสร็จ — กราฟอยู่ใน /kaggle/working/dr_yolo/")

# %%
# ---- predict โชว์รูปตรวจจับรอยโรค ----
best_w = OUT / "dr_yolo" / "weights" / "best.pt"
det = YOLO(str(best_w))
# หารูปทดสอบ
val_dir = cfg.get("val") or cfg.get("test") or cfg.get("train")
imgs = []
for ext in ("*.jpg", "*.png", "*.jpeg"):
    imgs += glob.glob(os.path.join(str(val_dir), "**", ext), recursive=True)
imgs = imgs[:6]
if imgs:
    res = det.predict(imgs, imgsz=640, conf=0.25, save=False)
    n = len(res); fig, axs = plt.subplots((n + 2) // 3, 3, figsize=(13, 4 * ((n + 2) // 3)))
    axs = axs.ravel()
    for i, r in enumerate(res):
        axs[i].imshow(cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB))
        axs[i].set_title(f"{len(r.boxes)} lesions", fontsize=10)
        axs[i].set_xticks([]); axs[i].set_yticks([])
    for j in range(n, len(axs)): axs[j].axis("off")
    fig.suptitle("YOLOv8 — Lesion Detection (blood spots / exudates)", fontsize=14)
    fig.tight_layout(); fig.savefig(OUT / "yolo_detections.png", dpi=130); plt.close(fig)
    print("[saved] yolo_detections.png")
print("✅ เสร็จ — dr_yolo/ (กราฟ+best.pt) + yolo_detections.png อยู่ใน /kaggle/working")
