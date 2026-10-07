# AI Diabetic Retinopathy Screening — Kaggle

โมเดลเดียว EfficientNet-B3 (PyTorch) ออก 2 ผลลัพธ์: **เป็น/ไม่เป็น (Referable DR)** + **ระดับ 0–4**

## ไฟล์ในโฟลเดอร์นี้
- `dr_screening_kaggle.ipynb` ← **เอาอันนี้ขึ้น Kaggle** (notebook พร้อมรัน)
- `dr_screening_kaggle.py` ← โค้ดเดียวกันแบบ flat script (ก๊อปวาง cell เดียวก็ได้)
- `_to_ipynb.py` ← ตัวแปลง .py → .ipynb (ถ้าแก้ .py แล้วอยากสร้าง notebook ใหม่)

## วิธีรันบน Kaggle (≈10 นาที setup + เทรน 5–8 ชม.)

1. ไป **kaggle.com → Create → New Notebook**
2. มุมขวา **Notebook → File → Import Notebook** → อัป `dr_screening_kaggle.ipynb`
   (หรือสร้าง notebook เปล่าแล้วก๊อปเนื้อ `.py` วาง cell เดียว)
3. แถบขวา **Settings → Accelerator → GPU** (P100 หรือ T4 ×2) ← **สำคัญ ถ้าไม่เปิดโค้ดจะหยุดทันที**
4. แถบขวา **Add Data** → ค้นและ Add 2 ชุด:
   - `eyepacs aptos messidor diabetic retinopathy`  (train + val)
   - `IDRiD`  (external test — ค้น "IDRiD" เลือกตัวที่มี Disease Grading)
5. กด **Run All** → ปิดเบราว์เซอร์ได้ Kaggle รันต่อให้

## ผลลัพธ์ (อยู่ใน `/kaggle/working` → โหลดกลับมาได้จากแถบ Output)
| ไฟล์ | คืออะไร |
|---|---|
| `best.pt` | น้ำหนักโมเดลที่ดีสุด |
| `metrics.json` | accuracy, QWK, F2, AUC, sensitivity/specificity (internal + external) |
| `confusion_external_idrid.png` | confusion matrix บนข้อสอบจริง |
| `roc_curve.png` | ROC curve (referable DR) |
| `training_curves.png` | accuracy / loss / precision / recall |
| `gradcam_eyes.png` | ภาพเปรียบเทียบดวงตา original / preprocessed / heatmap |

## การแบ่งข้อมูล (กัน leakage)
- **train + val** = ชุด combined (143k) — แบ่งตาม split ที่มี หรือ 85/15 ถ้าไม่มี val
- **external test = IDRiD** — โมเดลไม่เคยเห็น → คะแนนน่าเชื่อถือ (ตัวที่เอาขึ้นสไลด์)
- รายงานทั้ง internal (optimistic) และ external (trustworthy) ให้เห็นต่าง

## ปรับแต่ง (แก้ใน cell config — section 0)
- `TRAIN_FRAC = 0.15` → ทดสอบเร็วๆ ก่อนรันเต็ม
- `BATCH` → ถ้า OOM โค้ดลดครึ่งให้อัตโนมัติ
- `EPOCHS_STAGE2`, `IMG_SIZE` → จูนความแม่น/เวลา

## รันบนเครื่องตัวเอง (ถ้าไม่ใช้ Kaggle)
ตั้ง env `DR_INPUT` ชี้ไปโฟลเดอร์ที่มีทั้ง combined + IDRiD แล้ว `python dr_screening_kaggle.py`
(ต้องมี NVIDIA GPU + CUDA — โค้ดบังคับ GPU)
