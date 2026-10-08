# RetinaPlus — AI Diabetic Retinopathy Screening

ระบบ AI **คัดกรองเบาหวานขึ้นจอประสาทตา (Diabetic Retinopathy, DR)** จากภาพ fundus พร้อม **อธิบายรอยโรคช่วยแพทย์** ออกแบบให้ใช้ร่วมกับอุปกรณ์ถ่ายภาพจอตาราคาประหยัด (อะแดปเตอร์ 3D print + เลนส์ 20D ต่อกับสมาร์ทโฟน) เพื่อลดช่องว่างการเข้าถึงการคัดกรองใน รพ.สต.

> ⚠️ เป็นเครื่องมือ **ช่วยตัดสินใจ (decision-support)** ไม่ใช่การวินิจฉัยขั้นสุดท้าย ผลต้องได้รับการยืนยันโดยจักษุแพทย์

| | |
|---|---|
| 🌐 เว็บ demo | https://retinaplus-app.hf.space/ |
| 🎬 วิดีโอ demo | https://youtu.be/LSEzxK5R0N4 |

## Pipeline

```
ภาพ fundus
  → [1] Quality gate     heuristic + EfficientNet-B0 (EyeQ) — กรองภาพไม่ใช่จอตา / ไม่ชัด
  → [2] คัดกรอง DR        ensemble EfficientNet-B4 (v3 + v4) ordinal regression + TTA → calibrated risk
  → [3] โซนการส่งต่อ       🟢 ไม่ต้องส่งต่อ / 🔴 ควรพบจักษุแพทย์
  → [4] อธิบายรอยโรค      UNet++ (EfficientNet-B4 encoder) segment MA / HE / EX / SE
                          → แปลง mask เป็นตัวเลข + ประโยคเหตุผล (แทน Grad-CAM)
```

- หน้าจอ 2 ชั้น: ผู้คัดกรองเห็นผลพบ/ไม่พบแพทย์ + ความเสี่ยง · แพทย์ดู overlay รอยโรคและรายงานเชิงตัวเลขได้
- `/predict` (คัดกรอง) แยกจาก `/explain` (อธิบายรอยโรค) เพื่อให้ผลหลักออกเร็ว

## ผลการประเมิน

| ชุดข้อมูล | QWK | Referable AUC | Sensitivity / Specificity |
|---|---|---|---|
| internal test | 0.88 | 0.94 | 0.83 / 0.94 |
| external — IDRiD | 0.85 | 0.98 | 0.93 / 0.91 |
| external — DDR (n=12,522) | 0.80 | 0.95 | — |

- Train/val: EyePACS + APTOS + Messidor (รวมประมาณ 143k ภาพ) · external test ไม่ถูกใช้ตอนเทรน
- Notebook ที่ใช้ประเมินอยู่ใน [`notebooks/`](notebooks/) (inference อย่างเดียว ใช้ preprocess / TTA / threshold เดียวกับ `app.py`)

## ข้อจำกัด (limitations)

- การ validate ทั้งหมดใช้ภาพจาก **กล้อง fundus มาตรฐาน** ยังไม่ได้ validate กับภาพที่ถ่ายผ่านอุปกรณ์ 3D print + มือถือจริง ซึ่งมี domain shift (คุณภาพ แสง มุมมอง) ต้องทดสอบเพิ่มก่อนใช้งานจริง
- ยังไม่ผ่านการรับรองทางการแพทย์/ขออนุมัติจริยธรรมการวิจัย (IRB) ไม่ควรใช้ตัดสินการรักษา
- Quality gate เป็นด่านกรองเบื้องต้น ไม่ได้รับประกันว่าภาพที่ผ่านจะเหมาะกับการวินิจฉัยทุกกรณี

## โครงสร้าง repo

```
hf-space/     แอปที่ deploy จริง (Flask + PyTorch, Dockerfile, หน้าเว็บ, config/ สำหรับ calibrator และ threshold)
notebooks/    ขั้นตอนทั้งหมดตามลำดับ: เทรน v3 / v4 → quality gate → lesion rationale → external validation
archive/      เวอร์ชันเก่า, การทดลอง (YOLO lesion, U-Net vessel, Grad-CAM), สคริปต์ช่วยงาน, webapp เวอร์ชันแรก
```

ตัวที่ใช้จริงคือ `hf-space/` + notebook ใน `notebooks/` ส่วน `archive/` เก็บไว้เป็นประวัติการพัฒนา (ไม่ได้ใช้ใน pipeline ปัจจุบัน)

## รัน local

น้ำหนักโมเดล (`models/*.pt`) **ไม่ได้อยู่ใน repo นี้** ถ้าไม่มี แอปจะรันในโหมดสาธิต วางไฟล์ไว้ที่ `hf-space/models/`:
`v3_best.pt`, `v4_best.pt` (classifier), `unet_lesion_v2.pt` (lesion), `quality_gate.pt` (EyeQ B0)

```bash
cd hf-space
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python app.py        # http://localhost:8080
```

## API

| route | |
|---|---|
| `GET /status` | สถานะโมเดลและโปรไฟล์ quality gate |
| `POST /predict` | form-data `image`, `mode` → JSON: grade, risk_percent, zone |
| `POST /explain` | form-data `image` → lesion layers + features + rationale |
| `GET /history` · `GET /stats` | ประวัติและสรุปสถิติ |

## Deploy

- **Hugging Face Spaces (Docker):** ใช้ `hf-space/` เป็นรากของ Space (`app_port: 8080`)
- **Google Cloud Run:** `gcloud run deploy retinaplus --source hf-space --memory 2Gi --cpu 2 --allow-unauthenticated`

## Tech stack

PyTorch · timm (EfficientNet-B4/B0) · segmentation-models-pytorch (UNet++) · OpenCV · Flask + Gunicorn · Docker
