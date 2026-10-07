---
title: RetinaPlus
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8080
pinned: false
short_description: AI diabetic retinopathy screening + lesion explanation
---

# RetinaPlus — AI Diabetic Retinopathy Screening

เครื่องมือ **คัดกรองเบาหวานขึ้นจอประสาทตา (DR)** จากภาพ fundus ด้วย AI พร้อม **คำอธิบายรอยโรคช่วยแพทย์** · Flask + PyTorch

> ⚠️ เครื่องมือ **ช่วยตัดสินใจ (decision-support)** ไม่ใช่การวินิจฉัยขั้นสุดท้าย — ผลต้องได้รับการยืนยันโดยจักษุแพทย์

## Pipeline
```
ภาพ → [1] Quality gate (heuristic + EyeQ, 2 โหมด: กล้องมาตรฐาน / มือถือ+20D)
        → [2] คัดกรอง: ensemble v3+v4 (EfficientNet-B4 ordinal) + TTA → calibrated risk
        → [3] โซน: 🟢 ไม่ต้องส่งต่อ / 🔴 ควรพบจักษุแพทย์ (grade-anchored)
        → [4] อธิบายรอยโรค (lazy): UNet++ segment MA/HE/EX/SE + disc/fovea/ETDRS + ระยะ DD
```

- **คัดกรองเร็ว** (`/predict` ~3s) แยกจาก **อธิบายรอยโรค** (`/explain`) ที่โหลดเฉพาะตอนเปิด "ข้อมูลเชิงลึก"
- หน้าจอ 2 ชั้น: คนคัดกรองเห็นผล พบ/ไม่พบแพทย์ + ความเสี่ยง · แพทย์กดดู overlay รอยโรค + รายงานเชิงตัวเลข

## ผลการประเมิน (external validation)
| ชุด | QWK | Referable AUC | sens / spec |
|---|---|---|---|
| internal test | 0.88 | 0.94 | 0.83 / 0.94 |
| external IDRiD | 0.85 | 0.98 | 0.93 / 0.91 |
| external DDR (n=12,522) | 0.80 | 0.95 | — |

## หน้าเว็บ
- `/` หน้าแรก · `/detect` ตรวจ (ถ่าย/อัปโหลด) · `/dashboard` สรุปสถิติ

## โมเดล (`models/`)
`v3_best.pt` + `v4_best.pt` (classifier ensemble) · `unet_lesion_v2.pt` (lesion) · `quality_gate.pt` (EyeQ B0)
config (`config/`): `calibrator.pkl`, `thresholds.json`, `quality_gate_config.json`

## รัน Local
```bash
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python app.py        # http://localhost:8080
```

## Deploy
- **Hugging Face Spaces (Docker):** push repo → build จาก Dockerfile อัตโนมัติ (`app_port: 8080`)
- **Cloud Run:** `gcloud run deploy retinaplus --source . --memory 2Gi --cpu 2 --allow-unauthenticated`

## API
| route | |
|---|---|
| `GET /status` | สถานะโมเดล + โปรไฟล์ quality gate |
| `POST /predict` | form-data `image`, `mode` → JSON: grade, risk_percent, zone, layers.raw |
| `POST /explain` | form-data `image` → lesion layers + features + rationale + disc/fovea (lazy) |
| `GET /history` · `GET /stats` | ประวัติ + สรุป |

## ข้อจำกัด / งานในอนาคต
- โมเดลเทรนบน **กล้อง fundus ตั้งโต๊ะ** — ภาพ **มือถือ+เลนส์ 20D** เป็น domain shift ต้องเก็บภาพจริง + fine-tune + recalibrate ก่อนใช้คลินิก
- จุดอ่อน: Moderate precision, EyeQ gate เข้มกับบางโดเมน → แก้ด้วย fine-tune (NB5) + ONNX/int8 (NB4) ในอนาคต

Developed by Nathakorn
