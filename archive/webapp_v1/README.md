---
title: RetinaPlus
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8080
pinned: false
short_description: AI diabetic retinopathy screening
---

# RetinaPlus — AI Diabetic Retinopathy Screening

เครื่องมือคัดกรองเบาหวานขึ้นจอตาด้วย AI · Flask + PyTorch (EfficientNet-B4 ordinal regression, ensemble v3+v4 + TTA) + Grad-CAM

> ⚠️ เครื่องมือคัดกรองเบื้องต้น ไม่ใช่การวินิจฉัย ผลควรได้รับการยืนยันโดยจักษุแพทย์

## หน้าเว็บ
- `/` หน้าแรก · `/detect` ถ่าย/อัปโหลดภาพจอตา → ผลตรวจ · `/dashboard` สรุปผล

## โมเดล
EfficientNet-B4 ensemble (2 seeds) — `models/v3_best.pt`, `models/v4_best.pt`
ถ้าโฟลเดอร์ว่าง → รันได้แต่เป็น "โหมดสาธิต"

## รัน Local
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python app.py        # http://localhost:8080
```

## Deploy
- **Hugging Face Spaces (Docker):** push repo นี้ → HF build จาก Dockerfile อัตโนมัติ (`app_port: 8080`)
- **Google Cloud Run:** `gcloud run deploy retinaplus --source . --memory 2Gi --cpu 2 --allow-unauthenticated`

## API
| route | |
|---|---|
| `GET /status` | สถานะโมเดล |
| `POST /predict` | form-data `image` → JSON: grade, score, referable, risk_percent, gradcam(base64) |
| `GET /history` · `GET /stats` | ประวัติ + สรุป |

Developed by Nathakorn
