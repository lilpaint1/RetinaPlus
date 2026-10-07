# ── RetinaPlus · Cloud Run ──
FROM python:3.11-slim

# lib ที่ opencv ต้องใช้
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# torch CPU build (เล็กกว่า + Cloud Run ไม่มี GPU)
RUN pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run ส่ง PORT=8080
ENV PORT=8080
# history เขียนที่ /tmp (เขียนได้เสมอ ทั้ง HF Spaces / Cloud Run)
ENV HISTORY_FILE=/tmp/retinaplus_history.json

# 1 worker / หลาย thread → โมเดลโหลดครั้งเดียวแชร์ใน RAM
# timeout 0 = ไม่ตัด request ระหว่างประมวลผลภาพ/Grad-CAM
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
