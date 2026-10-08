"""
RetinaPlus — AI Diabetic Retinopathy Screening (web)
Flask + PyTorch (EfficientNet-B4 ordinal regression) + Grad-CAM
โครงเดียวกับ Cough AI: เสิร์ฟ static html + /predict API + Cloud Run ready
"""
import os, io, json, base64, datetime, traceback
from pathlib import Path

import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import torch
import torch.nn as nn
import timm

# ------------------------------------------------------------------ config
HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"
HISTORY_FILE = Path(os.environ.get("HISTORY_FILE") or (HERE / "history.json"))
IMG_SIZE = 380
BACKBONE = "efficientnet_b4"
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
# OptimizedRounder coefs จาก v4 (เส้นแบ่งเกรด) + referable = coef[1]
ROUNDER_COEFS = [0.515, 1.264, 2.376, 3.079]
REFERABLE_THR = ROUNDER_COEFS[1]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------ model
def build_model():
    m = timm.create_model(BACKBONE, pretrained=False, num_classes=1,
                          drop_rate=0.3, drop_path_rate=0.2)
    return m.to(DEVICE).eval()

MEMBERS = []
def load_models():
    """โหลด best*.pt ทุกตัวใน models/ มาเป็น ensemble (ถ้าไม่มี = โหมด demo)"""
    global MEMBERS
    MEMBERS = []
    if MODELS_DIR.exists():
        for p in sorted(MODELS_DIR.glob("*.pt")):
            try:
                m = build_model()
                m.load_state_dict(torch.load(str(p), map_location=DEVICE))
                MEMBERS.append(m)
                print(f"[model] loaded {p.name}")
            except Exception as e:
                print(f"[model] skip {p.name}: {e}")
    print(f"[model] {len(MEMBERS)} model(s) | device={DEVICE}")
load_models()

# ------------------------------------------------------------------ preprocessing
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
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # RGB uint8

def to_tensor(rgb):
    x = ((rgb / 255.0 - _MEAN) / _STD).transpose(2, 0, 1).astype(np.float32)
    return torch.from_numpy(x).unsqueeze(0).to(DEVICE)

def grade_from_score(s):
    return int(np.clip(np.digitize([s], ROUNDER_COEFS)[0], 0, 4))

def is_fundus(bgr):
    """heuristic: ภาพจอตา = โทนแดง/ส้ม + มีบริเวณสว่างเป็นก้อนใหญ่"""
    img = cv2.resize(bgr, (256, 256))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(float)
    redness = float(rgb[:, :, 0].mean() - rgb[:, :, 2].mean())   # R - B
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 18, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fill = (max(cv2.contourArea(c) for c in cnts) / (256 * 256)) if cnts else 0.0
    return (redness > 18) and (fill > 0.15)

# ------------------------------------------------------------------ inference + grad-cam
@torch.no_grad()
def ensemble_score(x):
    ss = []
    for m in MEMBERS:
        o = m(x).float().squeeze(1)
        o = o + m(torch.flip(x, [3])).float().squeeze(1) + m(torch.flip(x, [2])).float().squeeze(1)
        ss.append(float((o / 3).item()))
    return float(np.mean(ss))

def gradcam_overlay(rgb, x):
    """heatmap จากโมเดลตัวแรก + mask วงจอตา -> base64 png"""
    try:
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.image import show_cam_on_image
        m0 = MEMBERS[0]
        tl = m0.blocks[-2] if (hasattr(m0, "blocks") and len(m0.blocks) >= 2) else m0.conv_head
        cam = GradCAMPlusPlus(model=m0, target_layers=[tl])
        class T:
            def __call__(self, o): return o[0]
        gray = cam(input_tensor=x, targets=[T()])[0]
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        mask = cv2.erode((g > 15).astype(np.float32), np.ones((25, 25), np.uint8))
        gray = gray * mask
        if gray.max() > 1e-6: gray = gray / gray.max()
        ov = show_cam_on_image(rgb.astype(np.float32) / 255.0, gray, use_rgb=True)
        ok, buf = cv2.imencode(".png", cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buf).decode() if ok else None
    except Exception as e:
        print("[gradcam] skip:", e); return None

def b64_png(rgb):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode() if ok else None

# ------------------------------------------------------------------ history
def save_history(rec):
    try:
        data = load_history()
        data.insert(0, rec); data = data[:200]
        HISTORY_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass   # read-only filesystem (e.g. HF Spaces) — skip persistence silently

def load_history(limit=200):
    if HISTORY_FILE.exists():
        try: return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))[:limit]
        except Exception: return []
    return []

# ------------------------------------------------------------------ flask
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

@app.route("/")
def home(): return send_from_directory(".", "homepage.html")
@app.route("/detect")
def detect(): return send_from_directory(".", "detect.html")
@app.route("/dashboard")
def dash(): return send_from_directory(".", "dashboard.html")

@app.route("/<path:f>")
def assets(f):
    if (HERE / f).is_file(): return send_from_directory(".", f)
    return ("", 404)

@app.route("/status")
def status():
    return jsonify({"ok": True, "models": len(MEMBERS), "device": DEVICE,
                    "demo": len(MEMBERS) == 0})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "no image"}), 400
        raw = np.frombuffer(request.files["image"].read(), np.uint8)
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if bgr is None: return jsonify({"error": "bad image"}), 400

        if not is_fundus(bgr):
            return jsonify({"not_fundus": True,
                "message_th": "ภาพนี้ไม่ใช่ภาพจอประสาทตา — กรุณาถ่าย/อัปโหลดภาพถ่ายจอตา (fundus)",
                "message_en": "This is not a retinal fundus image — please capture/upload a fundus photo."})

        rgb = preprocess(bgr)

        if len(MEMBERS) == 0:                         # demo mode (ไม่มีโมเดล)
            score = 1.9
            res = {"demo": True}
        else:
            x = to_tensor(rgb)
            score = ensemble_score(x)
            res = {"demo": False}

        grade = grade_from_score(score)
        referable = score >= REFERABLE_THR
        # map score 0-4 -> referable risk % (sigmoid รอบ threshold)
        risk = float(1 / (1 + np.exp(-(score - REFERABLE_THR) * 2.2)) * 100)
        cam = gradcam_overlay(rgb, x) if len(MEMBERS) else None

        res.update({
            "grade": grade, "grade_name": CLASS_NAMES[grade],
            "score": round(score, 2),
            "referable": bool(referable),
            "risk_percent": round(risk, 1),
            "preprocessed": b64_png(rgb),
            "gradcam": cam,
        })
        save_history({"time": datetime.datetime.now().isoformat(timespec="seconds"),
                      "grade": grade, "grade_name": CLASS_NAMES[grade],
                      "referable": bool(referable), "risk_percent": round(risk, 1)})
        return jsonify(res)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/history")
def history(): return jsonify(load_history())

@app.route("/stats")
def stats():
    h = load_history()
    n = len(h); ref = sum(1 for x in h if x.get("referable"))
    dist = {c: 0 for c in CLASS_NAMES}
    for x in h: dist[x.get("grade_name", "No DR")] = dist.get(x.get("grade_name", "No DR"), 0) + 1
    return jsonify({"total": n, "referable": ref, "normal": n - ref, "dist": dist})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
