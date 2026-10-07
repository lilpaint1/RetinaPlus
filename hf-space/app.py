"""
RetinaPlus — AI Diabetic Retinopathy Screening (web)
Flask + PyTorch. Pipeline:
  quality gate (heuristic + EyeQ, mobile/20D aware)
    -> ensemble v3+v4 (EfficientNet-B4 ordinal) + TTA
    -> isotonic calibration -> 3 zones (normal / review / refer)
    -> UNet++ lesion segmentation -> rationale + disc/fovea/ETDRS + DD distance
Returns layered overlays (raw base + lesion layer + landmark layer) for toggling.
Runs local (python app.py) and server (gunicorn) — HF Spaces / Cloud Run ready.
"""
import os, io, json, base64, datetime, traceback
from pathlib import Path

import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import torch
import timm

# ------------------------------------------------------------------ paths / config
HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"
CONFIG_DIR = HERE / "config"
HISTORY_FILE = Path(os.environ.get("HISTORY_FILE") or (HERE / "history.json"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 380
BACKBONE = "efficientnet_b4"
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
LESIONS = ["MA", "HE", "EX", "SE"]
PALETTE = {"MA": (255, 0, 0), "HE": (0, 0, 255), "EX": (0, 255, 0), "SE": (255, 255, 0)}
LES_NAME = {"MA": "microaneurysm", "HE": "hemorrhage", "EX": "hard exudate", "SE": "soft exudate/cotton-wool"}
WORK = int(os.environ.get("LESION_WORK", 1024))   # UNet sliding-window resolution (เล็กลง=เร็วขึ้นบน CPU)
PATCH = 768
MIN_AREA = 4
_MEAN = np.array([0.485, 0.456, 0.406]); _STD = np.array([0.229, 0.224, 0.225])

def _load_json(name, default):
    p = CONFIG_DIR / name
    if p.exists():
        try: return json.loads(p.read_text(encoding="utf-8"))
        except Exception: pass
    return default

THRESH = _load_json("thresholds.json", {"rounder_coefs": [0.515, 1.264, 2.376, 3.079],
                                         "P_LO": 0.03, "P_HI": 0.33})
ROUNDER_COEFS = THRESH.get("rounder_coefs", [0.515, 1.264, 2.376, 3.079])
P_LO = float(THRESH.get("P_LO", 0.03)); P_HI = float(THRESH.get("P_HI", 0.33))
QCFG = _load_json("quality_gate_config.json", {
    "QIMG": 384, "classes": ["Good", "Usable", "Reject"], "default_profile": "fundus",
    "profiles": {"fundus": {"AREA_MIN": 0.3, "BLUR_MIN": 15, "BRI_LO": 40, "BRI_HI": 150,
                            "REDNESS_MIN": 12, "REJECT_THR": 0.55, "use_model": True},
                 "mobile": {"AREA_MIN": 0.05, "BLUR_MIN": 7, "BRI_LO": 22, "BRI_HI": 220,
                            "REDNESS_MIN": 6, "REJECT_THR": 0.9, "use_model": False}}})
QIMG = int(QCFG.get("QIMG", 384))

# ------------------------------------------------------------------ models
def build_dr():
    return timm.create_model(BACKBONE, pretrained=False, num_classes=1,
                             drop_rate=0.3, drop_path_rate=0.2).to(DEVICE).eval()

MEMBERS, QMODEL, UNET, CALIB = [], None, None, None

def load_models():
    global MEMBERS, QMODEL, UNET, CALIB
    # --- DR ensemble (v3 + v4) ---
    MEMBERS = []
    for p in sorted(MODELS_DIR.glob("*best*.pt")) + sorted(MODELS_DIR.glob("best*.pt")):
        if p in [getattr(m, "_path", None) for m in MEMBERS]: continue
        try:
            m = build_dr(); sd = torch.load(str(p), map_location=DEVICE)
            if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
            m.load_state_dict(sd); m._path = p; MEMBERS.append(m); print(f"[dr] {p.name}")
        except Exception as e: print(f"[dr] skip {p.name}: {e}")
    # --- quality gate (EfficientNet-B0 3-class) ---
    qp = MODELS_DIR / "quality_gate.pt"
    if qp.exists():
        try:
            QMODEL = timm.create_model("efficientnet_b0", pretrained=False, num_classes=3, drop_rate=0.2).to(DEVICE).eval()
            sd = torch.load(str(qp), map_location=DEVICE)
            if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
            QMODEL.load_state_dict(sd); print("[quality] loaded")
        except Exception as e: QMODEL = None; print(f"[quality] skip: {e}")
    # --- UNet++ lesion ---
    up = MODELS_DIR / "unet_lesion_v2.pt"
    if up.exists():
        try:
            import segmentation_models_pytorch as smp
            UNET = smp.UnetPlusPlus("efficientnet-b4", encoder_weights=None, in_channels=3, classes=4).to(DEVICE).eval()
            sd = torch.load(str(up), map_location=DEVICE)
            if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
            UNET.load_state_dict(sd); print("[unet] loaded")
        except Exception as e: UNET = None; print(f"[unet] skip: {e}")
    # --- calibrator (isotonic) ---
    cp = CONFIG_DIR / "calibrator.pkl"
    if cp.exists():
        try:
            import joblib; CALIB = joblib.load(str(cp)); print("[calib] loaded")
        except Exception as e: CALIB = None; print(f"[calib] skip: {e}")
    print(f"[ready] dr={len(MEMBERS)} quality={QMODEL is not None} unet={UNET is not None} calib={CALIB is not None} device={DEVICE}")

load_models()

# ------------------------------------------------------------------ preprocessing
def _crop_bbox(bgr, tol=7):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY); m = g > tol
    if m.sum() == 0: return 0, 0, bgr.shape[0], bgr.shape[1]
    c = np.argwhere(m); y0, x0 = c.min(0); y1, x1 = c.max(0) + 1
    return int(y0), int(x0), int(y1), int(x1)

def _clahe(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB); l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(2.0, (8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def dr_preprocess(bgr):
    y0, x0, y1, x1 = _crop_bbox(bgr); img = bgr[y0:y1, x0:x1]
    if img.size == 0: img = bgr
    img = _clahe(cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def to_tensor(rgb):
    x = ((rgb / 255.0 - _MEAN) / _STD).transpose(2, 0, 1).astype(np.float32)
    return torch.from_numpy(x).unsqueeze(0).to(DEVICE)

def crop_resize(bgr, work, clahe=True):
    """crop fundus + resize (max side=work). คืน rgb + bbox สำหรับ warp กลับ"""
    y0, x0, y1, x1 = _crop_bbox(bgr); crop = bgr[y0:y1, x0:x1]
    if crop.size == 0: crop = bgr; y0, x0, y1, x1 = 0, 0, bgr.shape[0], bgr.shape[1]
    h, w = crop.shape[:2]; s = work / max(h, w)
    r = cv2.resize(crop, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    if clahe: r = _clahe(r)
    return cv2.cvtColor(r, cv2.COLOR_BGR2RGB), (y0, x0, y1, x1)

# ------------------------------------------------------------------ quality gate
def fundus_metrics(bgr):
    y0, x0, y1, x1 = _crop_bbox(bgr); crop = bgr[y0:y1, x0:x1]
    if crop.size == 0: crop = bgr
    g = cv2.cvtColor(cv2.resize(crop, (QIMG, QIMG)), cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(g, cv2.CV_64F).var())
    bright = float(g.mean())
    fg = (cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) > 7).mean()
    rgb = cv2.cvtColor(cv2.resize(bgr, (256, 256)), cv2.COLOR_BGR2RGB).astype(float)
    redness = float(rgb[:, :, 0].mean() - rgb[:, :, 2].mean())
    return {"area_ratio": float(fg), "blur": blur, "bright": bright, "redness": redness}

@torch.no_grad()
def quality_prob_reject(bgr):
    if QMODEL is None: return None
    y0, x0, y1, x1 = _crop_bbox(bgr); crop = bgr[y0:y1, x0:x1]
    if crop.size == 0: crop = bgr
    rgb = cv2.cvtColor(cv2.resize(crop, (QIMG, QIMG), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    x = to_tensor(rgb)
    return float(torch.softmax(QMODEL(x), 1)[0, 2].cpu())   # P(Reject)

def check_quality(bgr, profile):
    pf = QCFG["profiles"].get(profile, QCFG["profiles"][QCFG["default_profile"]])
    m = fundus_metrics(bgr)
    if m["redness"] < pf["REDNESS_MIN"] or m["area_ratio"] < pf["AREA_MIN"]:
        return {"ok": False, "stage": "heuristic", "reason_th": "ไม่ใช่ภาพจอประสาทตา / ถ่ายไม่ติดจอ",
                "reason_en": "Not a fundus image / retina not captured", "metrics": m}
    if m["blur"] < pf["BLUR_MIN"]:
        return {"ok": False, "stage": "heuristic", "reason_th": "ภาพเบลอเกินไป ถ่ายใหม่",
                "reason_en": "Image too blurry, retake", "metrics": m}
    if not (pf["BRI_LO"] <= m["bright"] <= pf["BRI_HI"]):
        return {"ok": False, "stage": "heuristic", "reason_th": "ภาพมืด/สว่างเกินไป ถ่ายใหม่",
                "reason_en": "Image too dark/bright, retake", "metrics": m}
    qrej = None
    if pf.get("use_model", True) and QMODEL is not None:
        qrej = quality_prob_reject(bgr)
        if qrej is not None and qrej >= pf["REJECT_THR"]:
            return {"ok": False, "stage": "model", "reason_th": "ภาพไม่ชัดพอสำหรับวินิจฉัย ถ่ายใหม่",
                    "reason_en": "Image quality insufficient, retake", "metrics": m, "p_reject": round(qrej, 3)}
    return {"ok": True, "stage": "passed", "metrics": m,
            "p_reject": round(qrej, 3) if qrej is not None else None}

# ------------------------------------------------------------------ screening (ensemble + calibration)
@torch.no_grad()
def ensemble_score(x):
    ss = []
    for m in MEMBERS:
        o = m(x).float().squeeze(1)
        o = o + m(torch.flip(x, [3])).float().squeeze(1) + m(torch.flip(x, [2])).float().squeeze(1)
        ss.append(float((o / 3).item()))
    return float(np.mean(ss))

def calibrate(score):
    if CALIB is not None:
        try: return float(np.clip(CALIB.predict([score])[0], 0, 1))
        except Exception: pass
    return float(1 / (1 + np.exp(-(score - ROUNDER_COEFS[1]) * 2.2)))   # fallback sigmoid

def decide_zone(prob, grade):
    # grade-anchored เพื่อความสอดคล้อง: referable DR = grade>=2 (Moderate ขึ้นไป)
    if grade >= 2: return "refer"
    if prob >= P_HI: return "review"     # No DR/Mild แต่โมเดลเสี่ยงสูง -> ให้คนตรวจซ้ำ
    return "normal"

def grade_from_score(s):
    return int(np.clip(np.digitize([s], ROUNDER_COEFS)[0], 0, 4))

# ------------------------------------------------------------------ lesion (UNet) + landmarks
@torch.no_grad()
def predict_full(rgb):
    H, W = rgb.shape[:2]; st = PATCH // 2
    prob = np.zeros((4, H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    ys = list(range(0, max(1, H - PATCH + 1), st)); xs = list(range(0, max(1, W - PATCH + 1), st))
    if H > PATCH and ys[-1] != H - PATCH: ys.append(H - PATCH)
    if W > PATCH and xs[-1] != W - PATCH: xs.append(W - PATCH)
    for y in ys:
        for x in xs:
            p = rgb[y:y + PATCH, x:x + PATCH]
            xt = torch.from_numpy(((p / 255.0 - _MEAN) / _STD).transpose(2, 0, 1).astype(np.float32)).unsqueeze(0).to(DEVICE)
            pr = torch.sigmoid(UNET(xt))[0].float().cpu().numpy()
            prob[:, y:y + PATCH, x:x + PATCH] += pr; cnt[y:y + PATCH, x:x + PATCH] += 1
    return prob / np.maximum(cnt, 1e-6)

def lesion_features(prob, rgb=None, thr=0.5):
    C, H, W = prob.shape; feats, masks = {}, {}
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) if rgb is not None else None
    bt = float(np.percentile(gray[gray > 10], 55)) if gray is not None and (gray > 10).any() else None
    for ci, les in enumerate(LESIONS):
        mk = (prob[ci] > thr).astype(np.uint8)
        if bt is not None and les in ("EX", "SE"):       # bright lesions: ตัดบริเวณมืด (กัน fovea false-EX)
            mk = (mk & (gray > bt).astype(np.uint8)).astype(np.uint8)
        masks[les] = mk
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(mk, 8)
        comps = [int(stats[k, cv2.CC_STAT_AREA]) for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= MIN_AREA]
        area = int(sum(comps))
        feats[les] = {"count": len(comps), "area_px": area, "area_pct": round(100.0 * area / (H * W), 3)}
    return feats, masks

def quadrant_spread(mask):
    H, W = mask.shape; my, mx = H // 2, W // 2
    q = {"บนซ้าย": mask[:my, :mx], "บนขวา": mask[:my, mx:], "ล่างซ้าย": mask[my:, :mx], "ล่างขวา": mask[my:, mx:]}
    return [k for k, v in q.items() if v.sum() >= MIN_AREA]

def detect_disc(rgb):
    H, W = rgb.shape[:2]; gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    fg = gray > 10; blur = cv2.GaussianBlur(gray, (0, 0), max(3.0, W * 0.015))
    _, _, _, (cx, cy) = cv2.minMaxLoc(np.where(fg, blur, -1.0))
    r = int(0.05 * W)
    try:
        t = np.percentile(blur[fg], 99.0); bright = ((blur >= t) & fg).astype(np.uint8)
        n, lbl, stats, cent = cv2.connectedComponentsWithStats(bright, 8)
        if n > 1:
            lab = lbl[cy, cx]
            if lab == 0: lab = min(range(1, n), key=lambda i: (cent[i][0] - cx) ** 2 + (cent[i][1] - cy) ** 2)
            r = int(np.clip(np.sqrt(stats[lab, cv2.CC_STAT_AREA] / np.pi), 0.03 * W, 0.12 * W))
    except Exception: pass
    return int(cx), int(cy), int(r)

def detect_fovea(rgb, disc):
    H, W = rgb.shape[:2]; cx, cy, r = disc; DD = 2 * r
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32); fg = (gray > 10).astype(np.uint8)
    ys0, xs0 = np.where(fg > 0); gx = xs0.mean() if len(xs0) else W / 2
    sidex = 1.0 if (gx - cx) >= 0 else -1.0
    fg_e = cv2.erode(fg, np.ones((max(3, int(0.6 * DD)),) * 2, np.uint8), 1).astype(bool)
    blur = cv2.GaussianBlur(gray, (0, 0), max(3.0, DD * 0.35))
    yy, xx = np.mgrid[0:H, 0:W]; dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    band = fg_e & (dist > 1.5 * DD) & (dist < 3.5 * DD) & (np.abs(yy - cy) < 1.3 * DD) & (np.sign(xx - cx) == sidex)
    if band.sum() < 50: band = fg_e & (dist > 1.2 * DD) & (dist < 4.0 * DD) & (np.sign(xx - cx) == sidex)
    if band.sum() < 50: band = fg_e & (dist > 1.0 * DD)
    if band.sum() < 50: band = fg_e
    fy, fx = np.unravel_index(int(np.argmin(np.where(band, blur, 1e9))), (H, W))
    return int(fx), int(fy)

def fovea_metrics(masks, fovea, DD):
    fx, fy = fovea; out = {}
    for les in LESIONS:
        ys, xs = np.where(masks[les] > 0)
        if len(xs) == 0: out[les] = {"min_dd": None}; continue
        d = np.sqrt((xs - fx) ** 2 + (ys - fy) ** 2) / max(DD, 1)
        out[les] = {"min_dd": round(float(d.min()), 2)}
    return out

def build_rationale(feats, masks, fov, grade_name):
    bits = []
    if feats["MA"]["count"] > 0: bits.append(f"microaneurysm ~{feats['MA']['count']} จุด")
    he = feats["HE"]
    if he["area_px"] > 0:
        q = quadrant_spread(masks["HE"])
        bits.append(f"hemorrhage {he['count']} ก้อน กระจาย {len(q)} quadrant (~{he['area_pct']}% ของจอ)")
    ex = feats["EX"]
    if ex["area_px"] > 0:
        d = fov["EX"]["min_dd"]
        loc = (f"ใกล้ fovea ~{d} DD — ภายใน 1 DD เสี่ยง CSME" if d is not None and d <= 1.0
               else (f"ห่าง fovea ~{d} DD" if d is not None else ""))
        bits.append(f"hard exudate ~{ex['area_pct']}% {loc}".strip())
    if feats["SE"]["area_px"] > 0: bits.append("soft exudate / cotton-wool spot")
    if not bits: return "segmentation ไม่พบรอยโรคชัดเจน (รอยเล็กมากอาจตรวจไม่พบ)"
    head = f"เหตุผลประกอบ (จัดเป็น {grade_name}): " if grade_name else "รอยโรคที่ตรวจพบ: "
    return head + "; ".join(bits)

# ------------------------------------------------------------------ rendering layers
def _png_b64(img_bgr_or_bgra):
    ok, buf = cv2.imencode(".png", img_bgr_or_bgra)
    return base64.b64encode(buf).decode() if ok else None

def make_layers(raw_rgb, enh_rgb, masks, disc, fovea):
    """raw base, enhanced base, lesion layer (RGBA โปร่งใส), landmark layer (RGBA)"""
    H, W = raw_rgb.shape[:2]
    base_raw = _png_b64(cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR))
    base_enh = _png_b64(cv2.cvtColor(enh_rgb, cv2.COLOR_RGB2BGR))
    # lesion layer
    les = np.zeros((H, W, 4), np.uint8)
    for k in ["HE", "EX", "SE"]:                      # ปื้น = mask โปร่งแสง
        m = masks[k].astype(bool); c = PALETTE[k]
        les[m] = (c[0], c[1], c[2], 200)
    ys, xs = np.where(masks["MA"] > 0)                # MA = จุด marker
    for x, y in zip(xs[::3], ys[::3]):
        cv2.circle(les, (int(x), int(y)), max(2, int(0.004 * W)), (255, 0, 0, 255), -1)
    les_bgra = cv2.cvtColor(les, cv2.COLOR_RGBA2BGRA)
    # landmark layer
    lm = np.zeros((H, W, 4), np.uint8); cx, cy, r = disc; fx, fy = fovea; DD = 2 * r
    cv2.circle(lm, (cx, cy), r, (0, 255, 255, 255), 2)
    s = max(6, int(0.03 * W))
    cv2.line(lm, (fx - s, fy), (fx + s, fy), (255, 0, 255, 255), 2)
    cv2.line(lm, (fx, fy - s), (fx, fy + s), (255, 0, 255, 255), 2)
    for kk in (1, 2): cv2.circle(lm, (fx, fy), int(kk * DD), (255, 255, 255, 160), 1)
    lm_bgra = cv2.cvtColor(lm, cv2.COLOR_RGBA2BGRA)
    return {"raw": base_raw, "enhanced": base_enh,
            "lesion": _png_b64(les_bgra), "landmark": _png_b64(lm_bgra)}

def explain_lesions(bgr, grade_name):
    enh_rgb, _ = crop_resize(bgr, WORK, clahe=True)
    raw_rgb, _ = crop_resize(bgr, WORK, clahe=False)
    prob = predict_full(enh_rgb)
    feats, masks = lesion_features(prob, enh_rgb)
    disc = detect_disc(enh_rgb); fovea = detect_fovea(enh_rgb, disc); DD = 2 * disc[2]
    fov = fovea_metrics(masks, fovea, DD)
    layers = make_layers(raw_rgb, enh_rgb, masks, disc, fovea)
    rationale = build_rationale(feats, masks, fov, grade_name)
    return {"features": feats, "fovea_metrics": fov, "layers": layers,
            "disc": list(disc), "fovea": list(fovea), "rationale": rationale}

def raw_only_layer(bgr):
    return {"raw": _png_b64(cv2.cvtColor(crop_resize(bgr, WORK, clahe=False)[0], cv2.COLOR_RGB2BGR))}

# ------------------------------------------------------------------ history
def load_history(limit=200):
    if HISTORY_FILE.exists():
        try: return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))[:limit]
        except Exception: return []
    return []

def save_history(rec):
    try:
        data = load_history(); data.insert(0, rec)
        HISTORY_FILE.write_text(json.dumps(data[:200], ensure_ascii=False), encoding="utf-8")
    except Exception: pass

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
    return send_from_directory(".", f) if (HERE / f).is_file() else ("", 404)

@app.route("/status")
def status():
    return jsonify({"ok": True, "dr_models": len(MEMBERS), "quality": QMODEL is not None,
                    "lesion": UNET is not None, "calibrated": CALIB is not None,
                    "device": DEVICE, "demo": len(MEMBERS) == 0,
                    "profiles": {k: v.get("label_th", k) for k, v in QCFG["profiles"].items()}})

ZONE_TH = {"normal": "ปกติ — ไม่ต้องส่งต่อ", "review": "ไม่แน่ใจ — ให้ผู้เชี่ยวชาญตรวจซ้ำ", "refer": "พบความผิดปกติ — ส่งต่อจักษุแพทย์"}
ZONE_EN = {"normal": "Normal — no referral", "review": "Uncertain — needs human review", "refer": "Refer to ophthalmologist"}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files: return jsonify({"error": "no image"}), 400
        raw = np.frombuffer(request.files["image"].read(), np.uint8)
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if bgr is None: return jsonify({"error": "bad image"}), 400
        profile = request.form.get("mode", QCFG.get("default_profile", "fundus"))

        # --- stage 0: quality gate ---
        q = check_quality(bgr, profile)
        if not q["ok"]:
            return jsonify({"not_gradable": True, "mode": profile, "quality": q,
                            "message_th": q["reason_th"], "message_en": q["reason_en"]})

        # --- stage 1: screening (ensemble + calibration) ---
        if len(MEMBERS) == 0:
            score, demo = 1.9, True
        else:
            score = ensemble_score(to_tensor(dr_preprocess(bgr))); demo = False
        prob = calibrate(score); grade = grade_from_score(score); zone = decide_zone(prob, grade)

        res = {"demo": demo, "mode": profile, "quality": q,
               "grade": grade, "grade_name": CLASS_NAMES[grade], "score": round(score, 2),
               "risk_percent": round(prob * 100, 1), "zone": zone,
               "zone_th": ZONE_TH[zone], "zone_en": ZONE_EN[zone],
               "referable": zone == "refer"}

        # lesion explain แยกไป /explain (lazy) — คัดกรองตอบเร็ว ส่งแค่ภาพดิบมาก่อน
        res["layers"] = raw_only_layer(bgr)
        res["lesion_available"] = UNET is not None

        save_history({"time": datetime.datetime.now().isoformat(timespec="seconds"),
                      "grade": grade, "grade_name": CLASS_NAMES[grade], "zone": zone,
                      "referable": zone == "refer", "risk_percent": round(prob * 100, 1)})
        return jsonify(res)
    except Exception as e:
        traceback.print_exc(); return jsonify({"error": str(e)}), 500

@app.route("/explain", methods=["POST"])
def explain_route():
    """lazy lesion — เรียกตอนเปิด deep-dive เท่านั้น (UNet + disc/fovea + overlay layers)"""
    try:
        if "image" not in request.files: return jsonify({"error": "no image"}), 400
        raw = np.frombuffer(request.files["image"].read(), np.uint8)
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if bgr is None: return jsonify({"error": "bad image"}), 400
        if UNET is None: return jsonify({"lesion_unavailable": True})
        return jsonify(explain_lesions(bgr, request.form.get("grade_name", "")))
    except Exception as e:
        traceback.print_exc(); return jsonify({"error": str(e)}), 500

@app.route("/history")
def history(): return jsonify(load_history())

@app.route("/stats")
def stats():
    h = load_history(); n = len(h)
    dist = {c: 0 for c in CLASS_NAMES}
    for x in h: dist[x.get("grade_name", "No DR")] = dist.get(x.get("grade_name", "No DR"), 0) + 1
    ref = sum(1 for x in h if x.get("referable") or x.get("zone") == "refer")
    review = sum(1 for x in h if x.get("zone") == "review")
    return jsonify({"total": n, "referable": ref, "refer": ref, "review": review,
                    "normal": n - ref, "dist": dist})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
