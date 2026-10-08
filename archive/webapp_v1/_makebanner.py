# -*- coding: utf-8 -*-
"""Generate OG share banner (1200x630) for RetinaPlus."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 630

# diagonal brand-blue gradient
c1 = np.array([74, 166, 245])   # #4AA6F5 light
c2 = np.array([21, 101, 192])   # #1565C0 deep
yy, xx = np.mgrid[0:H, 0:W]
t = np.clip(xx / W * 0.55 + yy / H * 0.45, 0, 1)[..., None]
grad = (c1 * (1 - t) + c2 * t).astype(np.uint8)
img = Image.fromarray(grad, "RGB").convert("RGBA")
draw = ImageDraw.Draw(img)

# subtle dot texture
for gy in range(0, H, 34):
    for gx in range(0, W, 34):
        draw.ellipse([gx, gy, gx + 2, gy + 2], fill=(255, 255, 255, 22))

# white circle plate + eye icon
cx, cy, r = 300, 315, 180
draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255, 255))
icon = Image.open("icon.png").convert("RGBA")
isz = 280
icon = icon.resize((isz, isz), Image.LANCZOS)
img.paste(icon, (cx - isz // 2, cy - isz // 2 + 4), icon)

def font(path, size, fb="C:/Windows/Fonts/arialbd.ttf"):
    for p in (path, fb, "arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()

f_title = font("C:/Windows/Fonts/segoeuib.ttf", 104)
f_sub   = font("C:/Windows/Fonts/segoeui.ttf", 42, "C:/Windows/Fonts/arial.ttf")
f_tag   = font("C:/Windows/Fonts/segoeui.ttf", 31, "C:/Windows/Fonts/arial.ttf")

tx = 545
draw.text((tx, 205), "RetinaPlus", font=f_title, fill=(255, 255, 255, 255))
draw.text((tx + 3, 330), "AI Diabetic Retinopathy Screening",
          font=f_sub, fill=(233, 243, 253, 255))
# pill-ish tagline
draw.text((tx + 3, 398), "5-level severity   •   Grad-CAM explainable",
          font=f_tag, fill=(196, 223, 250, 255))

img.convert("RGB").save("banner.png", quality=92)
print("saved banner.png", img.size)
