import numpy as np
from PIL import Image

im = Image.open("logo.png").convert("RGBA")
arr = np.array(im)
a = arr[:, :, 3]
cols = (a > 16).sum(axis=0)
nz = np.where(cols > 0)[0]
start, end = int(nz.min()), int(nz.max())
print("content x:", start, end)

# find first big gap after the eye blob (eye -> text)
gap_start = None
run = 0
for x in range(start, a.shape[1]):
    if cols[x] == 0:
        run += 1
        if run >= 40 and gap_start is None:
            gap_start = x - run + 1
            break
    else:
        run = 0
print("eye ends ~", gap_start)

x0 = start
x1 = gap_start if gap_start else int(start + (end - start) * 0.4)

# crop the eye columns, then trim rows by alpha
eye = arr[:, x0:x1, :]
ea = eye[:, :, 3]
rows = np.where((ea > 16).sum(axis=1) > 0)[0]
y0, y1 = int(rows.min()), int(rows.max())
eye = eye[y0:y1 + 1, :, :]
print("eye crop size:", eye.shape[1], eye.shape[0])

# pad to square (transparent) so it scales cleanly everywhere
h, w = eye.shape[0], eye.shape[1]
s = max(h, w)
pad = int(s * 0.06)  # small breathing room
canvas = np.zeros((s + 2 * pad, s + 2 * pad, 4), dtype=np.uint8)
oy = (canvas.shape[0] - h) // 2
ox = (canvas.shape[1] - w) // 2
canvas[oy:oy + h, ox:ox + w, :] = eye
out = Image.fromarray(canvas, "RGBA")
out.save("icon.png")
print("saved icon.png", out.size)
