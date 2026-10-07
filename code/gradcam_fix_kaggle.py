# %% [markdown]
# # Grad-CAM Fix — ensemble + TTA + เลือกรูปคุณภาพดี (ไม่ต้องเทรนใหม่)
#
# โหลดโมเดล v3+v4 ที่เซฟไว้ -> ทำ Grad-CAM ใหม่ให้ "ตรงกับผลจริง" (ensemble) + เลี่ยงรูปเบลอ
#
# Add Data: (1) combined  (2) IDRiD  (3) output v3 (best.pt)  (4) output v4 (best.pt)
# GPU T4 -> Run All (~10-15 นาที)

# %%
import subprocess, sys
for p in ["grad-cam", "timm", "albumentations==1.3.1"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", p], check=False)

import os, glob, math, random, warnings
from collections import deque, Counter
from pathlib import Path
import numpy as np, pandas as pd, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torch, torch.nn as nn, timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score
warnings.filterwarnings("ignore"); cv2.setNumThreads(0)

SEED = 2024; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
assert torch.cuda.is_available(), "เปิด GPU ก่อน"
DEVICE = "cuda"; OUT = Path("/kaggle/working"); OUT.mkdir(exist_ok=True)

# ---- config (ต้องตรงกับ v4) ----
BACKBONE="efficientnet_b4"; IMG_SIZE=380; DROP_RATE=0.3; DROP_PATH=0.2; NUM_CLASSES=5; BATCH=16
CLASS_NAMES=["No DR","Mild","Moderate","Severe","Proliferative"]
_MEAN=(0.485,0.456,0.406); _STD=(0.229,0.224,0.225)

# %%
# ---- discovery (combined val + IDRiD) ----
IMG_EXT={".jpg",".jpeg",".png"}
GMAP={"0":0,"no_dr":0,"nodr":0,"normal":0,"1":1,"mild":1,"2":2,"moderate":2,"3":3,"severe":3,
      "4":4,"proliferate_dr":4,"proliferative_dr":4,"proliferate":4,"proliferative":4,"pdr":4}
def nlab(s):
    k=str(s).strip().lower().replace(" ","_").replace("-","_"); return GMAP.get(k,GMAP.get(k.replace("_",""),None))
def sdirs(p):
    try: return [d for d in os.listdir(p) if os.path.isdir(os.path.join(p,d))]
    except: return []
def limgs(d):
    o=[]
    for r,_,fs in os.walk(d):
        for f in fs:
            if os.path.splitext(f)[1].lower() in IMG_EXT: o.append(os.path.join(r,f))
    return o
def find_dir(root,*names):
    m={s.lower():s for s in sdirs(root)}
    for n in names:
        if n in m: return os.path.join(root,m[n])
def collect(split):
    rows=[]
    for sub in sdirs(split):
        lab=nlab(sub)
        if lab is None: continue
        for f in limgs(os.path.join(split,sub)): rows.append((f,lab))
    return rows
def discover_combined(base):
    q=deque([(base,0)])
    while q:
        d,dp=q.popleft()
        if dp>6: continue
        su=sdirs(d); low=[s.lower() for s in su]
        if "train" in low and any(x in low for x in ("val","valid","validation")): return d
        for s in su: q.append((os.path.join(d,s),dp+1))
def discover_idrid(base):
    cs=[os.path.join(r,f) for r,_,fs in os.walk(base) if "idrid" in r.lower() for f in fs if f.lower().endswith(".csv")]
    if not cs: return None
    root=os.path.dirname(cs[0]); idx={os.path.splitext(os.path.basename(p))[0].lower():p for p in limgs(root)}
    return cs[0],idx

root=discover_combined("/kaggle/input"); assert root,"หา combined ไม่เจอ"
d_va=find_dir(root,"val","valid","validation")
df_val=pd.DataFrame(collect(d_va),columns=["path","label"])
print(f"[val] {len(df_val)} dist={dict(sorted(Counter(df_val['label']).items()))}")

df_idrid=None; idr=discover_idrid("/kaggle/input")
if idr:
    csv,idx=idr; t=pd.read_csv(csv); t.columns=[str(c).strip() for c in t.columns]
    ci="id_code" if "id_code" in t.columns else t.columns[0]
    cg="diagnosis" if "diagnosis" in t.columns else t.columns[1]
    rows=[]
    for _,r in t.iterrows():
        p=idx.get(str(r[ci]).strip().lower())
        if p is None: continue
        try: lab=int(r[cg])
        except: continue
        if 0<=lab<=4: rows.append((p,lab))
    df_idrid=pd.DataFrame(rows,columns=["path","label"]).drop_duplicates("path")
    print(f"[IDRiD] {len(df_idrid)}")

# %%
# ---- preprocess + model + load ensemble ----
def _crop(img,tol=7):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); m=g>tol
    if m.sum()==0: return img
    c=np.argwhere(m); y0,x0=c.min(0); y1,x1=c.max(0)+1; return img[y0:y1,x0:x1]
def _clahe(img):
    lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB); l,a,b=cv2.split(lab)
    l=cv2.createCLAHE(2.0,(8,8)).apply(l); return cv2.cvtColor(cv2.merge((l,a,b)),cv2.COLOR_LAB2BGR)
def preprocess_image(path):
    img=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(path)
    img=_crop(img)
    if img.size==0: img=cv2.imread(str(path),cv2.IMREAD_COLOR)
    img=cv2.resize(img,(IMG_SIZE,IMG_SIZE),interpolation=cv2.INTER_AREA); img=_clahe(img)
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
eval_aug=A.Compose([A.Normalize(_MEAN,_STD),ToTensorV2()])

def build_model():
    return timm.create_model(BACKBONE,pretrained=False,num_classes=1,drop_rate=DROP_RATE,drop_path_rate=DROP_PATH).to(DEVICE)

MEMBERS=[]
for cp in sorted(glob.glob("/kaggle/input/**/best*.pt",recursive=True)):
    try:
        m=build_model(); m.load_state_dict(torch.load(cp,map_location=DEVICE)); m.eval()
        MEMBERS.append(m); print(f"[ensemble] + {cp}")
    except Exception as e: print(f"ข้าม {cp} ({type(e).__name__})")
assert MEMBERS, "ไม่เจอ best.pt — Add Data output v3/v4 (best.pt) ด้วย"
print(f"[ensemble] {len(MEMBERS)} โมเดล")

@torch.no_grad()
def ens_score(rgb):
    x=eval_aug(image=rgb)["image"].unsqueeze(0).to(DEVICE); ss=[]
    for m in MEMBERS:
        with torch.autocast("cuda"):
            o=m(x).float().squeeze(1)
            o=o+m(torch.flip(x,[3])).float().squeeze(1)+m(torch.flip(x,[2])).float().squeeze(1)
        ss.append(float((o/3).item()))
    return float(np.mean(ss))

# %%
# ---- fit OptimizedRounder บน val (ensemble+TTA) ----
from tqdm.auto import tqdm
def qwk(a,b): return cohen_kappa_score(a,b,weights="quadratic")
class Rounder:
    def __init__(s): s.c=[0.5,1.5,2.5,3.5]
    def fit(s,X,y):
        r=minimize(lambda c:-qwk(y,np.digitize(X,sorted(c))),s.c,method="Nelder-Mead"); s.c=sorted(r.x)
    def predict(s,X): return np.clip(np.digitize(X,s.c),0,4).astype(int)
# subsample val ~1500 รูป (พอสำหรับ fit rounder, เร็วขึ้น ~9 เท่า)
df_vs=df_val.groupby("label",group_keys=False).apply(lambda g:g.sample(min(len(g),300),random_state=SEED))
vs=np.array([ens_score(preprocess_image(p)) for p in tqdm(df_vs["path"],desc="val(1500)")])
ROUNDER=Rounder(); ROUNDER.fit(vs,df_vs["label"].values)
print(f"[rounder] {[round(c,2) for c in ROUNDER.c]}")

# %%
# ---- Grad-CAM (heatmap จากโมเดลตัวแรก) + เลือกรูปคม + ทายด้วย ensemble ----
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
class RegTarget:
    def __call__(s,o): return o[0]
m0=MEMBERS[0]
# layer ตื้นขึ้น (blocks[-2]) -> spatial ละเอียดกว่า conv_head -> localize รอยโรคดีขึ้น
tl = m0.blocks[-2] if (hasattr(m0,"blocks") and len(m0.blocks)>=2) else (getattr(m0,"conv_head",m0.blocks[-1]))
cam=GradCAMPlusPlus(model=m0,target_layers=[tl])   # GradCAM++ คมกว่า GradCAM
def sharp(path):
    g=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2GRAY); return cv2.Laplacian(g,cv2.CV_64F).var()
def fundus_mask(path):   # mask เฉพาะวงจอตา ตัดมุม/พื้นหลังออก
    g=cv2.cvtColor(cv2.resize(cv2.imread(str(path)),(IMG_SIZE,IMG_SIZE)),cv2.COLOR_BGR2GRAY)
    m=(g>15).astype(np.float32)
    return cv2.erode(m,np.ones((25,25),np.uint8))

src=df_idrid if df_idrid is not None else df_val
samples=[]
for g in range(5):
    sub=src[src["label"]==g]
    if not len(sub): continue
    sub=sub.sample(min(len(sub),30),random_state=SEED).copy()
    sub["s"]=sub["path"].map(sharp)
    cand=sub.sort_values("s",ascending=False).head(8)   # คมสุด 8 รูป
    best=None; bd=9
    for p in cand["path"]:                               # เลือกอันที่ ensemble ทายใกล้ true สุด
        d=abs(ens_score(preprocess_image(p))-g)
        if d<bd: bd=d; best=p
    samples.append((best,g))

n=len(samples); fig,axs=plt.subplots(n,3,figsize=(10,3.2*n))
if n==1: axs=axs[None,:]
for r,(path,g) in enumerate(samples):
    rgb=preprocess_image(path); rgbf=rgb.astype(np.float32)/255.0
    x=eval_aug(image=rgb)["image"].unsqueeze(0).to(DEVICE)
    cont=ens_score(rgb); pg=int(ROUNDER.predict(np.array([cont]))[0])  # ทายด้วย ensemble
    gray=cam(input_tensor=x,targets=[RegTarget()])[0]
    gray=gray*fundus_mask(path)                      # ตัดมุม/พื้นหลัง
    if gray.max()>1e-6: gray=gray/gray.max()         # re-normalize หลัง mask
    ov=show_cam_on_image(rgbf,gray,use_rgb=True)
    orig=cv2.cvtColor(cv2.resize(cv2.imread(str(path)),(IMG_SIZE,IMG_SIZE)),cv2.COLOR_BGR2RGB)
    axs[r,0].imshow(orig); axs[r,0].set_ylabel(f"True: {CLASS_NAMES[g]}",fontsize=11)
    axs[r,1].imshow(rgb); axs[r,2].imshow(ov)
    if r==0:
        axs[r,0].set_title("Original"); axs[r,1].set_title("Preprocessed"); axs[r,2].set_title("Grad-CAM")
    axs[r,2].set_xlabel(f"Pred: {CLASS_NAMES[pg]} (score={cont:.2f})")
    for c in range(3): axs[r,c].set_xticks([]); axs[r,c].set_yticks([])
fig.suptitle("Grad-CAM — Eye Comparison (v4 ensemble + TTA)",fontsize=14)
fig.tight_layout(); fig.savefig(OUT/"gradcam_eyes_fixed.png",dpi=130); plt.close(fig)
print("✅ เสร็จ -> gradcam_eyes_fixed.png (ensemble+TTA, รูปคม, ทายตรงกับ QWK)")
