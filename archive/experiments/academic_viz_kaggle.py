# %% [markdown]
# # Academic Visualizations — Optimizer comparison + Hyperparameter heatmap
#
# notebook เล็ก แยกจาก v4 — สร้างภาพวิชาการแบบ KidneyLife+ (ใช้ B0 + subset เล็ก ~1.5h)
# ก่อนรัน: GPU T4 + Add Data: eyepacs-aptos-messidor

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=False)

import os, time, random
from collections import deque, Counter
from pathlib import Path
import numpy as np, pandas as pd, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import timm
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
assert torch.cuda.is_available(), "GPU ไม่เจอ! Settings -> Accelerator -> GPU"
DEVICE = "cuda"
OUT = Path("/kaggle/working"); OUT.mkdir(exist_ok=True)
IMG = 224; PER_CLASS = 800; EPOCHS = 3      # subset เล็ก + epoch น้อย -> เร็ว
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

# %%
# ---- discover combined + balanced subset ----
IMG_EXT = {".jpg", ".jpeg", ".png"}
GMAP = {"0":0,"no_dr":0,"nodr":0,"normal":0,"1":1,"mild":1,"2":2,"moderate":2,
        "3":3,"severe":3,"4":4,"proliferate_dr":4,"proliferative_dr":4,"proliferate":4,"proliferative":4,"pdr":4}
def nlab(s):
    k=str(s).strip().lower().replace(" ","_").replace("-","_")
    return GMAP.get(k, GMAP.get(k.replace("_",""), None))
def sdirs(p):
    try: return [d for d in os.listdir(p) if os.path.isdir(os.path.join(p,d))]
    except Exception: return []
def limgs(d):
    o=[]
    for r,_,fs in os.walk(d):
        for f in fs:
            if os.path.splitext(f)[1].lower() in IMG_EXT: o.append(os.path.join(r,f))
    return o
def discover(base):
    q=deque([(base,0)]); flat=None
    while q:
        d,dp=q.popleft()
        if dp>6: continue
        su=sdirs(d); low=[s.lower() for s in su]
        if "train" in low: return os.path.join(d, {s.lower():s for s in su}["train"])
        if sum(nlab(s) is not None for s in su)>=3 and flat is None: flat=d
        for s in su: q.append((os.path.join(d,s),dp+1))
    return flat
root = discover("/kaggle/input")
assert root, "หา combined ไม่เจอ"
rows=[]
for sub in sdirs(root):
    lab=nlab(sub)
    if lab is None: continue
    fs=limgs(os.path.join(root,sub)); random.shuffle(fs)
    for f in fs[:PER_CLASS]: rows.append((f,lab))
df=pd.DataFrame(rows,columns=["path","label"])
tr,va=train_test_split(df,test_size=0.2,stratify=df["label"],random_state=SEED)
print(f"subset: train={len(tr)} val={len(va)} dist={dict(sorted(Counter(df['label']).items()))}")

# %%
# ---- dataset (simple crop+resize) ----
_M=[0.485,0.456,0.406]; _S=[0.229,0.224,0.225]
tf_tr=T.Compose([T.RandomHorizontalFlip(),T.RandomRotation(15),T.ToTensor(),T.Normalize(_M,_S)])
tf_va=T.Compose([T.ToTensor(),T.Normalize(_M,_S)])
def prep(path):
    img=cv2.imread(str(path));
    if img is None: img=np.zeros((IMG,IMG,3),np.uint8)
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); m=g>7
    if m.sum()>0:
        c=np.argwhere(m); y0,x0=c.min(0); y1,x1=c.max(0)+1; img=img[y0:y1,x0:x1]
    img=cv2.resize(img,(IMG,IMG)); return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
class DS(Dataset):
    def __init__(s,d,tr): s.d=d.reset_index(drop=True); s.tr=tr
    def __len__(s): return len(s.d)
    def __getitem__(s,i):
        r=s.d.iloc[i]; im=Image.fromarray(prep(r["path"]))
        return (tf_tr if s.tr else tf_va)(im), int(r["label"])
def loader(d,tr): return DataLoader(DS(d,tr),batch_size=64,shuffle=tr,num_workers=os.cpu_count() or 2,pin_memory=True)
TR=loader(tr,True); VA=loader(va,False)

# %%
# ---- quick train -> คืน val_acc, val_loss, time ----
def quick_train(opt_name="adamw", lr=3e-4, wd=1e-4, epochs=EPOCHS):
    torch.manual_seed(SEED)
    m=timm.create_model("efficientnet_b0",pretrained=True,num_classes=5).to(DEVICE)
    crit=nn.CrossEntropyLoss(label_smoothing=0.1)
    opts={"adam":torch.optim.Adam,"adamw":torch.optim.AdamW,"rmsprop":torch.optim.RMSprop}
    op=opts[opt_name](m.parameters(),lr=lr,weight_decay=wd)
    sc=torch.cuda.amp.GradScaler(); t0=time.time()
    for ep in range(epochs):
        m.train()
        for x,y in TR:
            x,y=x.to(DEVICE),y.to(DEVICE); op.zero_grad()
            with torch.autocast("cuda"): loss=crit(m(x),y)
            sc.scale(loss).backward(); sc.step(op); sc.update()
    m.eval(); cor=0; n=0; vl=0.0
    with torch.no_grad():
        for x,y in VA:
            x,y=x.to(DEVICE),y.to(DEVICE)
            with torch.autocast("cuda"): o=m(x); vl+=crit(o,y).item()*x.size(0)
            cor+=(o.argmax(1)==y).sum().item(); n+=x.size(0)
    return cor/n, vl/n, time.time()-t0

# %%
# ---- (Hyperparameter heatmap: LR x weight-decay) ----
# (ตัด optimizer table ออกแล้ว — มันขัดกับที่ใช้จริง)
print("=== hyperparameter heatmap ===")
# grid รอบค่าที่โมเดลจริง (v4) ใช้: LR=2e-4, wd=1e-4 -> เป็นช่องกลาง (ตรงกับที่ใช้จริง)
LRs=[1e-4,2e-4,5e-4]; WDs=[1e-5,1e-4,1e-3]
H=np.zeros((len(LRs),len(WDs)))
for i,lr in enumerate(LRs):
    for j,wd in enumerate(WDs):
        a,_,_=quick_train(opt_name="adamw",lr=lr,wd=wd,epochs=2)
        H[i,j]=a; print(f"lr={lr} wd={wd} -> acc={a:.4f}")
fig,ax=plt.subplots(figsize=(6.5,5.5))
im=ax.imshow(H,cmap="viridis");
ax.set_xticks(range(len(WDs))); ax.set_xticklabels([f"{w:.0e}" for w in WDs])
ax.set_yticks(range(len(LRs))); ax.set_yticklabels([f"{l:.0e}" for l in LRs])
ax.set_xlabel("Weight decay"); ax.set_ylabel("Learning rate")
for i in range(len(LRs)):
    for j in range(len(WDs)):
        ax.text(j,i,f"{H[i,j]:.3f}",ha="center",va="center",
                color="white" if H[i,j]<H.max()-0.02 else "black",fontsize=10)
ax.set_title("Validation accuracy — LR × weight decay"); fig.colorbar(im,fraction=0.046)
fig.tight_layout(); fig.savefig(OUT/"hyperparam_heatmap.png",dpi=140); plt.close(fig)
print("[saved] hyperparam_heatmap.png")

# %%
# ---- (C) Contour plot (แบบ Validation Loss Contour ของ KidneyLife+) ----
fy=np.linspace(0,len(LRs)-1,60); fx=np.linspace(0,len(WDs)-1,60)
try:
    from scipy.interpolate import RectBivariateSpline
    spl=RectBivariateSpline(np.arange(len(LRs)),np.arange(len(WDs)),H,
                            kx=min(2,len(LRs)-1),ky=min(2,len(WDs)-1))
    Z=spl(fy,fx)
except Exception:
    Z=H; fx=np.arange(len(WDs)); fy=np.arange(len(LRs))
fig,ax=plt.subplots(figsize=(6.5,5.5))
cf=ax.contourf(fx,fy,Z,levels=18,cmap="viridis")
ax.contour(fx,fy,Z,levels=8,colors="white",linewidths=0.5,alpha=0.5)
ax.set_xticks(range(len(WDs))); ax.set_xticklabels([f"{w:.0e}" for w in WDs])
ax.set_yticks(range(len(LRs))); ax.set_yticklabels([f"{l:.0e}" for l in LRs])
ax.set_xlabel("Weight decay"); ax.set_ylabel("Learning rate")
ax.set_title("Validation accuracy contour — LR × weight decay")
fig.colorbar(cf,fraction=0.046); fig.tight_layout()
fig.savefig(OUT/"hyperparam_contour.png",dpi=140); plt.close(fig)
print("[saved] hyperparam_contour.png")
print("\n✅ เสร็จ — optimizer_table + hyperparam_heatmap + hyperparam_contour อยู่ใน /kaggle/working")
