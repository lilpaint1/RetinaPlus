# -*- coding: utf-8 -*-
"""แยกภาพ DDR เป็นโฟลเดอร์ตาม grade (สำหรับเทสต์ predict)"""
import os, shutil, collections
import pandas as pd

CSV     = r"C:\Users\Acer\Downloads\archive (1)\DR_grading.csv"
IMG_DIR = r"C:\Users\Acer\Downloads\archive (1)\DR_grading\DR_grading"
OUT     = r"C:\Users\Acer\Downloads\DDR_test_byGrade"
PER_CLASS = 50          # จำนวนต่อ grade — ตั้ง None = เอาทั้งหมด

NAMES = {0: "0_NoDR", 1: "1_Mild", 2: "2_Moderate",
         3: "3_Severe", 4: "4_Proliferative", 5: "5_Ungradable"}

df = pd.read_csv(CSV)
name_col, lab_col = df.columns[0], df.columns[-1]
print(f"CSV: {len(df)} แถว · คอลัมน์ {list(df.columns)}")

copied, missing = collections.Counter(), 0
for _, row in df.iterrows():
    lab = int(row[lab_col])
    if PER_CLASS is not None and copied[lab] >= PER_CLASS:
        continue
    name = str(row[name_col]).strip()
    src = os.path.join(IMG_DIR, name)
    if not os.path.exists(src):
        missing += 1
        continue
    folder = os.path.join(OUT, NAMES.get(lab, f"{lab}_other"))
    os.makedirs(folder, exist_ok=True)
    shutil.copy2(src, os.path.join(folder, name))
    copied[lab] += 1

print(f"\nคัดลอกเสร็จ -> {OUT}")
for k in sorted(copied):
    print(f"  {NAMES.get(k, k):<16}: {copied[k]} ภาพ")
print(f"  (หาไฟล์ไม่เจอ {missing})")
