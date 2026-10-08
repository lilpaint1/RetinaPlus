# ============================================================
# 8b. EXTERNAL TEST — IDRiD
# วางเป็น cell ใหม่ "ท้ายสุด" แล้วรัน "หลังเทรนจบ" (cell 8 รันแล้ว)
# ใช้ model + THR ที่มีอยู่ในหน่วยความจำ -> ไม่ต้องเทรนใหม่ (ไม่กี่นาที)
# ============================================================
import glob

# 1) หา IDRiD csv (path ที่มีคำว่า idrid)
cands = [p for p in glob.glob("/kaggle/input/**/*.csv", recursive=True) if "idrid" in p.lower()]
assert cands, "ไม่เจอ csv ของ IDRiD — เช็คว่า Add Data 'IDRiD' แล้วหรือยัง"
csv_path = cands[0]
idrid_root = os.path.dirname(csv_path)
print("[IDRiD] csv:", csv_path)

# 2) index รูป "เฉพาะใน IDRiD" (ไม่แตะ combined) — match ด้วยชื่อไฟล์ไม่รวมนามสกุล
img_idx = {}
for p in glob.glob(os.path.join(idrid_root, "**", "*.*"), recursive=True):
    if os.path.splitext(p)[1].lower() in IMG_EXT:
        img_idx[os.path.splitext(os.path.basename(p))[0].lower()] = p
print("[IDRiD] รูปที่ index ได้:", len(img_idx))

# 3) อ่าน csv -> df_idrid
t = pd.read_csv(csv_path)
t.columns = [str(c).strip() for c in t.columns]
col_img = "id_code" if "id_code" in t.columns else t.columns[0]
col_grd = "diagnosis" if "diagnosis" in t.columns else t.columns[1]
print(f"[IDRiD] ใช้ column: image='{col_img}' grade='{col_grd}'")

rows = []
for _, r in t.iterrows():
    key = str(r[col_img]).strip().lower()
    p = img_idx.get(key) or img_idx.get(os.path.splitext(key)[0])
    if p is None:
        continue
    try:
        lab = int(r[col_grd])
    except Exception:
        continue
    if 0 <= lab <= 4:
        rows.append((p, lab))
df_idrid = pd.DataFrame(rows, columns=["path", "label"]).drop_duplicates("path")
print(f"[IDRiD] จับคู่รูป+label ได้: {len(df_idrid)}  dist={dict(sorted(Counter(df_idrid['label']).items()))}")
assert len(df_idrid) > 0, "จับคู่ไม่ได้เลย — เช็คว่า id_code ตรงกับชื่อไฟล์รูปไหม"

# 4) ประเมินใหม่ทั้ง internal + external -> ได้ ROC รวมสวยๆ (ใช้ model+THR เดิม ไม่เทรนใหม่)
all_res = globals().get("all_res", {})
roc_data = {}
if df_test is not None:
    all_res["internal_test"], roc_data["internal_test (optimistic)"] = evaluate("internal_test", df_test)
all_res["external_idrid"], roc_data["external IDRiD (trustworthy)"] = evaluate("external_idrid", df_idrid)

# 5) วาด ROC รวม + เซฟ metrics.json ใหม่
fig, ax = plt.subplots(figsize=(6.5, 6))
for label, (fpr, tpr, auc) in roc_data.items():
    ax.plot(fpr, tpr, lw=2, label=f"{label}  AUC={auc:.3f}")
ax.plot([0, 1], [0, 1], "--", color="gray")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC — Referable DR (grade ≥ 2)"); ax.legend(loc="lower right"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "roc_curve.png", dpi=130); plt.close(fig)

json.dump(all_res, open(OUT / "metrics.json", "w"), ensure_ascii=False, indent=2)
print("\n✅ เติม external IDRiD ครบแล้ว -> metrics.json, roc_curve.png, confusion_external_idrid.png อัปเดตแล้ว")
