"""แปลง dr_screening_kaggle.py (มี cell markers) -> dr_screening_kaggle.ipynb"""
import json, sys, os

_name = sys.argv[1] if len(sys.argv) > 1 else "dr_screening_kaggle.py"
src = os.path.join(os.path.dirname(__file__), _name)
dst = src[:-3] + ".ipynb"

lines = open(src, encoding="utf-8").read().split("\n")
cells, cur, ctype = [], [], None

def flush():
    if ctype is None:
        return
    body = "\n".join(cur).strip("\n")
    if ctype == "markdown":
        md = []
        for l in body.split("\n"):
            if l.startswith("# "):   md.append(l[2:])
            elif l == "#":           md.append("")
            elif l.startswith("#"):  md.append(l[1:])
            else:                    md.append(l)
        text = "\n".join(md)
        cells.append({"cell_type": "markdown", "metadata": {},
                      "source": [s + "\n" for s in text.split("\n")][:-1] + [text.split("\n")[-1]]})
    else:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": [s + "\n" for s in body.split("\n")][:-1] + [body.split("\n")[-1]]})

for l in lines:
    if l.strip() == "# %% [markdown]":
        flush(); cur, ctype = [], "markdown"
    elif l.strip() == "# %%":
        flush(); cur, ctype = [], "code"
    else:
        cur.append(l)
flush()

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"wrote {dst} ({len(cells)} cells)")
