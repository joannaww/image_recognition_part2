"""Generuje wszystkie figury raportu do img/ (liczy z cache embeddingów — szybko)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import roc_curve, auc

import evaluate as E
from config import DATA_DIR, COS_DIST_THRESHOLD, ROOT
from pipeline.matcher import load_db

IMG = ROOT / "img"
IMG.mkdir(exist_ok=True)
DIFFS = ["clean", "lowres", "multiface"]
COLORS = {"clean": "#2a9d8f", "lowres": "#e76f51", "multiface": "#264653"}

manifest = json.loads((DATA_DIR / "manifest.json").read_text())
db = load_db()
gallery = E.build_gallery(db)
rec = {c: E.compute_embeddings(manifest, c) for c in ("full", "baseline")}
rec_by = {}
for c in rec:
    d = {x: [] for x in DIFFS}
    for r in rec[c]:
        d.setdefault(r["difficulty"], []).append(r)
    rec_by[c] = d

v_clean = E.verification(rec_by["full"]["clean"], gallery, COS_DIST_THRESHOLD, True)
THR = E.threshold_at_far(v_clean["impostor"], 0.01)
print(f"próg={THR:.4f}")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("zapisano", name)


# 1. rozmiary zbiorów
from collections import Counter
cnt = Counter(m["difficulty"] for m in manifest)
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(DIFFS, [cnt[d] for d in DIFFS], color=[COLORS[d] for d in DIFFS])
for i, d in enumerate(DIFFS):
    ax.text(i, cnt[d], str(cnt[d]), ha="center", va="bottom")
ax.set_ylabel("liczba próbek"); ax.set_title("Rozmiar zbiorów testowych")
save(fig, "dataset_sizes.png")

# 2. przykłady clean/lowres/multiface
by_id = {}
for m in manifest:
    if "identity" in m:
        by_id.setdefault(m["identity"], {}).setdefault(m["difficulty"], m["path"])
person = next(i for i, d in by_id.items() if all(k in d for k in DIFFS))
fig, axs = plt.subplots(1, 3, figsize=(9, 3.2))
for ax, d in zip(axs, DIFFS):
    im = cv2.cvtColor(cv2.imread(by_id[person][d]), cv2.COLOR_BGR2RGB)
    ax.imshow(im); ax.set_title(d, fontsize=10); ax.axis("off")
save(fig, "examples.png")

# 3. ROC
fig, ax = plt.subplots(figsize=(5, 5))
for d in DIFFS:
    v = E.verification(rec_by["full"][d], gallery, THR, True)
    if not len(v["genuine"]):
        continue
    s = np.concatenate([v["genuine"], v["impostor"]])
    y = np.concatenate([np.ones(len(v["genuine"])), np.zeros(len(v["impostor"]))])
    fpr, tpr, _ = roc_curve(y, -s)
    ax.plot(fpr, tpr, color=COLORS[d], label=f"{d} (AUC={auc(fpr,tpr):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=.7)
ax.set_xlabel("FAR"); ax.set_ylabel("1 - FRR (TPR)"); ax.set_title("ROC — pełny pipeline")
ax.legend()
save(fig, "roc.png")

# 4. FAR/FRR vs próg
ths = np.linspace(0.2, 1.2, 200)
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, d in zip(axs, DIFFS):
    v = E.verification(rec_by["full"][d], gallery, THR, True)
    g, im = v["genuine"], v["impostor"]
    if not len(g):
        continue
    ax.plot(ths, [np.mean(g > t) for t in ths], label="FRR", color="#e76f51")
    ax.plot(ths, [np.mean(im <= t) for t in ths], label="FAR", color="#2a9d8f")
    ax.axvline(THR, ls="--", c="gray", lw=.8)
    ax.set_title(d); ax.set_xlabel("próg dystansu"); ax.legend()
axs[0].set_ylabel("błąd")
save(fig, "far_frr.png")

# 5. TIR
tir = {d: E.identification(rec_by["full"][d], gallery, THR, True) for d in DIFFS}
x = np.arange(len(DIFFS)); w = .35
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.bar(x - w/2, [100*tir[d]["TIR_top1"] for d in DIFFS], w, label="Top-1", color="#264653")
ax.bar(x + w/2, [100*tir[d]["TIR_top5"] for d in DIFFS], w, label="Top-5", color="#8ab17d")
ax.set_xticks(x); ax.set_xticklabels(DIFFS); ax.set_ylabel("TIR [%]"); ax.set_ylim(0, 105)
ax.set_title("True Identification Rate"); ax.legend()
save(fig, "tir.png")

# 6. ablacja
rows = []
for d in DIFFS:
    fb = E.verification(rec_by["baseline"][d], gallery, THR, False)["FRR"]
    ff = E.verification(rec_by["full"][d], gallery, THR, True)["FRR"]
    tb = E.identification(rec_by["baseline"][d], gallery, THR, False)["TIR_top1"]
    tf = tir[d]["TIR_top1"]
    rows.append((d, fb, ff, tb, tf))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8))
a1.bar(x - w/2, [100*r[1] for r in rows], w, label="baseline", color="#bbb")
a1.bar(x + w/2, [100*r[2] for r in rows], w, label="full", color="#e76f51")
a1.set_title("FRR [%]"); a1.set_xticks(x); a1.set_xticklabels(DIFFS); a1.legend()
a2.bar(x - w/2, [100*r[3] for r in rows], w, label="baseline", color="#bbb")
a2.bar(x + w/2, [100*r[4] for r in rows], w, label="full", color="#264653")
a2.set_title("TIR Top-1 [%]"); a2.set_xticks(x); a2.set_xticklabels(DIFFS)
a2.set_ylim(0, 105); a2.legend()
save(fig, "ablation.png")

# 7. histogramy dystansów genuine vs impostor
fig, axs = plt.subplots(1, 3, figsize=(13, 3.4), sharex=True)
for ax, d in zip(axs, DIFFS):
    v = E.verification(rec_by["full"][d], gallery, THR, True)
    if len(v["genuine"]):
        ax.hist(v["genuine"], bins=40, alpha=.6, color="#2a9d8f", density=True, label="genuine")
        ax.hist(v["impostor"], bins=40, alpha=.6, color="#e76f51", density=True, label="impostor")
    ax.axvline(THR, ls="--", c="gray", lw=1)
    ax.set_title(d); ax.set_xlabel("dystans kosinusowy"); ax.legend()
axs[0].set_ylabel("gęstość")
save(fig, "distances.png")

print("\nGOTOWE — wszystkie figury w img/")
