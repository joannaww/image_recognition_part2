"""Generuje zbiory testowe + manifesty JSON.

  clean      — held-out CZYSTE zdjęcia wdrożonych osób + zdjęcia impostorów (probe)
  lowres(#3) — syntetyczna degradacja CCTV tych samych czystych probe (SPAROWANE)
  multiface(#1) — kompozyty: twarz osoby (centralna, duża) + dystraktory na tle

Manifest (lista rekordów):
  {"path", "identity"|"impostor", "difficulty": clean|lowres|multiface,
   "source_clean": <ścieżka czystego oryginału lub null>, "severity": ...}

Parowanie genuine/impostor i metryki liczone są w evaluate.py / notebookach —
manifesty są opisane tożsamością, więc pary da się odtworzyć deterministycznie.

Użycie:
    python build_test_sets.py --lowres-per 6 --multiface-per 6
"""
import argparse
import json
import random

import cv2
import numpy as np

from config import (SPLIT_PATH, DB_PATH, LOWRES_DIR, MULTIFACE_DIR,
                    BACKGROUNDS_DIR, DATA_DIR, SEED)
from pipeline import detect_align


# ---------------------------------------------------------------- degradacja
def degrade_cctv(bgr, rng, out_size=112):
    """Symuluje kadr CCTV: downscale -> blur -> szum -> JPEG -> upscale."""
    h, w = bgr.shape[:2]
    # realistyczny poziom CCTV/TinyFace: twarz ~20-40 px (nie ekstremalne 12 px)
    small = rng.randint(20, 40)                       # docelowa "rozdzielczość twarzy"
    scale = small / max(h, w)
    ds = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5:
        ds = cv2.GaussianBlur(ds, (3, 3), 0)
    if rng.random() < 0.5:
        noise = rng.normalvariate
        n = np.array([[ [noise(0, rng.uniform(2, 8)) for _ in range(3)]
                        for _ in range(ds.shape[1])] for _ in range(ds.shape[0])], np.float32)
        ds = np.clip(ds.astype(np.float32) + n, 0, 255).astype(np.uint8)
    q = rng.randint(35, 65)
    ok, enc = cv2.imencode(".jpg", ds, [cv2.IMWRITE_JPEG_QUALITY, q])
    if ok:
        ds = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    up = cv2.resize(ds, (out_size, out_size), interpolation=cv2.INTER_CUBIC)
    return up, {"face_px": small, "jpeg_q": q}


# --------------------------------------------------------------- kompozycja
def make_background(rng, size=512):
    """Syntetyczne tło (gradient + szum), gdy brak realnych w BACKGROUNDS_DIR."""
    bgs = [p for p in BACKGROUNDS_DIR.iterdir() if p.suffix.lower() in
           {".jpg", ".jpeg", ".png"}] if BACKGROUNDS_DIR.exists() else []
    if bgs:
        img = cv2.imread(str(rng.choice(bgs)))
        return cv2.resize(img, (size, size))
    c1 = np.array([rng.randint(40, 220) for _ in range(3)], np.float32)
    c2 = np.array([rng.randint(40, 220) for _ in range(3)], np.float32)
    ramp = np.linspace(0, 1, size, dtype=np.float32)[:, None, None]
    img = (c1 * (1 - ramp) + c2 * ramp)
    img = np.repeat(img, size, axis=1)
    img += np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, 8, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _pad_replicate(bgr, frac=0.35):
    """Dokłada margines (replikacja brzegu), żeby wklejona twarz miała kontekst
    i była wykrywalna przez detektor w scenie (FaceCard to ciasne cropy)."""
    h, w = bgr.shape[:2]
    m = int(frac * max(h, w))
    return cv2.copyMakeBorder(bgr, m, m, m, m, cv2.BORDER_REPLICATE)


def crop_face_region(bgr, margin=0.4):
    """Obszar twarzy z marginesem. Fallback: całe zdjęcie (gdy detektor nic nie
    znajdzie — częste dla ciasnych cropów FaceCard)."""
    faces = detect_align.detect_faces(bgr)
    if not faces:
        h, w = bgr.shape[:2]
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        return bgr[y0:y0 + s, x0:x0 + s]
    f = max(faces, key=lambda x: x.det_score)
    x1, y1, x2, y2 = f.bbox
    w, h = x2 - x1, y2 - y1
    x1 = int(max(0, x1 - margin * w)); y1 = int(max(0, y1 - margin * h))
    x2 = int(min(bgr.shape[1], x2 + margin * w)); y2 = int(min(bgr.shape[0], y2 + margin * h))
    crop = bgr[y1:y2, x1:x2]
    return crop if crop.size else None


def paste(scene, patch, cx, cy, target_w):
    """Wkleja patch wyśrodkowany w (cx, cy), przeskalowany do szerokości target_w."""
    ph, pw = patch.shape[:2]
    scale = target_w / pw
    patch = cv2.resize(patch, (int(pw * scale), int(ph * scale)))
    ph, pw = patch.shape[:2]
    x0, y0 = int(cx - pw / 2), int(cy - ph / 2)
    x1, y1 = x0 + pw, y0 + ph
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(scene.shape[1], x1), min(scene.shape[0], y1)
    if sx1 <= sx0 or sy1 <= sy0:
        return
    px0, py0 = sx0 - x0, sy0 - y0
    scene[sy0:sy1, sx0:sx1] = patch[py0:py0 + (sy1 - sy0), px0:px0 + (sx1 - sx0)]


def build_multiface_scene(subject_bgr, distractor_bgrs, rng, size=512):
    """Subiekt: duży, centralny. Dystraktory: mniejsze, ale w całości w kadrze
    i na tyle duże, by detektor je widział (inaczej selekcja #1 nie ma wyboru)."""
    scene = make_background(rng, size)
    subj = crop_face_region(subject_bgr)
    if subj is None:
        return None
    paste(scene, _pad_replicate(subj), size // 2, int(size * 0.58), target_w=int(size * 0.46))
    # sloty rozmieszczone tak, by cały patch mieścił się w kadrze
    slots = [(0.20, 0.22), (0.80, 0.22), (0.20, 0.78), (0.80, 0.78)]
    rng.shuffle(slots)
    for d, (fx, fy) in zip(distractor_bgrs, slots):
        patch = crop_face_region(d)
        if patch is None:
            continue
        patch = _pad_replicate(patch)
        tw = int(size * rng.uniform(0.30, 0.36))
        cx = int(min(max(fx * size, tw / 2), size - tw / 2))
        cy = int(min(max(fy * size, tw / 2), size - tw / 2))
        paste(scene, patch, cx, cy, target_w=tw)
    return scene


# ------------------------------------------------------------------- budowa
def held_out_images(split):
    """Czyste zdjęcia wdrożonych osób NIE użyte do enrollmentu."""
    used = json.loads((DB_PATH.parent / "enroll_used.json").read_text())
    out = {}
    for user, imgs in split["enrolled"].items():
        held = [p for p in imgs if p not in set(used.get(user, []))]
        if held:
            out[user] = held
    return out


def main(lowres_per, multiface_per, max_clean_per_user):
    rng = random.Random(SEED)
    split = json.loads(SPLIT_PATH.read_text())
    held = held_out_images(split)
    impostors = split["impostors"]
    distractor_ids = list(split["distractors"].values())

    manifest = []

    # --- CLEAN: held-out wdrożonych + impostorzy ---
    for user, imgs in held.items():
        for p in imgs[:max_clean_per_user]:
            manifest.append({"path": p, "identity": user, "difficulty": "clean",
                             "source_clean": p})
    for user, imgs in impostors.items():
        manifest.append({"path": imgs[0], "impostor": user, "difficulty": "clean",
                         "source_clean": imgs[0]})

    clean_probes = [m for m in manifest if m["difficulty"] == "clean" and "identity" in m]
    print(f"clean: {len([m for m in manifest if m['difficulty']=='clean'])} próbek")

    # --- LOWRES: degradacja czystych probe (sparowane) ---
    n = 0
    for m in clean_probes:
        src = detect_align.imread(m["source_clean"])
        for _ in range(lowres_per):
            deg, meta = degrade_cctv(src, rng)
            name = f"{m['identity'].replace(':', '_')}_{n:06d}.png"
            path = str(LOWRES_DIR / name)
            cv2.imwrite(path, deg)
            manifest.append({"path": path, "identity": m["identity"],
                             "difficulty": "lowres", "source_clean": m["source_clean"],
                             **meta})
            n += 1
    print(f"lowres: {n} próbek")

    # --- MULTIFACE: kompozyty (subiekt + dystraktory) ---
    n = 0
    for m in clean_probes:
        subj = detect_align.imread(m["source_clean"])
        for _ in range(multiface_per):
            d_imgs = [detect_align.imread(rng.choice(d_list))
                      for d_list in random.Random(rng.random()).sample(
                          distractor_ids, rng.randint(1, 3))]
            scene = build_multiface_scene(subj, d_imgs, rng)
            if scene is None:
                continue
            name = f"{m['identity'].replace(':', '_')}_{n:06d}.png"
            path = str(MULTIFACE_DIR / name)
            cv2.imwrite(path, scene)
            manifest.append({"path": path, "identity": m["identity"],
                             "difficulty": "multiface", "source_clean": m["source_clean"],
                             "n_distractors": len(d_imgs)})
            n += 1
    print(f"multiface: {n} próbek")

    out = DATA_DIR / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    by_diff = {}
    for m in manifest:
        by_diff[m["difficulty"]] = by_diff.get(m["difficulty"], 0) + 1
    print(f"Zapisano {out}  ->  {by_diff}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lowres-per", type=int, default=6)
    ap.add_argument("--multiface-per", type=int, default=6)
    ap.add_argument("--max-clean-per-user", type=int, default=6)
    args = ap.parse_args()
    main(args.lowres_per, args.multiface_per, args.max_clean_per_user)
