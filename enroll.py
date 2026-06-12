"""Enrollment: buduje bazę użytkowników z artifacts/data_split.json.

Dla każdej wdrożonej tożsamości bierze pierwsze ENROLL_PER_USER zdjęć (CZYSTE
warunki), liczy po embeddingu na zdjęcie i zapisuje galerię. Pozostałe zdjęcia
zostają jako held-out na zbiory testowe (patrz build_test_sets.py).

Użycie:
    python enroll.py                 # enroll wszystkich z data_split.json
    python enroll.py --per-user 3
"""
import argparse
import json

from config import SPLIT_PATH, DB_PATH
from pipeline import detect_align, embed
from pipeline.matcher import add_user, save_db

ENROLL_PER_USER = 3


def enroll(per_user=ENROLL_PER_USER):
    split = json.loads(SPLIT_PATH.read_text())
    enrolled = split["enrolled"]
    db = {}
    used_for_enroll = {}

    for i, (user, imgs) in enumerate(enrolled.items(), 1):
        gallery = imgs[:per_user]
        embs = []
        for path in gallery:
            try:
                img = detect_align.imread(path)
            except FileNotFoundError:
                continue
            aligned = detect_align.align_best(img)
            embs.append(embed.embed_aligned(aligned, tta=True))
        if embs:
            add_user(db, user, embs, meta={"n_enroll": len(embs)})
            used_for_enroll[user] = gallery
        if i % 25 == 0:
            print(f"  ...{i}/{len(enrolled)} osób")

    save_db(db)
    (DB_PATH.parent / "enroll_used.json").write_text(json.dumps(used_for_enroll, indent=2))
    print(f"Wdrożono {len(db)} użytkowników -> {DB_PATH}")
    print(f"Zapisano listę zdjęć użytych do enrollmentu -> enroll_used.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-user", type=int, default=ENROLL_PER_USER)
    args = ap.parse_args()
    enroll(args.per_user)
