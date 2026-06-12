"""Baza enrollmentu (multi-embedding na osobę) + weryfikacja i identyfikacja.

Rekord użytkownika: {"embeddings": [np.float32(512), ...], "meta": {...}}.
Dopasowanie: dystans kosinusowy do NAJBLIŻSZEGO embeddingu w galerii osoby
(max-similarity) — odporniejsze niż pojedynczy centroid.
"""
import pickle

import numpy as np

from config import DB_PATH, COS_DIST_THRESHOLD


def load_db(path=DB_PATH):
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def save_db(db, path=DB_PATH):
    with open(path, "wb") as f:
        pickle.dump(db, f)


def add_user(db, user, embeddings, meta=None):
    db[user] = {"embeddings": [np.asarray(e, np.float32) for e in embeddings],
                "meta": meta or {}}


def _dist_to_user(emb, record):
    sims = [float(np.dot(emb, g)) for g in record["embeddings"]]
    return 1.0 - max(sims)   # dystans do najbliższego wzorca


def verify(db, user, emb, threshold=COS_DIST_THRESHOLD):
    """Weryfikacja 1:1. Zwraca (accepted, distance)."""
    if user not in db:
        return False, 1.0
    d = _dist_to_user(emb, db[user])
    return d <= threshold, d


def identify(db, emb, threshold=COS_DIST_THRESHOLD, top_k=5):
    """Identyfikacja 1:N (open-set). Zwraca (pred_user|None, ranking).

    ranking: lista (user, distance) posortowana rosnąco. pred_user=None gdy
    najbliższy dystans > threshold (odrzucenie).
    """
    dists = [(user, _dist_to_user(emb, rec)) for user, rec in db.items()]
    dists.sort(key=lambda t: t[1])
    ranking = dists[:top_k]
    if not dists:
        return None, ranking
    best_user, best_d = dists[0]
    pred = best_user if best_d <= threshold else None
    return pred, ranking
