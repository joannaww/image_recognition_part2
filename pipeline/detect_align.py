"""Detekcja i align twarzy (InsightFace buffalo_l + norm_crop).

Zwraca WSZYSTKIE twarze (potrzebne do utrudnienia #1 — wiele twarzy), a nie tylko
najlepszą. Detektor buffalo_l pobiera się automatycznie przy pierwszym użyciu (CPU).
"""
import cv2
import numpy as np

from config import IMG_SIZE

_face_app = None


def get_face_app():
    """Leniwa, pojedyncza instancja FaceAnalysis (CPU)."""
    global _face_app
    if _face_app is None:
        from insightface.app import FaceAnalysis
        _face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _face_app.prepare(ctx_id=-1, det_size=(640, 640))
    return _face_app


def detect_faces(img_bgr):
    """Lista obiektów twarzy InsightFace (każdy ma .bbox, .kps, .det_score)."""
    return get_face_app().get(img_bgr)


def pad_replicate(img_bgr, frac=0.4):
    """Margines (replikacja brzegu) — daje detektorowi kontekst dla ciasnych cropów."""
    m = int(frac * max(img_bgr.shape[:2]))
    return cv2.copyMakeBorder(img_bgr, m, m, m, m, cv2.BORDER_REPLICATE)


def detect_faces_robust(img_bgr):
    """Detekcja z fallbackiem na padding (ciasne cropy: FaceCard, kadry CCTV).

    Zwraca (faces, work_img): obraz, na którym twarze zostały wykryte (raw albo
    paddowany) — używaj go do dalszego align/crop, bo .bbox/.kps są w jego układzie.
    """
    faces = detect_faces(img_bgr)
    if faces:
        return faces, img_bgr
    padded = pad_replicate(img_bgr)
    return detect_faces(padded), padded


def align_from_kps(img_bgr, kps, image_size=IMG_SIZE):
    """Align na podstawie 5 punktów -> crop 112x112 RGB uint8."""
    from insightface.utils.face_align import norm_crop
    crop = norm_crop(img_bgr, landmark=kps, image_size=image_size)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def fallback_crop(img_bgr, image_size=IMG_SIZE):
    """Center-crop, gdy detektor nic nie zwróci."""
    h, w = img_bgr.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    crop = img_bgr[y0:y0 + s, x0:x0 + s]
    crop = cv2.resize(crop, (image_size, image_size))
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def align_best(img_bgr, image_size=IMG_SIZE):
    """Najlepsza (najwyższy det_score) twarz lub fallback. RGB uint8 112x112.

    Używa robust detekcji (padding) — kluczowe dla ciasnych cropów FaceCard i CCTV,
    gdzie detekcja na surowym obrazie zawodzi i bez tego leciałby center-crop bez alignu.
    """
    faces, work = detect_faces_robust(img_bgr)
    if faces:
        face = max(faces, key=lambda f: f.det_score)
        return align_from_kps(work, face.kps, image_size)
    return fallback_crop(img_bgr, image_size)


def imread(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Nie wczytano obrazu: {path}")
    return img
