"""Selekcja właściwej twarzy, gdy na obrazie jest ich wiele (utrudnienie #1).

Polityka: faworyzujemy twarz dużą, centralną i pewnie wykrytą. To trik
test-time — żaden model nie jest dotrenowywany.
"""
import numpy as np


def _centrality(bbox, img_shape):
    """1.0 = środek kadru, maleje ku brzegom."""
    h, w = img_shape[:2]
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    dx = abs(cx - w / 2) / (w / 2)
    dy = abs(cy - h / 2) / (h / 2)
    return 1.0 - min(1.0, (dx + dy) / 2)


def _area_frac(bbox, img_shape):
    h, w = img_shape[:2]
    bw = max(0.0, bbox[2] - bbox[0])
    bh = max(0.0, bbox[3] - bbox[1])
    return (bw * bh) / float(h * w)


def score_faces(faces, img_shape, w_area=1.0, w_center=0.6, w_det=0.4):
    """Zwraca listę (face, score) posortowaną malejąco."""
    scored = []
    for f in faces:
        area = _area_frac(f.bbox, img_shape)
        center = _centrality(f.bbox, img_shape)
        det = float(getattr(f, "det_score", 1.0))
        score = w_area * np.sqrt(area) + w_center * center + w_det * det
        scored.append((f, float(score)))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored


def select(faces, img_shape, ambiguity_margin=0.10):
    """Wybiera twarz do uwierzytelnienia.

    Zwraca (face, info). face=None gdy brak twarzy albo wybór jest
    niejednoznaczny (dwie twarze o zbliżonym score < ambiguity_margin) —
    wtedy lepiej odrzucić niż zgadywać (minimalizacja FAR).
    """
    if not faces:
        return None, {"reason": "no_face"}
    scored = score_faces(faces, img_shape)
    if len(scored) == 1:
        return scored[0][0], {"reason": "single", "score": scored[0][1]}
    top, second = scored[0], scored[1]
    if (top[1] - second[1]) < ambiguity_margin:
        return None, {"reason": "ambiguous", "top": top[1], "second": second[1]}
    return top[0], {"reason": "selected", "score": top[1], "n_faces": len(scored)}
