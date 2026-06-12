"""Bramka liveness / anti-spoof — POZA ZAKRESEM projektu (opcjonalna).

Utrudnienie #1 zawężono do "wiele twarzy -> wybierz właściwą"; wątek twarzy na
plakatach/ekranach (anti-spoofing) pominięto, bo wymaga zbioru realnych ataków
print/replay do uczciwej ewaluacji. Moduł zostaje jako interfejs + placeholder na
wypadek rozszerzenia (Silent-Face MiniFASNet). Domyślnie WYŁĄCZONY w pipeline
(PipelineConfig.use_liveness=False).

is_live(bgr, bbox) -> (bool, score). score w [0,1] = p(żywa twarz).
"""
import cv2
import numpy as np

_BACKEND = "placeholder"   # 'placeholder' | 'minifasnet'
_model = None


def is_live(img_bgr, bbox=None, threshold=0.5):
    if _BACKEND == "placeholder":
        score = _placeholder_score(img_bgr, bbox)
    elif _BACKEND == "minifasnet":
        score = _minifasnet_score(img_bgr, bbox)
    else:
        raise ValueError(f"Nieznany backend liveness: {_BACKEND}")
    return score >= threshold, float(score)


def set_backend(name):
    global _BACKEND
    _BACKEND = name


def _placeholder_score(img_bgr, bbox):
    """Heurystyka tymczasowa (NIE do raportu): wyższy 'kolor/tekstura' -> żywsze.

    Zwraca ~1.0 dla typowych zdjęć, lekko karze obszary o niskiej saturacji i
    silnych periodycznych wzorach (proxy moiré). Zastąpić MiniFASNet.
    """
    crop = _crop(img_bgr, bbox)
    if crop.size == 0:
        return 1.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1].mean() / 255.0
    return float(np.clip(0.5 + 0.5 * sat, 0.0, 1.0))


def _crop(img_bgr, bbox):
    if bbox is None:
        return img_bgr
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return img_bgr[y1:y2, x1:x2]


def _minifasnet_score(img_bgr, bbox):
    """Placeholder pod Silent-Face MiniFASNet (uzupełnić w kroku 4)."""
    global _model
    if _model is None:
        return _placeholder_score(img_bgr, bbox)
    raise NotImplementedError
