"""Ocena jakości twarzy (FIQA proxy) + adaptacyjny próg (utrudnienie #3).

Pełne FIQA (CR-FIQA / SER-FIQ) wymaga osobnej sieci; tu używamy taniego proxy:
ostrość (wariancja Laplasjanu) + rozmiar wykrytej twarzy. Quality w [0, 1].

Adaptacyjny próg: dla niskiej jakości LUZUJEMY próg dystansu (akceptujemy nieco
większy dystans), żeby nie wystrzelił FRR — przy jednoczesnym podbiciu wymogu
przez normalizację score'ów na etapie matchera. To jeden z kluczowych trików do
utrzymania degradacji < 5 pp.
"""
import cv2
import numpy as np


def sharpness(rgb_uint8):
    """Wariancja Laplasjanu — im wyżej, tym ostrzej."""
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def face_quality(rgb_uint8, face_px=None, sharp_ref=150.0, size_ref=80.0):
    """Quality w [0, 1] z ostrości i rozmiaru twarzy (w px, przed alignem).

    sharp_ref / size_ref to wartości "dobrej" próbki; kalibrowalne na zbiorze
    czystym. face_px=None -> liczona tylko ostrość.
    """
    s = min(1.0, sharpness(rgb_uint8) / sharp_ref)
    if face_px is None:
        return float(s)
    sz = min(1.0, face_px / size_ref)
    return float(0.6 * s + 0.4 * sz)


def adaptive_threshold(base_threshold, quality, max_relax=0.08):
    """Luzuje próg dystansu dla niskiej jakości.

    quality=1 -> base_threshold; quality=0 -> base_threshold + max_relax.
    """
    return base_threshold + max_relax * (1.0 - max(0.0, min(1.0, quality)))
