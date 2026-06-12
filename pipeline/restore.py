"""Restauracja / super-rozdzielczość twarzy przed ArcFace (utrudnienie #3).

MVP: baseline bicubic upscale + unsharp mask (działa na CPU, natychmiast).
Interfejs `restore(bgr) -> bgr` jest celowo prosty, żeby podmienić baseline na
GFPGAN / CodeFormer / Real-ESRGAN bez zmian w reszcie pipeline'u.

Stosujemy restaurację WARUNKOWO — tylko gdy quality < próg (patrz pipeline.quality),
żeby nie psuć dobrych próbek.
"""
import cv2
import numpy as np

_BACKEND = "baseline"   # 'baseline' | 'gfpgan' | 'codeformer'
_gfpgan = None


def _unsharp(bgr, amount=1.0, sigma=1.0):
    blur = cv2.GaussianBlur(bgr, (0, 0), sigma)
    return cv2.addWeighted(bgr, 1 + amount, blur, -amount, 0)


def restore_baseline(bgr, target=256):
    """Upscale do `target` (bicubic) + delikatne wyostrzenie."""
    h, w = bgr.shape[:2]
    if max(h, w) < target:
        scale = target / max(h, w)
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_CUBIC)
    return _unsharp(bgr, amount=0.6, sigma=1.0)


def restore(bgr):
    """Główny punkt wejścia. Zwraca obraz BGR po restauracji."""
    if _BACKEND == "baseline":
        return restore_baseline(bgr)
    if _BACKEND == "gfpgan":
        return _restore_gfpgan(bgr)
    raise ValueError(f"Nieznany backend restauracji: {_BACKEND}")


def set_backend(name):
    global _BACKEND
    _BACKEND = name


def _restore_gfpgan(bgr):
    """Placeholder pod GFPGAN — wymaga `pip install gfpgan` + wag.

    Uzupełnić w kroku 3 (triki #3). Na razie fallback do baseline, żeby
    pipeline był uruchamialny end-to-end bez dodatkowych zależności.
    """
    global _gfpgan
    if _gfpgan is None:
        # from gfpgan import GFPGANer
        # _gfpgan = GFPGANer(model_path=..., upscale=2, arch='clean', ...)
        return restore_baseline(bgr)
    _, _, out = _gfpgan.enhance(bgr, has_aligned=False, paste_back=True)
    return out
