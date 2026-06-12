"""Pełny łańcuch przetwarzania jednej próbki -> embedding + diagnostyka.

detect -> select(#1) -> liveness(#1) -> quality(#3) -> restore(#3) -> align -> embed(+TTA)

Każdy trik ma flagę w `PipelineConfig` -> łatwa ablacja "z trikiem / bez".
Zwraca obiekt z .embedding (lub None gdy odrzucono) i .info (powody/decyzje).
"""
from dataclasses import dataclass

import cv2

from config import IMG_SIZE
from pipeline import detect_align, select_face, quality, restore, liveness, embed


@dataclass
class PipelineConfig:
    use_select: bool = True       # selekcja właściwej twarzy (#1)
    use_liveness: bool = False    # bramka anti-spoof (#1) — placeholder do kroku 4
    use_quality: bool = True      # FIQA -> adaptacyjny próg (#3)
    use_restore: bool = True      # restauracja gdy niska jakość (#3)
    use_tta: bool = True          # test-time augmentation (flip)
    restore_quality_thr: float = 0.45   # restauruj tylko gdy quality < próg
    liveness_thr: float = 0.5
    ambiguity_margin: float = 0.10


@dataclass
class AuthResult:
    embedding: object   # np.ndarray(512) lub None
    quality: float
    info: dict


def process(img_bgr, cfg: PipelineConfig = PipelineConfig()):
    info = {}
    # robust detekcja: surowy obraz, a gdy pusto — paddowany (ciasne cropy/CCTV).
    # `work` to obraz, w którego układzie są bbox/kps wykrytych twarzy.
    faces, work = detect_align.detect_faces_robust(img_bgr)

    # --- wybór twarzy (#1) ---
    if cfg.use_select:
        face, sel = select_face.select(faces, work.shape, cfg.ambiguity_margin)
        info["select"] = sel
        if face is None:
            return AuthResult(None, 0.0, {**info, "rejected": sel["reason"]})
    else:
        face = max(faces, key=lambda f: f.det_score) if faces else None

    # --- liveness (#1) ---
    if cfg.use_liveness and face is not None:
        live, lscore = liveness.is_live(work, getattr(face, "bbox", None), cfg.liveness_thr)
        info["liveness"] = {"live": live, "score": lscore}
        if not live:
            return AuthResult(None, 0.0, {**info, "rejected": "spoof"})

    # --- crop do oceny jakości (przed restauracją) ---
    if face is not None:
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        face_px = max(1, min(x2 - x1, y2 - y1))
        roi = work[max(0, y1):y2, max(0, x1):x2]
    else:
        face_px = None
        roi = work

    # --- jakość (#3) ---
    if cfg.use_quality and roi.size > 0:
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        q = quality.face_quality(roi_rgb, face_px=face_px)
    else:
        q = 1.0
    info["quality"] = q

    # --- restauracja warunkowa (#3) ---
    if cfg.use_restore and q < cfg.restore_quality_thr:
        restored = restore.restore(work)
        info["restored"] = True
        faces2, work2 = detect_align.detect_faces_robust(restored)
        if faces2:
            face = max(faces2, key=lambda f: f.det_score)
            work = work2
        else:
            work = restored

    # --- align + embedding ---
    if face is not None:
        aligned = detect_align.align_from_kps(work, face.kps, IMG_SIZE)
    else:
        aligned = detect_align.fallback_crop(work, IMG_SIZE)
    emb = embed.embed_aligned(aligned, tta=cfg.use_tta)
    return AuthResult(emb, q, info)
