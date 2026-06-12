"""CLI identyfikacji 1:N (open-set) przez pełny pipeline.

    python identify.py <ścieżka_obrazu>
"""
import argparse

from config import COS_DIST_THRESHOLD
from pipeline import detect_align, authenticate, quality
from pipeline.matcher import load_db, identify as identify_match


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--no-tricks", action="store_true")
    args = ap.parse_args()

    cfg = authenticate.PipelineConfig()
    if args.no_tricks:
        cfg = authenticate.PipelineConfig(use_select=False, use_quality=False,
                                          use_restore=False, use_tta=False)

    img = detect_align.imread(args.image)
    res = authenticate.process(img, cfg)
    if res.embedding is None:
        print(f"ODRZUCONO na pipeline: {res.info.get('rejected')}")
        return

    db = load_db()
    thr = quality.adaptive_threshold(COS_DIST_THRESHOLD, res.quality) if cfg.use_quality \
        else COS_DIST_THRESHOLD
    pred, ranking = identify_match(db, res.embedding, threshold=thr)
    print(f"PREDYKCJA: {pred or '<odrzucony / brak w bazie>'}  (thr={thr:.3f}, q={res.quality:.2f})")
    print("Top-5:")
    for user, dist in ranking:
        print(f"  {user:30s}  dist={dist:.4f}")


if __name__ == "__main__":
    main()
