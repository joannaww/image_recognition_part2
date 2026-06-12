"""CLI weryfikacji 1:1 przez pełny pipeline odpornościowy.

    python verify.py <ścieżka_obrazu> <user>
"""
import argparse

from config import COS_DIST_THRESHOLD
from pipeline import detect_align, authenticate, quality
from pipeline.matcher import load_db, verify as verify_match


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("user")
    ap.add_argument("--no-tricks", action="store_true", help="wyłącz triki (baseline)")
    args = ap.parse_args()

    cfg = authenticate.PipelineConfig()
    if args.no_tricks:
        cfg = authenticate.PipelineConfig(use_select=False, use_quality=False,
                                          use_restore=False, use_tta=False)

    img = detect_align.imread(args.image)
    res = authenticate.process(img, cfg)
    if res.embedding is None:
        print(f"ODRZUCONO na pipeline: {res.info.get('rejected')}  ({res.info})")
        return

    db = load_db()
    thr = quality.adaptive_threshold(COS_DIST_THRESHOLD, res.quality) if cfg.use_quality \
        else COS_DIST_THRESHOLD
    ok, dist = verify_match(db, args.user, res.embedding, threshold=thr)
    print(f"user={args.user}  dist={dist:.4f}  thr={thr:.4f}  quality={res.quality:.2f}  "
          f"-> {'AKCEPTACJA' if ok else 'ODRZUCENIE'}")


if __name__ == "__main__":
    main()
