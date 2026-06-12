"""Ewaluacja: weryfikacja (EER/FAR/FRR) + identyfikacja (TIR) na clean vs trudnych,
z ablacją trików (pełny pipeline vs baseline).

Embeddingi liczone są RAZ na konfigurację i cache'owane (wznawialne) w
artifacts/results/emb_<cfg>.pkl — bo to najdroższy krok. Metryki liczone z cache.

Protokół:
  - próg dobrany w punkcie EER na zbiorze CLEAN (pełny pipeline), potem stosowany
    do FAR/FRR na trudnych (cel: FRR nie rośnie o > 5 pp).
  - weryfikacja: genuine = probe vs galeria swojej osoby; impostor = ten sam probe
    vs galeria innej (deterministycznie wybranej) osoby.
  - identyfikacja: closed-set TIR Top-1/Top-5 na probe wdrożonych; open-set =
    odsetek poprawnych odrzuceń dla probe spoza bazy (impostorzy).
  - przy use_quality ON decyzja używa progu ADAPTACYJNEGO (luzowanego dla niskiej
    jakości) — to jeden z trików #3.

Użycie:
    python evaluate.py                      # full + baseline, wszystkie utrudnienia
    python evaluate.py --configs full       # tylko pełny pipeline
    python evaluate.py --limit 300          # szybki podgląd na podpróbce
"""
import argparse
import json
import pickle

import numpy as np
from sklearn.metrics import roc_curve, auc

from config import DATA_DIR, RESULTS_DIR, COS_DIST_THRESHOLD, SEED
from pipeline import detect_align, authenticate, quality
from pipeline.matcher import load_db

CONFIGS = {
    "full": authenticate.PipelineConfig(),
    "baseline": authenticate.PipelineConfig(use_select=False, use_quality=False,
                                            use_restore=False, use_tta=False),
}


# --------------------------------------------------------- embeddingi (cache)
def compute_embeddings(manifest, cfg_name, limit=None):
    cache = RESULTS_DIR / f"emb_{cfg_name}.pkl"
    done = {}
    if cache.exists():
        done = pickle.load(open(cache, "rb"))
    cfg = CONFIGS[cfg_name]
    records = []
    items = manifest if limit is None else manifest[:limit]
    for i, m in enumerate(items):
        key = m["path"]
        if key in done:
            rec = done[key]
        else:
            try:
                img = detect_align.imread(m["path"])
                res = authenticate.process(img, cfg)
                rec = {"emb": res.embedding, "quality": res.quality,
                       "rejected": res.info.get("rejected")}
            except Exception as e:  # noqa: BLE001
                rec = {"emb": None, "quality": 0.0, "rejected": f"error:{e}"}
            done[key] = rec
        rec = {**rec, **m}
        records.append(rec)
        if (i + 1) % 250 == 0:
            pickle.dump(done, open(cache, "wb"))
            print(f"  [{cfg_name}] {i + 1}/{len(items)}")
    pickle.dump(done, open(cache, "wb"))
    return records


# --------------------------------------------------------------- galeria
def build_gallery(db):
    return {u: np.stack(rec["embeddings"]) for u, rec in db.items()}


def dist_to_user(emb, gal):
    return float(1.0 - np.max(gal @ emb))


# ----------------------------------------------------------- metryki
def eer_threshold(genuine, impostor):
    """EER + próg na dystansach (label: genuine=1). Zwraca (eer, thr, auc)."""
    scores = np.concatenate([genuine, impostor])
    labels = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    # similarity = -dist (większa = bardziej genuine)
    fpr, tpr, thr = roc_curve(labels, -scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return float(eer), float(-thr[idx]), float(auc(fpr, tpr))


def threshold_at_far(impostor, target_far=0.01):
    """Próg operacyjny przy zadanym FAR na impostorach (akceptacja gdy dist<=thr).

    Stabilniejszy niż EER przy idealnej separacji na clean i zgodny z celem
    'minimalizować FAR'. target_far=0.01 -> 1% impostorów przechodzi.
    """
    if len(impostor) == 0:
        return COS_DIST_THRESHOLD
    return float(np.quantile(impostor, target_far))


def verification(records, gallery, base_thr, use_adaptive):
    """Zwraca dict z genuine/impostor dystansami + FAR/FRR przy base_thr."""
    users = sorted(gallery.keys())
    rng = np.random.default_rng(SEED)
    gen, imp = [], []
    gen_acc, imp_acc = 0, 0  # decyzje przy progu (z adaptacją)
    gen_n = imp_n = 0
    for r in records:
        if r.get("emb") is None or "identity" not in r:
            continue
        u = r["identity"]
        if u not in gallery:
            continue
        thr_eff = quality.adaptive_threshold(base_thr, r["quality"]) if use_adaptive else base_thr
        # genuine
        dg = dist_to_user(r["emb"], gallery[u])
        gen.append(dg); gen_n += 1
        gen_acc += int(dg <= thr_eff)
        # impostor: inny użytkownik (deterministycznie)
        other = users[(users.index(u) + 1 + rng.integers(0, len(users) - 1)) % len(users)]
        if other == u:
            other = users[(users.index(u) + 1) % len(users)]
        di = dist_to_user(r["emb"], gallery[other])
        imp.append(di); imp_n += 1
        imp_acc += int(di <= thr_eff)
    gen, imp = np.array(gen), np.array(imp)
    frr = 1 - gen_acc / max(1, gen_n)   # genuine odrzucone
    far = imp_acc / max(1, imp_n)       # impostor zaakceptowani
    return {"genuine": gen, "impostor": imp, "FRR": frr, "FAR": far,
            "n_genuine": gen_n, "n_impostor": imp_n}


def identification(records, gallery, base_thr, use_adaptive):
    users = sorted(gallery.keys())
    top1 = top5 = closed_n = 0
    rej_ok = rej_n = 0
    for r in records:
        if r.get("emb") is None:
            continue
        # dystans do każdej osoby = min po galerii
        dists = np.array([dist_to_user(r["emb"], gallery[u]) for u in users])
        order = np.argsort(dists)
        thr_eff = quality.adaptive_threshold(base_thr, r["quality"]) if use_adaptive else base_thr
        if "identity" in r and r["identity"] in gallery:
            closed_n += 1
            true_i = users.index(r["identity"])
            top1 += int(order[0] == true_i)
            top5 += int(true_i in order[:5])
        elif "impostor" in r:        # spoza bazy: poprawnie = odrzucony
            rej_n += 1
            rej_ok += int(dists[order[0]] > thr_eff)
    return {"TIR_top1": top1 / max(1, closed_n), "TIR_top5": top5 / max(1, closed_n),
            "n_closed": closed_n, "reject_rate_impostor": rej_ok / max(1, rej_n),
            "n_impostor": rej_n}


# --------------------------------------------------------------- raport
def evaluate_config(cfg_name, manifest, db, base_thr=None, limit=None):
    gallery = build_gallery(db)
    use_adaptive = CONFIGS[cfg_name].use_quality
    records = compute_embeddings(manifest, cfg_name, limit)

    by_diff = {}
    for r in records:
        by_diff.setdefault(r["difficulty"], []).append(r)

    # próg operacyjny z CLEAN przy zadanym FAR (stabilniejszy niż EER), jeśli nie podano
    if base_thr is None:
        clean = by_diff.get("clean", [])
        v = verification(clean, gallery, COS_DIST_THRESHOLD, use_adaptive)
        base_thr = threshold_at_far(v["impostor"], target_far=0.01)

    out = {"config": cfg_name, "threshold": base_thr, "per_difficulty": {}}
    for diff in ["clean", "lowres", "multiface"]:
        recs = by_diff.get(diff, [])
        if not recs:
            continue
        v = verification(recs, gallery, base_thr, use_adaptive)
        eer, _, roc_auc = eer_threshold(v["genuine"], v["impostor"]) if len(v["genuine"]) else (None, None, None)
        idi = identification(recs, gallery, base_thr, use_adaptive)
        rejected = sum(1 for r in recs if r.get("emb") is None)
        out["per_difficulty"][diff] = {
            "n": len(recs), "rejected_by_pipeline": rejected,
            "EER": eer, "ROC_AUC": roc_auc, "FAR": v["FAR"], "FRR": v["FRR"],
            "n_genuine": v["n_genuine"], "n_impostor_pairs": v["n_impostor"],
            "TIR_top1": idi["TIR_top1"], "TIR_top5": idi["TIR_top5"],
            "n_closed": idi["n_closed"],
            "reject_rate_impostor": idi["reject_rate_impostor"],
        }
    return out, base_thr


def print_report(results):
    for res in results:
        print(f"\n{'='*70}\nKONFIG: {res['config']}   próg={res['threshold']:.4f}")
        print(f"{'utrudnienie':<12} {'n':>5} {'EER':>7} {'FRR':>7} {'FAR':>7} "
              f"{'TIR@1':>7} {'TIR@5':>7} {'rej_imp':>8}")
        clean = res["per_difficulty"].get("clean", {})
        for diff, m in res["per_difficulty"].items():
            def pp(x):
                return f"{100*x:6.2f}%" if isinstance(x, (int, float)) and x is not None else "   -  "
            dfrr = ""
            if diff != "clean" and clean:
                dfrr = f"  ΔFRR={100*(m['FRR']-clean['FRR']):+.2f}pp  ΔTIR={100*(m['TIR_top1']-clean['TIR_top1']):+.2f}pp"
            print(f"{diff:<12} {m['n']:>5} {pp(m['EER'])} {pp(m['FRR'])} {pp(m['FAR'])} "
                  f"{pp(m['TIR_top1'])} {pp(m['TIR_top5'])} {pp(m['reject_rate_impostor'])}{dfrr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["full", "baseline"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    db = load_db()
    print(f"manifest: {len(manifest)} próbek | baza: {len(db)} użytkowników")

    results = []
    shared_thr = None
    for cfg_name in args.configs:
        res, thr = evaluate_config(cfg_name, manifest, db,
                                   base_thr=None if cfg_name == "full" else shared_thr,
                                   limit=args.limit)
        if cfg_name == "full":
            shared_thr = thr   # baseline porównujemy przy tym samym progu
        results.append(res)

    print_report(results)
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps([{**r, "per_difficulty": r["per_difficulty"]} for r in results], indent=2,
                   default=lambda o: None))
    print(f"\nZapisano {RESULTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
