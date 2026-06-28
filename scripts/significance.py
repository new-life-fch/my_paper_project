#!/usr/bin/env python3
"""
Statistical significance for the internals>output edges, across all cached models.

Why this script: the multimodel table reports bare test AUCs. Reviewers will ask
whether the edges (internal_edge = best_resid - final; train_edge = final - judge;
attn - final) are distinguishable from noise. With a query-level 70/15/15 split,
the correct resampling unit is the QUERY, not the passage (passages under one query
are correlated). We therefore do a QUERY-CLUSTERED bootstrap on the fixed test
predictions of each reader.

Two honesty upgrades over internals_vs_output.py:
  * best resid layer is selected on VALIDATION, then reported on TEST (the original
    argmax-on-test is optimistically biased).
  * every edge gets a 95% CI and a one-sided bootstrap p-value (P[edge <= 0]).

Probes are fit ONCE on train (scores frozen); the bootstrap resamples the test
queries only -> this gives a CI on the evaluation, the standard approach for AUC.

Usage:
  python scripts/significance.py                 # all models
  python scripts/significance.py --boot 5000
"""
import os, json, argparse
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

N_JOBS = max(1, min(48, (os.cpu_count() or 4) - 2))

# tag -> (instruct_cache, judge_cache); base 3B uses the original q500_* dirs
MODELS = [
    ("llama32_3b",          "LLaMA-3.2-3B",        "q500"),
    ("llama31_8b",          "LLaMA-3.1-8B",        "llama31_8b"),
    ("mistral_7b",          "Mistral-7B-v0.3",     "mistral_7b"),
    ("llama32_3b_instruct", "LLaMA-3.2-3B-Inst",   "llama32_3b_instruct"),
    ("llama31_8b_instruct", "LLaMA-3.1-8B-Inst",   "llama31_8b_instruct"),
    ("mistral_7b_instruct", "Mistral-7B-Inst-v0.3","mistral_7b_instruct"),
    # second IN-DOMAIN relevance dataset (FiQA, trained+tested in-domain)
    ("llama32_3b_fiqa",     "LLaMA-3.2-3B FiQA",   "llama32_3b_fiqa"),
]


def lr(penalty="l2"):
    return LogisticRegression(penalty=penalty, C=(1.0 if penalty == "l2" else 1.0),
                              solver="lbfgs" if penalty == "l2" else "liblinear",
                              max_iter=3000, class_weight="balanced", random_state=42)


def auc(y, s):
    try: return roc_auc_score(y, s)
    except ValueError: return 0.5


def load(cache, scheme, split):
    d = torch.load(os.path.join(cache, f"{scheme}_{split}.pt"))
    return (d["acts"].to(torch.float32).numpy(), d["labels"].numpy(),
            d.get("query_ids"))


def boot_indices(qids, n_boot, rng):
    """Pre-generate query-clustered bootstrap row-index sets (shared across readers
    so that edge = AUC(reader_hi) - AUC(reader_lo) is a PAIRED difference: both
    readers are scored on the SAME resampled queries each iteration). Paired
    differencing cancels the large positive covariance between correlated readers,
    giving correct (tighter) CIs than independent resampling."""
    qids = np.asarray(qids)
    uq = np.unique(qids)
    idx_by_q = {q: np.where(qids == q)[0] for q in uq}
    sets = []
    for _ in range(n_boot):
        sampq = rng.choice(uq, size=len(uq), replace=True)
        sets.append(np.concatenate([idx_by_q[q] for q in sampq]))
    return sets


def boot_auc(y, s, idx_sets):
    """AUCs for one fixed score vector over pre-generated bootstrap index sets."""
    return np.array([auc(y[idx], s[idx]) for idx in idx_sets])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--suffix", default="", help="cache suffix, e.g. _q1500 -> looks for <cdir>_q1500_instruct")
    ap.add_argument("--out", default="results/significance.json")
    ap.add_argument("--only", default="", help="comma-separated tags to run (default: all)")
    args = ap.parse_args()
    rng = np.random.default_rng(42)

    only = set(t for t in args.only.split(",") if t)
    report = []
    for tag, name, cdir in MODELS:
        if only and tag not in only:
            continue
        # scaleup caches are tag-based (<tag>_q1500_*); legacy 500q base-3B uses cdir=q500
        base = tag if args.suffix else cdir
        ic = f"results/cache/{base}{args.suffix}_instruct"
        jc = f"results/cache/{base}{args.suffix}_judge"
        if not os.path.isdir(ic):
            print(f"skip {tag}: no cache {ic}"); continue
        print(f"\n=== {name} ({tag}) ===")

        # readers' frozen test scores --------------------------------------
        Rtr, ytr, _   = load(ic, "resid", "train")
        Rva, yva, _   = load(ic, "resid", "val")
        Rte, yte, qte = load(ic, "resid", "test")
        n_layers = Rtr.shape[1]

        jd = torch.load(os.path.join(jc, "judge_test.pt"))
        s_judge = jd["scores"].numpy()
        assert np.array_equal(jd["labels"].numpy(), yte), f"{tag}: judge/resid label mismatch"

        # final-layer (output-representation) probe
        m_fin = make_pipeline(StandardScaler(), lr()).fit(Rtr[:, -1, :], ytr)
        s_fin = m_fin.predict_proba(Rte[:, -1, :])[:, 1]

        # best resid layer SELECTED ON VALIDATION, reported on test
        def val_auc(li):
            m = make_pipeline(StandardScaler(), lr()).fit(Rtr[:, li, :], ytr)
            return auc(yva, m.predict_proba(Rva[:, li, :])[:, 1])
        va = Parallel(n_jobs=min(N_JOBS, n_layers))(delayed(val_auc)(li) for li in range(n_layers))
        best_layer = int(np.argmax(va))
        m_best = make_pipeline(StandardScaler(), lr()).fit(Rtr[:, best_layer, :], ytr)
        s_best = m_best.predict_proba(Rte[:, best_layer, :])[:, 1]

        # attn-head probe (top-k heads by val)
        Atr, _, _ = load(ic, "attn", "train")
        Ava, _, _ = load(ic, "attn", "val")
        Ate, _, _ = load(ic, "attn", "test")
        n_heads = Atr.shape[1]
        def hauc(h):
            p = lr(penalty="l1").fit(Atr[:, h, :], ytr)
            return auc(yva, p.predict_proba(Ava[:, h, :])[:, 1])
        vaucs = np.array(Parallel(n_jobs=N_JOBS)(delayed(hauc)(h) for h in range(n_heads)))
        sel = np.argsort(vaucs)[::-1][:20]
        feats = lambda X: np.concatenate([X[:, h, :] for h in sel], axis=1)
        m_attn = lr().fit(feats(Atr), ytr)
        s_attn = m_attn.predict_proba(feats(Ate))[:, 1]

        # query-clustered PAIRED bootstrap: one shared set of resampled queries,
        # all readers scored on it -> edges are paired differences.
        idx_sets = boot_indices(qte, args.boot, rng)
        bj = boot_auc(yte, s_judge, idx_sets)
        bf = boot_auc(yte, s_fin,   idx_sets)
        bb = boot_auc(yte, s_best,  idx_sets)
        ba = boot_auc(yte, s_attn,  idx_sets)

        def ci(b):
            return [round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)]
        def edge(b_hi, b_lo):
            d = b_hi - b_lo
            return {"mean": round(float(d.mean()), 4),
                    "ci95": [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)],
                    "p_le0": round(float((d <= 0).mean()), 4)}

        rec = {
            "tag": tag, "model": name, "n_layers": n_layers, "best_layer_val": best_layer,
            "auc": {
                "judge":      {"point": round(float(auc(yte, s_judge)), 3), "ci95": ci(bj)},
                "final":      {"point": round(float(auc(yte, s_fin)), 3),   "ci95": ci(bf)},
                "best_resid": {"point": round(float(auc(yte, s_best)), 3),  "ci95": ci(bb)},
                "attn":       {"point": round(float(auc(yte, s_attn)), 3),  "ci95": ci(ba)},
            },
            "edges": {
                "internal_edge(best_resid-final)": edge(bb, bf),
                "train_edge(final-judge)":         edge(bf, bj),
                "attn-final":                      edge(ba, bf),
                "best_resid-judge":                edge(bb, bj),
            },
        }
        report.append(rec)
        e = rec["edges"]["internal_edge(best_resid-final)"]
        print(f"  best_resid L{best_layer} {rec['auc']['best_resid']['point']} vs final {rec['auc']['final']['point']}"
              f" | internal_edge {e['mean']:+.3f} CI{e['ci95']} p(<=0)={e['p_le0']}")

    json.dump(report, open(args.out, "w"), indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
