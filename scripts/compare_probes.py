#!/usr/bin/env python3
"""
Compare probe methodologies on CACHED activations (no model reload).

Diagnoses and fixes the generalization failure found in the 300q run
(corr(val_auc, test_auc)=0.028, ensemble test_auc~0.50). Each method below
is a different answer to "how do we turn 672x128 head activations into a
relevance score that generalizes to held-out queries".

Methods (all share the same cached activations & query-disjoint splits):
  M0  baseline      : top-k heads by val AUC, concat RAW feats, L2 LR  (current pipeline)
  M1  +scaler       : M0 but StandardScaler on features
  M2  cv-select     : rank heads by k-fold CV AUC on TRAIN (not val), then scaled ensemble
  M3  whole-L2      : all heads, head-mean-pooled to 672 dims, scaled, heavy-L2 sweep
  M4  stacking      : per-head OOF relevance scores (672 meta-feats) -> L2 meta-probe
  M5  mean-probe    : single probe on mean over ALL heads (128 dims) -- cheap sanity floor

Reports train/val/test ROC-AUC for each so we can see the train->test gap.
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

N_JOBS = max(1, min(48, os.cpu_count() - 2))  # parallelize per-head fits across cores

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_split(cache_dir, scheme, split):
    d = torch.load(os.path.join(cache_dir, f"{scheme}_{split}.pt"))
    X = d["acts"].to(torch.float32).numpy()   # [N, total_heads, head_dim]
    y = d["labels"].numpy()
    return X, y


def auc(y, s):
    try:
        return roc_auc_score(y, s)
    except ValueError:
        return 0.5


def lr(C=1.0, penalty="l2"):
    # liblinear is much faster than saga for L1 on small (128-dim) feature sets
    return LogisticRegression(penalty=penalty, C=C,
                              solver="lbfgs" if penalty == "l2" else "liblinear",
                              max_iter=2000, class_weight="balanced", random_state=42)


def per_head_val_auc(Xtr, ytr, Xva, yva):
    """Train one LR per head on train, score each on val. Returns array[n_heads]."""
    n_heads = Xtr.shape[1]

    def one(h):
        p = lr(penalty="l1").fit(Xtr[:, h, :], ytr)
        return auc(yva, p.predict_proba(Xva[:, h, :])[:, 1])

    return np.array(Parallel(n_jobs=N_JOBS)(delayed(one)(h) for h in range(n_heads)))


def per_head_cv_auc(Xtr, ytr, k=5):
    """Rank heads by k-fold CV AUC on TRAIN only (avoids single-split selection bias)."""
    n_heads = Xtr.shape[1]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    def one(h):
        oof = cross_val_predict(lr(penalty="l1"), Xtr[:, h, :], ytr,
                                cv=skf, method="predict_proba")[:, 1]
        return auc(ytr, oof)

    return np.array(Parallel(n_jobs=N_JOBS)(delayed(one)(h) for h in range(n_heads)))


def ensemble(Xtr, ytr, Xte, heads, scale=True, C=1.0):
    """Concat selected heads' raw feats -> (scaled) L2 LR. Returns test scores."""
    def feats(X):
        return np.concatenate([X[:, h, :] for h in heads], axis=1)
    Ftr, Fte = feats(Xtr), feats(Xte)
    if scale:
        sc = StandardScaler().fit(Ftr)
        Ftr, Fte = sc.transform(Ftr), sc.transform(Fte)
    p = lr(C=C).fit(Ftr, ytr)
    return p.predict_proba(Ftr)[:, 1], p.predict_proba(Fte)[:, 1]


def evaluate(name, ytr, str_, yva, sva, yte, ste, log):
    log.append({"method": name,
                "train_auc": round(auc(ytr, str_), 3),
                "val_auc": round(auc(yva, sva), 3),
                "test_auc": round(auc(yte, ste), 3)})
    print(f"  {name:16s} train={auc(ytr,str_):.3f}  val={auc(yva,sva):.3f}  test={auc(yte,ste):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/cache/q500")
    # scheme = the .pt filename prefix in the cache dir (last_token/pooling, or attn/resid)
    ap.add_argument("--scheme", default="last_token")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.cache, "meta.json")))
    print(f"Cache: {args.cache} | scheme={args.scheme} | {meta['n_queries']}q | "
          f"heads={meta['n_layers']*meta['n_heads']} x {meta['head_dim']}")

    Xtr, ytr = load_split(args.cache, args.scheme, "train")
    Xva, yva = load_split(args.cache, args.scheme, "val")
    Xte, yte = load_split(args.cache, args.scheme, "test")
    print(f"shapes train={Xtr.shape} val={Xva.shape} test={Xte.shape}")

    results = []
    K = args.topk

    # --- head rankings ---
    print("\nRanking heads (val-AUC and CV-AUC)...")
    val_aucs = per_head_val_auc(Xtr, ytr, Xva, yva)
    cv_aucs = per_head_cv_auc(Xtr, ytr)
    heads_by_val = np.argsort(val_aucs)[::-1][:K]
    heads_by_cv = np.argsort(cv_aucs)[::-1][:K]
    # sanity: do val-selected and cv-selected heads overlap?
    overlap = len(set(heads_by_val.tolist()) & set(heads_by_cv.tolist()))
    print(f"  top-{K} head overlap (val vs cv selection): {overlap}/{K}")

    print(f"\n=== Methods (scheme={args.scheme}, top-{K}) ===")

    # M0 baseline: val-selected, raw, L2
    s_tr, s_te = ensemble(Xtr, ytr, Xte, heads_by_val, scale=False, C=1.0)
    _, s_va = ensemble(Xtr, ytr, Xva, heads_by_val, scale=False, C=1.0)
    evaluate("M0 baseline", ytr, s_tr, yva, s_va, yte, s_te, results)

    # M1 +scaler
    s_tr, s_te = ensemble(Xtr, ytr, Xte, heads_by_val, scale=True, C=1.0)
    _, s_va = ensemble(Xtr, ytr, Xva, heads_by_val, scale=True, C=1.0)
    evaluate("M1 +scaler", ytr, s_tr, yva, s_va, yte, s_te, results)

    # M2 cv-select + scaler
    s_tr, s_te = ensemble(Xtr, ytr, Xte, heads_by_cv, scale=True, C=1.0)
    _, s_va = ensemble(Xtr, ytr, Xva, heads_by_cv, scale=True, C=1.0)
    evaluate("M2 cv-select", ytr, s_tr, yva, s_va, yte, s_te, results)

    # M3 whole-L2 on head-mean-pooled (672 dims), C sweep
    Htr = Xtr.mean(axis=2); Hva = Xva.mean(axis=2); Hte = Xte.mean(axis=2)  # [N, n_heads]
    sc = StandardScaler().fit(Htr)
    Htr_s, Hva_s, Hte_s = sc.transform(Htr), sc.transform(Hva), sc.transform(Hte)
    best = None
    for C in [0.001, 0.01, 0.1, 1.0]:
        p = lr(C=C).fit(Htr_s, ytr)
        va = auc(yva, p.predict_proba(Hva_s)[:, 1])
        if best is None or va > best[0]:
            best = (va, C, p)
    p = best[2]
    evaluate(f"M3 whole-L2(C={best[1]})", ytr, p.predict_proba(Htr_s)[:, 1],
             yva, p.predict_proba(Hva_s)[:, 1], yte, p.predict_proba(Hte_s)[:, 1], results)

    # M4 stacking: per-head OOF scores on train -> meta L2
    print("  (building stacking meta-features...)")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n_heads = Xtr.shape[1]

    def head_meta(h):
        oof = cross_val_predict(lr(penalty="l1"), Xtr[:, h, :], ytr, cv=skf, method="predict_proba")[:, 1]
        ph = lr(penalty="l1").fit(Xtr[:, h, :], ytr)
        return oof, ph.predict_proba(Xva[:, h, :])[:, 1], ph.predict_proba(Xte[:, h, :])[:, 1]

    cols = Parallel(n_jobs=N_JOBS)(delayed(head_meta)(h) for h in range(n_heads))
    Mtr = np.column_stack([c[0] for c in cols])
    Mva = np.column_stack([c[1] for c in cols])
    Mte = np.column_stack([c[2] for c in cols])
    scm = StandardScaler().fit(Mtr)
    bestm = None
    for C in [0.001, 0.01, 0.1, 1.0]:
        pm = lr(C=C).fit(scm.transform(Mtr), ytr)
        va = auc(yva, pm.predict_proba(scm.transform(Mva))[:, 1])
        if bestm is None or va > bestm[0]:
            bestm = (va, C, pm)
    pm = bestm[2]
    evaluate(f"M4 stacking(C={bestm[1]})", ytr, pm.predict_proba(scm.transform(Mtr))[:, 1],
             yva, pm.predict_proba(scm.transform(Mva))[:, 1],
             yte, pm.predict_proba(scm.transform(Mte))[:, 1], results)

    # M5 mean-probe floor: mean over all heads -> 128 dims
    Atr = Xtr.mean(axis=1); Ava = Xva.mean(axis=1); Ate = Xte.mean(axis=1)
    sca = StandardScaler().fit(Atr)
    pa = lr(C=1.0).fit(sca.transform(Atr), ytr)
    evaluate("M5 mean-probe", ytr, pa.predict_proba(sca.transform(Atr))[:, 1],
             yva, pa.predict_proba(sca.transform(Ava))[:, 1],
             yte, pa.predict_proba(sca.transform(Ate))[:, 1], results)

    out = args.out or os.path.join(args.cache, f"probe_compare_{args.scheme}.json")
    json.dump({"scheme": args.scheme, "topk": K, "head_overlap_val_cv": overlap,
               "results": results}, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
