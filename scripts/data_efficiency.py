#!/usr/bin/env python3
"""
Data-efficiency curve: how few labeled samples does the probe need?

Motivation: the cross-encoder beats the probe in-domain (0.759 vs 0.685) but
is purpose-trained on ~500k MS MARCO pairs. The probe is a linear layer on a
frozen general LLM. This script measures probe test AUC as a function of the
number of TRAINING QUERIES, with val/test held FIXED — the data-efficiency
story. Runs entirely on cached activations (no GPU, seconds).

Subsampling is by QUERY (not sample) to respect no-leakage and mirror the
real "label N queries" cost. For each train size we:
  rank heads by val AUC (probes trained on the subsample) -> top-k
  -> L2 ensemble on the subsample -> eval on the fixed test split.
Repeated over several seeds per size to get mean +/- std (small sizes noisy).

Reference line: cross-encoder test AUC 0.759 (from compare_methods.py, needs
~500k pretraining pairs).

Usage:
  python scripts/data_efficiency.py --cache results/cache/q500_instruct --scheme attn
"""
import os
import sys
import argparse
import json
from collections import defaultdict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N_JOBS = max(1, min(48, os.cpu_count() - 2))
CROSS_ENCODER_TEST_AUC = 0.759  # ms-marco-MiniLM-L-6-v2, from compare_methods.py


def lr(C=1.0, penalty="l2"):
    return LogisticRegression(penalty=penalty, C=C,
                              solver="lbfgs" if penalty == "l2" else "liblinear",
                              max_iter=2000, class_weight="balanced", random_state=42)


def load(cache, scheme, split):
    d = torch.load(os.path.join(cache, f"{scheme}_{split}.pt"))
    return d["acts"].to(torch.float32).numpy(), d["labels"].numpy(), d["query_ids"].numpy()


def auc(y, s):
    try:
        return roc_auc_score(y, s)
    except ValueError:
        return 0.5


def fit_eval(Xtr, ytr, Xva, yva, Xte, yte, topk):
    """Train probe on (Xtr,ytr): rank heads by val AUC, top-k, L2 ensemble. Return test AUC."""
    n_heads = Xtr.shape[1]

    def head_auc(h):
        p = lr(penalty="l1").fit(Xtr[:, h, :], ytr)
        return auc(yva, p.predict_proba(Xva[:, h, :])[:, 1])

    vaucs = np.array(Parallel(n_jobs=N_JOBS)(delayed(head_auc)(h) for h in range(n_heads)))
    sel = np.argsort(vaucs)[::-1][:topk]

    def feats(X):
        return np.concatenate([X[:, h, :] for h in sel], axis=1)

    probe = lr().fit(feats(Xtr), ytr)
    return auc(yte, probe.predict_proba(feats(Xte))[:, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/cache/q500_instruct")
    ap.add_argument("--scheme", default="attn")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--sizes", type=int, nargs="+", default=[3, 7, 14, 35, 70, 140, 210])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="results/data_efficiency.json")
    args = ap.parse_args()

    Xtr, ytr, qtr = load(args.cache, args.scheme, "train")
    Xva, yva, _ = load(args.cache, args.scheme, "val")
    Xte, yte, _ = load(args.cache, args.scheme, "test")

    # group train sample indices by query
    by_q = defaultdict(list)
    for i, q in enumerate(qtr):
        by_q[q].append(i)
    all_qids = list(by_q.keys())
    n_total_q = len(all_qids)
    print(f"train: {len(ytr)} samples / {n_total_q} queries | val {len(yva)} | test {len(yte)}")
    print(f"scheme={args.scheme} topk={args.topk}  (cross-encoder ref test AUC={CROSS_ENCODER_TEST_AUC})\n")

    curve = []
    for size in args.sizes:
        size = min(size, n_total_q)
        aucs = []
        for seed in args.seeds:
            rng = np.random.RandomState(seed)
            chosen_q = rng.choice(all_qids, size=size, replace=False)
            idx = np.concatenate([by_q[q] for q in chosen_q])
            ys = ytr[idx]
            if ys.sum() == 0 or ys.sum() == len(ys):  # need both classes
                continue
            a = fit_eval(Xtr[idx], ys, Xva, yva, Xte, yte, args.topk)
            aucs.append(a)
        if not aucs:
            continue
        mean, std = float(np.mean(aucs)), float(np.std(aucs))
        n_samples = int(np.mean([len(np.concatenate([by_q[q] for q in
                       np.random.RandomState(s).choice(all_qids, size=size, replace=False)]))
                       for s in args.seeds]))
        curve.append({"n_queries": size, "approx_n_samples": n_samples,
                      "test_auc_mean": round(mean, 3), "test_auc_std": round(std, 3)})
        print(f"  {size:3d} queries (~{n_samples:4d} samples): test AUC {mean:.3f} +/- {std:.3f}")

    json.dump({"scheme": args.scheme, "topk": args.topk,
               "cross_encoder_ref": CROSS_ENCODER_TEST_AUC, "curve": curve},
              open(args.out, "w"), indent=2)
    print(f"\nSaved: {args.out}")
    if curve:
        full = curve[-1]["test_auc_mean"]
        for c in curve:
            if c["test_auc_mean"] >= full - 0.02:
                print(f"Reaches within 0.02 of full-data AUC ({full}) at "
                      f"{c['n_queries']} queries (~{c['approx_n_samples']} samples).")
                break


if __name__ == "__main__":
    main()
