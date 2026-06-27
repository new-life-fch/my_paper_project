#!/usr/bin/env python3
"""
Hyperparameter sensitivity sweep — answers "does tuning the probe (top-k heads,
scaling, regularization) change the verdict that it underperforms?".

Runs on cached activations only (CPU, fast). For a given cache:
  - sweep top-k in {5,10,20,50,100,200, ALL-heads-as-whole-L2}
  - with/without StandardScaler
  - selection metric L1-perhead val-AUC
Reports test AUC. Heads selected on val, ensemble trained on train, eval on test.

Usage:
  python scripts/sweep_topk.py --cache results/cache/q500_instruct      # in-domain
  python scripts/sweep_topk.py --cache results/cache/scifact_ood        # SciFact-adapted
"""
import os, sys, argparse
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

N_JOBS = max(1, min(48, (os.cpu_count() or 4) - 2))


def lr(C=1.0, penalty="l2"):
    return LogisticRegression(penalty=penalty, C=C,
                              solver="lbfgs" if penalty == "l2" else "liblinear",
                              max_iter=2000, class_weight="balanced", random_state=42)


def load(cache, split):
    d = torch.load(os.path.join(cache, f"attn_{split}.pt"))
    return d["acts"].to(torch.float32).numpy(), d["labels"].numpy()


def auc(y, s):
    try: return roc_auc_score(y, s)
    except ValueError: return 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/cache/q500_instruct")
    args = ap.parse_args()
    Xtr, ytr = load(args.cache, "train")
    Xva, yva = load(args.cache, "val")
    Xte, yte = load(args.cache, "test")
    n_heads = Xtr.shape[1]
    print(f"cache={args.cache} train={Xtr.shape} test={Xte.shape}")

    # per-head val AUC ranking (shared across top-k values)
    def hauc(h):
        p = lr(penalty="l1").fit(Xtr[:, h, :], ytr)
        return auc(yva, p.predict_proba(Xva[:, h, :])[:, 1])
    vaucs = np.array(Parallel(n_jobs=N_JOBS)(delayed(hauc)(h) for h in range(n_heads)))
    order = np.argsort(vaucs)[::-1]

    print(f"\n{'config':28s} {'test_AUC':>9s}")
    print("-"*40)
    for k in [5, 10, 20, 50, 100, 200]:
        sel = order[:k]
        def feats(X): return np.concatenate([X[:, h, :] for h in sel], axis=1)
        for scale, tag in [(False, "raw"), (True, "scaled")]:
            model = make_pipeline(StandardScaler(), lr()) if scale else lr()
            model.fit(feats(Xtr), ytr)
            a = auc(yte, model.predict_proba(feats(Xte))[:, 1])
            print(f"top{k:<4d} L2 {tag:7s}{'':10s} {a:9.3f}")
    # whole-activation strong-L2, no selection
    def flat(X): return X.reshape(X.shape[0], -1)
    for C, scale in [(0.01, True), (0.1, True), (1.0, True)]:
        model = make_pipeline(StandardScaler(), lr(C=C))
        model.fit(flat(Xtr), ytr)
        a = auc(yte, model.predict_proba(flat(Xte))[:, 1])
        print(f"whole-L2 C={C:<5g} scaled       {a:9.3f}")
    # mean-probe: average all head activations -> 128d
    def meanf(X): return X.mean(axis=1)
    model = make_pipeline(StandardScaler(), lr())
    model.fit(meanf(Xtr), ytr)
    print(f"mean-probe(128d) scaled      {auc(yte, model.predict_proba(meanf(Xte))[:,1]):9.3f}")


if __name__ == "__main__":
    main()
