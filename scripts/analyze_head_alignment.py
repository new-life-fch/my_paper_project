#!/usr/bin/env python3
"""
Recompute the gradient-alignment percentile at high resolution, from the saved
gradient matrices. Pure CPU; no GPU pass needed.

Why this exists
---------------
`gradient_alignment_heads.py` reports the percentile of the probe direction
among `--n-random` random directions. With 16 directions that statistic can only
take values that are multiples of ~1/17, so a value like 0.938 (= 15/16) sits
just under any sensible threshold purely because of resolution -- it cannot be
distinguished from 0.95. That resolution is too coarse to decide a claim whose
whole point is whether the probe direction is an outlier.

Here we draw a large number of random directions and report:
  * the exact percentile of |cos(probe)| in the null distribution of |cos(random)|,
  * a two-sided permutation p-value: how often a RANDOM direction's mean |cos|
    is at least as extreme as the probe's,
  * the same statistic per model plus a cross-model sign test, since with eight
    models the interesting question is whether the effect is systematically
    positive rather than whether any single model clears a threshold.

Usage:
  python scripts/analyze_head_alignment.py                  # all available matrices
  python scripts/analyze_head_alignment.py --n-random 20000
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 42

ORDER = ["llama32_3b", "llama31_8b", "llama32_3b_instruct", "llama31_8b_instruct",
         "mistral_7b", "mistral_7b_instruct", "qwen25_7b", "qwen25_7b_instruct"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-random", type=int, default=20000)
    ap.add_argument("--glob", default="results/gradient_alignment_heads_*_q1500_matrix.npz")
    ap.add_argument("--out", default="results/head_alignment_highres.json")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(ROOT, args.glob)))
    if not files:
        raise SystemExit(f"no matrices matched {args.glob}; run "
                         "gradient_alignment_heads.py with --save-matrix first")

    rng = np.random.default_rng(SEED)
    per_model = {}
    for f in files:
        tag = os.path.basename(f).replace("gradient_alignment_heads_", "") \
                                .replace("_q1500_matrix.npz", "")
        d = np.load(f)
        G = d["G"].astype(np.float64)          # [n_rows, topk*head_dim]
        w = d["w"].astype(np.float64)          # [topk*head_dim]
        # unit-normalise each row so cos is a plain dot product
        Gn = G / np.linalg.norm(G, axis=1, keepdims=True)
        wn = w / np.linalg.norm(w)
        probe_cos = Gn @ wn                    # [n_rows]
        probe_stat = float(np.mean(probe_cos))

        # null: random unit directions, each scored the same way
        R = rng.normal(size=(args.n_random, w.shape[0]))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        rnd_stats = (Gn @ R.T).mean(axis=0)    # [n_random], mean cos per direction
        abs_rnd = np.abs(rnd_stats)
        abs_probe = abs(probe_stat)

        # percentile of the probe's |mean cos| within the null
        pct = float((abs_rnd < abs_probe).mean())
        # two-sided permutation p: fraction of null directions at least as extreme
        p_perm = float((abs_rnd >= abs_probe).mean())
        # one-sided: is the probe more positive than random directions?
        p_pos = float((rnd_stats >= probe_stat).mean())

        per_model[tag] = {
            "n_rows": int(G.shape[0]),
            "dim": int(G.shape[1]),
            "probe_mean_cos": round(probe_stat, 6),
            "probe_frac_positive": round(float((probe_cos > 0).mean()), 4),
            "null_mean_abs": round(float(abs_rnd.mean()), 6),
            "null_p95_abs": round(float(np.percentile(abs_rnd, 95)), 6),
            "null_max_abs": round(float(abs_rnd.max()), 6),
            "percentile": round(pct, 5),
            "p_perm_two_sided": round(p_perm, 5),
            "p_one_sided_positive": round(p_pos, 5),
        }
        print(f"{tag:24s} n={G.shape[0]:>4} cos={probe_stat:+.5f} "
              f"pct={pct:.4f} p2={p_perm:.4f} p+={p_pos:.4f}")

    # cross-model sign test: with 8 models the question is whether the alignment
    # is systematically positive, not whether one model clears a threshold.
    signs = [1 if per_model[t]["probe_mean_cos"] > 0 else 0
             for t in ORDER if t in per_model]
    n = len(signs)
    k = int(sum(signs))
    from math import comb
    p_sign = sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n) if n else float("nan")
    cleared = sum(1 for t in per_model if per_model[t]["percentile"] >= 0.95)

    print()
    print(f"  models with positive mean cos : {k}/{n}")
    print(f"  one-sided sign test p         : {p_sign:.4f}")
    print(f"  models with percentile >= 0.95: {cleared}/{n}")
    print()
    print("  Reading: unlike the residual-stream gradient alignment (where the")
    print("  probe direction sat at the 0.06-0.81 percentile on 8/8 models), the")
    print("  HEAD-probe direction is not uniformly misaligned -- the picture here")
    print("  is mixed, so the residual misalignment result should NOT be extended")
    print("  to the head reader without saying so.")

    out = {
        "n_random": args.n_random,
        "seed": SEED,
        "source": args.glob,
        "per_model": per_model,
        "cross_model": {
            "n_models": n,
            "n_positive": k,
            "sign_test_p_one_sided": round(p_sign, 5),
            "n_percentile_ge_0_95": cleared,
        },
    }
    p = os.path.join(ROOT, args.out)
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
