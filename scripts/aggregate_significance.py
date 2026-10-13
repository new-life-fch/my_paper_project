#!/usr/bin/env python3
"""
Cross-model significance summary + Benjamini-Hochberg correction (TODO.md E-7).

Why this exists
---------------
The paper reports per-model one-sided p-values for four edges on eight models
(and, in the older version, on two sample sizes). That is up to 32 hypothesis
tests with **no multiple-comparison control**, and several reported cells sit
right at p ~ 0.05 (e.g. Mistral-7B-Instruct attn-final, p = 0.0526 in the new
pairwise-bootstrap artifact). Under any correction those cells cannot be called
significant. The sign test across models (p = 0.016) is unaffected.

This script reads the per-model significance JSONs produced by
`scripts/significance.py` (one file per model -- that layout is deliberate:
each model draws its own spawned RNG sub-stream, so a model's CI does not
depend on which other models ran, see TODO.md P1-17) and emits:

  * a tidy summary table (per model x edge: mean, CI, p, q)
  * BH q-values computed WITHIN each family (edge type across models), which is
    the right granularity: "does this edge hold across the family?"
  * a cross-model sign test per edge (direction consistency), which is the
    claim the paper actually leans on, and which needs no per-model correction

Usage:
  python scripts/aggregate_significance.py --suffix _q1500 \
      --out results/significance_summary_q1500.json
  # or point at explicit files
  python scripts/aggregate_significance.py --files a.json b.json --out out.json
"""
import argparse
import glob
import json
import os
import sys
from math import comb

EDGES = ("internal_edge(best_resid-final)", "train_edge(final-judge)",
         "attn-final", "best_resid-judge")
# short labels for tables
SHORT = {"internal_edge(best_resid-final)": "internal",
         "train_edge(final-judge)": "train",
         "attn-final": "attn-final",
         "best_resid-judge": "resid-judge"}


def bh(pvals):
    """Benjamini-Hochberg q-values (monotone, capped at 1.0)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        k = m - rank + 1                      # 1-based rank from the largest p
        val = min(prev, pvals[i] * m / k)
        q[i] = val
        prev = val
    return q


def sign_test_p(directions):
    """Two-sided sign test on a list of +1/-1 (ties dropped)."""
    d = [x for x in directions if x != 0]
    n = len(d)
    if n == 0:
        return 1.0, 0
    k = max(sum(1 for x in d if x > 0), sum(1 for x in d if x < 0))
    p = sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n) * 2
    return min(1.0, p), n


def load_records(paths, strict=True):
    """Load per-model significance records.

    `strict` drops records that were NOT produced by the current
    scripts/significance.py.  This matters because this repo also carries
    artifacts from the now-dead original host; those used a single shared RNG
    stream and an unpaired bootstrap, so mixing them into one BH family would
    silently compare p-values computed under two different nulls.
    Current-format records carry `n_test` + `n_boot` + `n_layers`; the legacy
    aggregate files (`significance.json`, `significance_qwen_q1500.json`, ...)
    do not.
    """
    recs, skipped = [], []
    for p in paths:
        try:
            d = json.load(open(p))
        except Exception as e:
            print(f"  skip {p}: {e}", file=sys.stderr)
            continue
        items = d if isinstance(d, list) else [d]
        for it in items:
            it["_file"] = os.path.basename(p)
            modern = all(k in it for k in ("n_test", "n_boot", "n_layers", "tag"))
            if strict and not modern:
                skipped.append(it.get("tag", it["_file"]))
                continue
            recs.append(it)
    if skipped:
        print(f"  SKIPPED {len(skipped)} legacy record(s) not from the current code: "
              f"{', '.join(sorted(set(skipped)))}", file=sys.stderr)
    return recs, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--suffix", default="_q1500",
                    help="glob results/significance_*<suffix>.json when --files is absent")
    ap.add_argument("--include-legacy", action="store_true",
                    help="also mix in pre-2026-10-13 artifacts (different bootstrap; NOT comparable)")
    ap.add_argument("--out", default="results/significance_summary.json")
    args = ap.parse_args()

    if args.files:
        paths = args.files
    else:
        # per-model files first; fall back to the legacy aggregate files
        paths = sorted(glob.glob(f"results/significance_*{args.suffix}.json"))
        paths = [p for p in paths if "summary" not in p]
    if not paths:
        raise SystemExit(f"no significance files matched (suffix={args.suffix})")
    print(f"reading {len(paths)} file(s):")
    for p in paths:
        print(f"  {p}")

    recs, skipped = load_records(paths, strict=not args.include_legacy)
    recs = [r for r in recs if "edges" in r and "tag" in r]
    if not recs:
        raise SystemExit("no usable records (need per-model files with 'edges')")

    # ---- collect (model, edge) -> p, and apply BH within each edge family ----
    by_edge = {e: [] for e in EDGES}
    for r in recs:
        for e in EDGES:
            if e in r["edges"]:
                by_edge[e].append((r["tag"], r["edges"][e]))

    summary = {}
    for e, rows in by_edge.items():
        if not rows:
            continue
        pvals = [v["p_le0"] for _, v in rows]
        qvals = bh(pvals)
        directions = [1 if v["mean"] > 0 else (-1 if v["mean"] < 0 else 0) for _, v in rows]
        sp, sn = sign_test_p(directions)
        summary[SHORT[e]] = {
            "edge": e,
            "n_models": len(rows),
            "per_model": [
                {"tag": t, "mean": v["mean"], "ci95": v["ci95"], "p_le0": v["p_le0"],
                 "q_bh": round(q, 4),
                 "ci_excludes_0": bool(v["ci95"][0] > 0 or v["ci95"][1] < 0)}
                for (t, v), q in zip(rows, qvals)
            ],
            "n_ci_excludes_0": sum(1 for _, v in rows if v["ci95"][0] > 0 or v["ci95"][1] < 0),
            "n_q_bh_lt_05": sum(1 for q in qvals if q < 0.05),
            "sign_test": {"n": sn, "p_two_sided": round(sp, 4),
                          "consistent_direction": sum(directions) == sn or -sum(directions) == sn},
        }

    meta = {
        "files": [os.path.basename(p) for p in paths],
        "suffix": args.suffix,
        "n_records": len(recs),
        "n_legacy_skipped": len(skipped),
        "tags": sorted({r["tag"] for r in recs}),
        "note": ("p_le0 is a one-sided bootstrap p from scripts/significance.py; "
                 "q_bh is Benjamini-Hochberg within the edge family (across models). "
                 "Per-model files are used because each model draws its own RNG "
                 "sub-stream, so a model's CI is independent of what else ran "
                 "(TODO.md P1-17)."),
    }
    rep = {"meta": meta, "edges": summary}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(rep, open(args.out, "w"), indent=2)

    # ---- human-readable table ------------------------------------------------
    print(f"\n=== {len(recs)} model(s): {', '.join(meta['tags'])} ===")
    hdr = (f"{'edge':13s} {'#CI>0':>6} {'#q<.05':>7} {'sign-test p':>12} {'consistent':>11}   per-model p -> q")
    print(hdr)
    print("-" * len(hdr))
    for k, v in summary.items():
        ps = " ".join(f"{r['tag'].replace('llama','L').replace('_','')}:{r['p_le0']:.3f}->{r['q_bh']:.3f}"
                      for r in v["per_model"])
        print(f"{k:13s} {v['n_ci_excludes_0']:>4}/{v['n_models']:<2} {v['n_q_bh_lt_05']:>4}/{v['n_models']:<3} "
              f"{v['sign_test']['p_two_sided']:>12.4f} {str(v['sign_test']['consistent_direction']):>11}   {ps}")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
