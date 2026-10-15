#!/usr/bin/env python3
"""
Rigorous analysis of the head-space causal intervention. Pure CPU; reads the
`pyes` vectors that `causal_steer_heads.py` stores in its JSON.

Why this exists
---------------
The runtime script reports a percentile of the probe's alignment gap among the
random directions' gaps. Two things make that number alone insufficient:

  1. Its resolution is 1/(n_random+1) and is set ONLY by the number of random
     directions -- resampling queries does not sharpen it. With 12 directions the
     statistic cannot take a value between 0.923 and 1.0.
  2. A percentile is not a significance statement. Comparing the probe's gap to
     the *sampling* distribution of a random direction's gap (pooled across
     directions) is what actually tests "this direction is special".

So this script reports, per shift target:
  * the probe's gap and its query-clustered bootstrap CI,
  * each random direction's gap and CI (listed, because with <=48 directions a
    single large random gap can carry an apparent effect),
  * the permutation p-value: P(|gap of a random direction| >= |gap of probe|),
    with both the direction-level null and the query-level pooled null,
  * a sign-consistency check across shift targets: pushing the verdict down
    should lower P(yes) more for relevant rows; a direction that merely adds
    noise gives gaps whose sign does not track the shift direction.

Usage:
  python scripts/analyze_causal_heads.py
  python scripts/analyze_causal_heads.py --tag llama32_3b
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 42

ORDER = ["llama32_3b", "llama31_8b", "llama32_3b_instruct", "llama31_8b_instruct",
         "mistral_7b", "mistral_7b_instruct", "qwen25_7b", "qwen25_7b_instruct"]


def query_index(qids):
    uq = np.unique(qids)
    return uq, {q: np.where(qids == q)[0] for q in uq}


def boot_gap_series(p, y, uq, byq, n=5000, seed=SEED):
    rng = np.random.default_rng(seed)
    pos, neg = y == 1, y == 0
    out = np.empty(n)
    for b in range(n):
        idx = np.concatenate([byq[q] for q in rng.choice(uq, size=len(uq), replace=True)])
        ps, ns = idx[pos[idx]], idx[neg[idx]]
        out[b] = p[ps].mean() - p[ns].mean() if len(ps) and len(ns) else np.nan
    return out[np.isfinite(out)]


def gap_of(p, y):
    return float(p[y == 1].mean() - p[y == 0].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    ap.add_argument("--cache-pattern", default="results/cache/{}_q1500_instruct")
    ap.add_argument("--out", default="results/causal_heads_analysis.json")
    args = ap.parse_args()

    tags = [args.tag] if args.tag else ORDER
    report = {}
    for tag in tags:
        jp = os.path.join(ROOT, f"results/causal_steer_heads_{tag}_q1500.json")
        cp = os.path.join(ROOT, args.cache_pattern.format(tag), "attn_test.pt")
        if not (os.path.exists(jp) and os.path.exists(cp)):
            print(f"  {tag}: missing artifacts, skipping")
            continue
        d = json.load(open(jp))
        cache = torch.load(cp, map_location="cpu")
        y = cache["labels"].numpy()
        q = cache["query_ids"].numpy()
        nrows = d["n_rows"]
        if nrows and nrows < len(y):           # row cap used for the big models
            y, q = y[:nrows], q[:nrows]
        uq, byq = query_index(q)

        rec = {"n_rows": int(len(y)), "dominant_layer": d["dominant_layer"],
               "head_probe_test_auc": d["head_probe_test_auc"],
               "per_shift": {}}
        print(f"\n{tag}  (L{d['dominant_layer']}, {nrows or len(y)} rows, "
              f"head-probe AUC {d['head_probe_test_auc']})")

        for tgt in d["shift_targets"]:
            k = f"{tgt:+.0f}"
            probe_v = d["per_direction"]["head_probe"]["per_target"].get(k)
            if probe_v is None or probe_v.get("degenerate"):
                rec["per_shift"][k] = {"degenerate": True}
                print(f"  shift {k}: probe point degenerate")
                continue
            pv = np.array(probe_v["pyes"])
            rnd = {}
            for name, dr in d["per_direction"].items():
                if not name.startswith("random_"):
                    continue
                v = dr["per_target"].get(k)
                if v is None or v.get("degenerate"):
                    continue
                rnd[name] = np.array(v["pyes"])
            if len(rnd) < 5:
                rec["per_shift"][k] = {"degenerate": True,
                                       "n_random_usable": len(rnd)}
                print(f"  shift {k}: only {len(rnd)} usable random directions")
                continue

            pg = gap_of(pv, y)
            ng = np.array([gap_of(v, y) for v in rnd.values()])
            pb = boot_gap_series(pv, y, uq, byq, n=5000)
            nb = np.concatenate([boot_gap_series(v, y, uq, byq, n=300)
                                 for v in rnd.values()])
            pct = float((np.abs(ng) < abs(pg)).mean())
            p_perm_pooled = float((np.abs(nb) >= abs(pg)).mean())
            p_perm_dir = float((np.abs(ng) >= abs(pg)).mean())

            # does the probe's gap have the sign the shift implies?  A negative
            # logit shift should reduce P(yes) more on relevant rows -> gap < 0.
            expect = -1 if tgt < 0 else 1
            sign_ok = bool(np.sign(pg) == expect)
            # how many random directions also get the "right" sign?
            n_rnd_sign = int((np.sign(ng) == expect).sum())

            rec["per_shift"][k] = {
                "probe_gap": round(pg, 5),
                "probe_ci95": [round(float(np.percentile(pb, 2.5)), 5),
                               round(float(np.percentile(pb, 97.5)), 5)],
                "probe_sign_matches_shift": sign_ok,
                "random_gaps": {n_: round(float(g), 5) for n_, g in zip(rnd, ng)},
                "random_gap_mean": round(float(ng.mean()), 5),
                "random_gap_min": round(float(ng.min()), 5),
                "random_gap_max": round(float(ng.max()), 5),
                "random_sign_matches_shift": f"{n_rnd_sign}/{len(ng)}",
                "percentile": round(pct, 4),
                "percentile_resolution": round(1.0 / (len(ng) + 1), 4),
                "p_perm_direction_level": round(p_perm_dir, 4),
                "p_perm_pooled_null": round(p_perm_pooled, 4),
                "n_random_usable": int(len(ng)),
            }
            print(f"  shift {k}: probe {pg:+.4f} "
                  f"CI[{rec['per_shift'][k]['probe_ci95'][0]:+.4f},"
                  f"{rec['per_shift'][k]['probe_ci95'][1]:+.4f}]  "
                  f"sign_ok={sign_ok}")
            print(f"            random: mean {ng.mean():+.4f} "
                  f"range [{ng.min():+.4f},{ng.max():+.4f}]  "
                  f"sign_ok {n_rnd_sign}/{len(ng)}")
            print(f"            percentile {pct:.4f} (res {1/(len(ng)+1):.4f})  "
                  f"p_perm dir-level {p_perm_dir:.4f}  pooled {p_perm_pooled:.4f}")

        # cross-shift consistency: the paper-relevant question is whether the
        # probe's gap flips sign with the shift direction, which random noise
        # does not do reliably.
        ks = [k for k, v in rec["per_shift"].items() if not v.get("degenerate")]
        if len(ks) == 2:
            g = [rec["per_shift"][k]["probe_gap"] for k in ks]
            rec["sign_consistency"] = {
                "gaps": dict(zip(ks, g)),
                "probe_flips_sign": bool(np.sign(g[0]) != np.sign(g[1])),
                "both_perm_p_below_0.05": bool(all(
                    rec["per_shift"][k]["p_perm_pooled_null"] < 0.05 for k in ks)),
                "any_perm_p_below_0.05": bool(any(
                    rec["per_shift"][k]["p_perm_pooled_null"] < 0.05 for k in ks)),
            }
            sc = rec["sign_consistency"]
            print(f"  cross-shift: probe flips sign = {sc['probe_flips_sign']}, "
                  f"any p<0.05 = {sc['any_perm_p_below_0.05']}, "
                  f"both p<0.05 = {sc['both_perm_p_below_0.05']}")
        report[tag] = rec

    # cross-model summary
    if len(report) > 1:
        n_any = sum(1 for v in report.values()
                    if v.get("sign_consistency", {}).get("any_perm_p_below_0.05"))
        n_both = sum(1 for v in report.values()
                     if v.get("sign_consistency", {}).get("both_perm_p_below_0.05"))
        n_flip = sum(1 for v in report.values()
                     if v.get("sign_consistency", {}).get("probe_flips_sign"))
        print(f"\n跨模型：任何位移 p<0.05 的模型 {n_any}/{len(report)}；"
              f"两个位移都 p<0.05 的 {n_both}/{len(report)}；"
              f"探针 gap 随位移翻转符号的 {n_flip}/{len(report)}")
        report["_summary"] = {"n_models": len(report), "n_any_p_lt_05": n_any,
                             "n_both_p_lt_05": n_both, "n_sign_flip": n_flip}

    p = os.path.join(ROOT, args.out)
    json.dump(report, open(p, "w"), indent=2)
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
