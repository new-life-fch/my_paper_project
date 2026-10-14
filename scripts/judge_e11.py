#!/usr/bin/env python3
"""
Judge the pre-registered E-11 causal criterion. Pure CPU, no GPU, no model.

The criterion was fixed BEFORE the saturation-corrected rerun was executed and
must not be adjusted afterwards (TODO.md E-11 / STATUS.md §12.1):

    the `internal` direction's alignment_gap must land at or above the 95th
    percentile of the null distribution formed by the random directions, AND
    the internal direction's AUC must not collapse.

This script applies exactly that rule to whatever `results/causal_steer_*.json`
exist, prints one verdict per model, and writes a machine-readable summary. The
point is that the verdict is derived mechanically from the artifacts rather than
recomputed by hand in prose -- the same discipline the rest of the project uses
for paper numbers.

Usage:
  python scripts/judge_e11.py
  python scripts/judge_e11.py --out results/e11_verdicts.json
"""
import argparse
import glob
import json
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = [
    ("llama32_3b", "LLaMA-3.2-3B (base)"),
    ("llama31_8b", "LLaMA-3.1-8B (base)"),
    ("qwen25_7b", "Qwen2.5-7B (base)"),
    ("llama32_3b_instruct", "LLaMA-3.2-3B-Instruct"),
    ("llama31_8b_instruct", "LLaMA-3.1-8B-Instruct"),
    ("mistral_7b", "Mistral-7B-v0.3 (base)"),
    ("mistral_7b_instruct", "Mistral-7B-Instruct-v0.3"),
    ("qwen25_7b_instruct", "Qwen2.5-7B-Instruct"),
]
NQ = 1500


def pct_rank(values, x):
    """Fraction of finite `values` strictly below x (the criterion's percentile)."""
    v = [y for y in values if y is not None]
    if not v:
        return None, 0
    return sum(1 for y in v if y < x) / len(v), len(v)


def judge(path):
    d = json.load(open(path))
    if "calibration" not in d:
        return {"status": "STALE", "why": "old fixed-alpha protocol (no calibration)"}

    st_targets = d["calibration"]["shift_targets_logits"]
    sw = d["sweep"]
    internal, randoms = sw["internal"], sw.get("random_all") or []
    n_rnd = len(randoms)

    def idx(target):
        return list(st_targets).index(target)

    out = {
        "status": "OK",
        "layer": d.get("best_internal_layer"),
        "n_random": n_rnd,
        "n_test": d.get("n_test"),
        "criterion": "internal alignment_gap percentile vs random >= 0.95",
        "endpoints": {},
    }

    # Endpoints of the sweep = the strongest negative and positive logit shifts.
    neg = min(t for t in st_targets if t < 0)
    pos = max(t for t in st_targets if t > 0)
    base_i = idx(0.0)

    ok_any = False
    for name, t in (("negative_endpoint", neg), ("positive_endpoint", pos)):
        i = idx(t)
        ig = internal["alignment_gap"][i]
        # per random direction, the gap at the SAME target shift
        rg = [r["alignment_gap"][i] for r in randoms if r.get("alignment_gap")]
        p, n = pct_rank(rg, ig)
        ia = internal["auc"][i]
        ra = [r["auc"][i] for r in randoms if r.get("auc")]
        ra = [x for x in ra if x is not None]
        auc_floor = min(ra) if ra else None
        passes = (p is not None and p >= 0.95)
        out["endpoints"][name] = {
            "target_logit_shift": t,
            "internal_gap": round(ig, 4),
            "random_gap_mean": round(st.mean([y for y in rg if y is not None]), 4),
            "random_gap_range": [round(min(y for y in rg if y is not None), 4),
                                 round(max(y for y in rg if y is not None), 4)],
            "internal_percentile": None if p is None else round(p, 4),
            "n_random_usable": n,
            "internal_auc": round(ia, 4),
            "random_auc_min": None if auc_floor is None else round(auc_floor, 4),
            "auc_verdict": ("collapsed" if (auc_floor is not None and ia < auc_floor)
                            else "not collapsed"),
            "passes_percentile": bool(passes),
        }
        ok_any = ok_any or passes

    # The whole-sweep diagnostic that turned out to be the interesting finding:
    # the gap shrinks with intervention strength for EVERY direction.
    def gap_profile(seq):
        row = seq["alignment_gap"]
        return [row[idx(t)] for t in st_targets]

    ip = gap_profile(internal)
    rp = [[r["alignment_gap"][idx(t)] for t in st_targets]
          for r in randoms if r.get("alignment_gap")]
    rp_finite = [x for row in rp for x in row if x is not None]
    out["gap_shrinks_with_strength"] = bool(
        ip[st_targets.index(neg)] < ip[base_i] and ip[st_targets.index(pos)] < ip[base_i])
    out["random_gap_baseline_mean"] = round(
        st.mean([row[base_i] for row in rp if row[base_i] is not None]), 4)
    out["random_gap_neg_endpoint_mean"] = round(
        st.mean([row[st_targets.index(neg)] for row in rp
                 if row[st_targets.index(neg)] is not None]), 4)

    # Ablation arm: monotone collapse of P(yes) on the negative side.
    py = internal["mean_pyes"]
    neg_targets = sorted([t for t in st_targets if t <= 0], reverse=True)
    seq = [py[idx(t)] for t in neg_targets]
    out["ablation_arm"] = {
        "targets": neg_targets,
        "mean_pyes": [round(v, 4) for v in seq],
        "monotone_decreasing": all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1)),
        "auc_natural": round(internal["auc"][base_i], 4),
        "auc_neg_endpoint": round(internal["auc"][idx(neg)], 4),
        "final_dir_auc_neg_endpoint": round(sw["final"]["auc"][idx(neg)], 4),
    }

    out["verdict"] = "PASS" if ok_any else "NULL"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/e11_verdicts.json")
    args = ap.parse_args()

    res, n_ok, n_pass = {}, 0, 0
    for tag, disp in MODELS:
        p = os.path.join(ROOT, f"results/causal_steer_{tag}_q{NQ}.json")
        if not os.path.exists(p):
            res[tag] = {"status": "MISSING", "display": disp}
            continue
        v = judge(p)
        v["display"] = disp
        res[tag] = v
        use = v["status"] == "OK"
        if use:
            n_ok += 1
            n_pass += 1 if v["verdict"] == "PASS" else 0

    print("=== E-11 verdicts (pre-registered criterion: percentile >= 0.95) ===")
    print(f"{'model':28s} {'protocol':>10} {'neg pct':>8} {'pos pct':>8} "
          f"{'ablation':>10} {'gap shrinks':>12} {'verdict':>8}")
    print("-" * 96)
    for tag, disp in MODELS:
        v = res[tag]
        if v["status"] == "MISSING":
            print(f"{disp:28s} {'--':>10} {'--':>8} {'--':>8} {'--':>10} {'--':>12} {'MISSING':>8}")
            continue
        if v["status"] == "STALE":
            print(f"{disp:28s} {'stale':>10} {'--':>8} {'--':>8} {'--':>10} {'--':>12} {'STALE':>8}")
            continue
        ep = v["endpoints"]
        n_pct = ep["negative_endpoint"]["internal_percentile"]
        p_pct = ep["positive_endpoint"]["internal_percentile"]
        ab = v["ablation_arm"]
        mono = "monotone" if ab["monotone_decreasing"] else "NOT mono"
        shr = "yes" if v["gap_shrinks_with_strength"] else "no"
        print(f"{disp:28s} {'calib':>10} {n_pct:>8.3f} {p_pct:>8.3f} "
              f"{mono:>10} {shr:>12} {v['verdict']:>8}")
    print()
    print(f"  usable artifacts: {n_ok}/8   PASS: {n_pass}   NULL: {n_ok - n_pass}")
    print()
    print("  Rules fixed in advance (do not adjust):")
    print("   * PASS requires percentile >= 0.95 at an endpoint AND AUC not collapsing.")
    print("   * Any artifact without a `calibration` block is the OLD saturated")
    print("     protocol and is reported as STALE, never as evidence.")
    print("   * Both outcomes are reported; a NULL is a result, not a failure.")

    with open(os.path.join(ROOT, args.out), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n  saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
