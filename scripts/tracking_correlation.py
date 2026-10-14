#!/usr/bin/env python3
"""
Does the patching "tracking" effect correlate with how far the probe leads the
output-side readers? Pure CPU; reads only result JSONs.

Why this exists
---------------
`paper/main.tex` reports a correlation between the cross-sample patching
tracking statistic (Delta-gap) and the probe's margin over the final-layer probe
/ the judge, and it is used precisely to say that the obvious mechanistic
explanation is NOT supported. A number that decides whether a mechanism claim is
made or withheld must be reproducible from the artifacts, not recomputed by
hand in prose -- so it lives here.

It also prints the pairwise table, because with n<=8 the correlation is easy to
over-read and the reader should see the points (STATUS.md honesty discipline #6:
list every member's value before claiming a grouping).

Usage:
  python scripts/tracking_correlation.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NQ = 1500
ORDER = ["llama32_3b", "llama31_8b", "llama32_3b_instruct", "llama31_8b_instruct",
         "mistral_7b", "mistral_7b_instruct", "qwen25_7b", "qwen25_7b_instruct"]


def pearson(a, b):
    n = len(a)
    if n < 3:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else float("nan")


def main():
    rows = []
    for tag in ORDER:
        p = os.path.join(ROOT, f"results/patch_v2_{tag}_q{NQ}.json")
        s = os.path.join(ROOT, f"results/significance_{tag}_q{NQ}.json")
        if not (os.path.exists(p) and os.path.exists(s)):
            continue
        pd = json.load(open(p))
        sg = json.load(open(s))
        sg = sg[0] if isinstance(sg, list) else sg
        sw = pd["sweep"]
        a0, a1 = sw["cross_Lstar@0.0"], sw["cross_Lstar@1.0"]
        dgap = a1["follow_gap_vs_source"] - a0["follow_gap_vs_source"]
        probe = pd["probe_test_auc_at_Lstar"]
        rows.append({
            "tag": tag,
            "L": pd["layer"],
            "probe_at_Lstar": round(probe, 4),
            "final": sg["auc"]["final"]["point"],
            "judge": sg["auc"]["judge"]["point"],
            "probe_minus_final": round(probe - sg["auc"]["final"]["point"], 4),
            "probe_minus_judge": round(probe - sg["auc"]["judge"]["point"], 4),
            "delta_gap": round(dgap, 4),
            "auc_vs_target_at_1": a1["auc_vs_target"],
            "norm_matched_vs_raw_at_1": [
                sw["final_normmatched@1.0"]["auc_vs_target"],
                sw["final_raw@1.0"]["auc_vs_target"]],
            "identity_max_absdiff": pd["controls_Lstar"]["self_patch_max_absdiff_vs_natural"],
        })

    print(f"{'tag':24s} {'L*':>3} {'probe':>6} {'final':>6} {'judge':>6} "
          f"{'p-final':>8} {'p-judge':>8} {'Dgap':>7}")
    print("-" * 82)
    for r in sorted(rows, key=lambda x: -x["delta_gap"]):
        print(f"{r['tag']:24s} {r['L']:>3} {r['probe_at_Lstar']:>6.3f} {r['final']:>6.3f} "
              f"{r['judge']:>6.3f} {r['probe_minus_final']:>+8.4f} "
              f"{r['probe_minus_judge']:>+8.4f} {r['delta_gap']:>+7.4f}")

    dg = [r["delta_gap"] for r in rows]
    pf = [r["probe_minus_final"] for r in rows]
    pj = [r["probe_minus_judge"] for r in rows]
    print()
    print(f"  n = {len(rows)}")
    print(f"  corr(probe - final, Delta-gap) = {pearson(pf, dg):+.3f}")
    print(f"  corr(probe - judge, Delta-gap) = {pearson(pj, dg):+.3f}")
    print()
    print("  Reading: a strong positive correlation would support 'tracking is weaker")
    print("  where the probe leads more'. It is not there, so the paper reports the")
    print("  patching pattern as model-dependent and leaves the mechanism open.")
    print(f"  With n={len(rows)} this correlation could not settle the question even if")
    print("  it were large; it is reported to show we looked, not as a finding.")

    # the two sanity checks that must hold everywhere, reported alongside
    worst_id = max(abs(r["identity_max_absdiff"]) for r in rows)
    worst_nm = max(abs(r["norm_matched_vs_raw_at_1"][0] - r["norm_matched_vs_raw_at_1"][1])
                   for r in rows)
    print()
    print(f"  sanity: worst identity max|delta| over all models = {worst_id}")
    print(f"  sanity: worst |norm-matched - raw| at alpha=1     = {worst_nm:.5f}")
    if worst_id > 1e-3:
        print("  !! identity control failed on some model -- results not trustworthy")
    if worst_nm > 0.005:
        print("  !! norm matching changes the outcome somewhere -- re-examine the")
        print("     'collapse is not a scale artefact' claim per model")

    out = os.path.join(ROOT, "results/tracking_correlation.json")
    json.dump({"n": len(rows), "per_model": rows,
               "corr_probe_minus_final_dgap": round(pearson(pf, dg), 4),
               "corr_probe_minus_judge_dgap": round(pearson(pj, dg), 4),
               "worst_identity_max_absdiff": worst_id,
               "worst_norm_matched_minus_raw": round(worst_nm, 5)},
              open(out, "w"), indent=2)
    print(f"\n  saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
