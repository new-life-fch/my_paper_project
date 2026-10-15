#!/usr/bin/env python3
"""
Layer-wise decomposition of the head-probe gradient alignment. Pure CPU; reads
the saved gradient matrices, no GPU pass.

Why this exists
---------------
`gradient_alignment_heads.py` gives one number per model: the cosine between the
verdict's gradient and the whole 20-head probe direction. That summary is
uninformative about *where* in the stack the alignment lives, and it came out
mixed (5/8 positive, 2/8 above the 0.95 threshold, sign test p=0.36). Before
running any causal intervention we need to know whether the mixed result is
noise or structure: if the alignment concentrates in a few heads, a causal test
has a target; if it is spread thinly across all 20, intervening on "the head
probe direction" is intervening on nothing in particular.

What is computed, per head h and per model
------------------------------------------
  contrib_h   = mean |g_h|                     -- how much this head moves the verdict
  cos_h       = mean cos(g_h, w_h)             -- per-head directional agreement
  abscos_h    = mean |cos(g_h, w_h)|
  probe_share = |w_h| / |w|                    -- the probe's own weight on this head

and the same statistics grouped by layer, plus a comparison of the layer-wise
attribution against the layer the residual probe would have chosen (L*).

A caveat stated up front: per-head gradients are noisier than the 20-head
concatenation, so `cos_h` is a low-precision quantity. We therefore lean on
`contrib_h` (a magnitude, robust) for locating leverage and treat `cos_h` as
suggestive only.

Usage:
  python scripts/analyze_head_alignment_layers.py
  python scripts/analyze_head_alignment_layers.py --tag llama31_8b
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HD = 128
TOPK = 20

ORDER = ["llama32_3b", "llama31_8b", "llama32_3b_instruct", "llama31_8b_instruct",
         "mistral_7b", "mistral_7b_instruct", "qwen25_7b", "qwen25_7b_instruct"]


def cos_rows(Gh, wh):
    """Row-wise cosine between Gh [n, d] and wh [d]."""
    ng = np.linalg.norm(Gh, axis=1)
    nw = np.linalg.norm(wh)
    ok = (ng > 0) & (nw > 0)
    out = np.zeros(Gh.shape[0])
    out[ok] = (Gh[ok] @ wh) / (ng[ok] * nw)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="single model; default = all")
    ap.add_argument("--out", default="results/head_alignment_by_layer.json")
    args = ap.parse_args()

    tags = [args.tag] if args.tag else ORDER
    report = {}
    for tag in tags:
        mp = os.path.join(ROOT, f"results/gradient_alignment_heads_{tag}_q1500_matrix.npz")
        jp = os.path.join(ROOT, f"results/gradient_alignment_heads_{tag}_q1500.json")
        if not (os.path.exists(mp) and os.path.exists(jp)):
            print(f"  {tag}: missing artifacts, skipping")
            continue
        m = np.load(mp)
        G = m["G"].astype(np.float64)          # [n_rows, 20*128]
        w = m["w"].astype(np.float64)          # [20*128]
        meta = json.load(open(jp))
        layers = meta["selected_heads_layers"]
        heads = meta["selected_heads_indices"]
        # the probe's own per-head weight (probe is fit on RAW features, so w_h is
        # directly comparable across heads)
        per_head = []
        for k, (li, h) in enumerate(zip(layers, heads)):
            sl = slice(k * HD, (k + 1) * HD)
            Gh = G[:, sl]
            wh = w[sl]
            ch = cos_rows(Gh, wh)
            per_head.append({
                "slot": k, "layer": int(li), "head": int(h),
                "contrib": float(np.abs(Gh).mean()),
                "cos_mean": float(ch.mean()),
                "abscos_mean": float(np.abs(ch).mean()),
                "probe_weight": float(np.linalg.norm(wh)),
                "probe_weight_norm": float(np.linalg.norm(wh) / np.linalg.norm(w)),
            })
        by_layer = {}
        for h_ in per_head:
            d = by_layer.setdefault(h_["layer"], {"contrib": [], "cos": [], "w": [],
                                                  "slots": []})
            d["contrib"].append(h_["contrib"]); d["cos"].append(h_["cos_mean"])
            d["w"].append(h_["probe_weight"]); d["slots"].append(h_["slot"])
        layer_stats = {int(L): {
            "n_heads": len(d["contrib"]),
            "contrib_sum": round(float(np.sum(d["contrib"])), 6),
            "contrib_max": round(float(np.max(d["contrib"])), 6),
            "cos_mean": round(float(np.mean(d["cos"])), 5),
            "probe_weight_share": round(float(np.sum(d["w"]) / np.linalg.norm(w)), 5),
        } for L, d in sorted(by_layer.items())}

        total = sum(v["contrib_sum"] for v in layer_stats.values())
        for L in layer_stats:
            layer_stats[L]["contrib_share"] = round(layer_stats[L]["contrib_sum"] / total, 4)

        # is the alignment concentrated? Herfindahl over heads on contrib and on |cos|
        c = np.array([h_["contrib"] for h_ in per_head]); c = c / c.sum()
        a = np.abs(np.array([h_["cos_mean"] for h_ in per_head]))
        a = a / a.sum() if a.sum() > 0 else a
        rep = {
            "n_rows": int(G.shape[0]),
            "layers": sorted(layer_stats),
            "per_head": per_head,
            "per_layer": layer_stats,
            "concentration": {
                "contrib_herfindahl": round(float((c ** 2).sum()), 4),
                "contrib_top1_share": round(float(c.max()), 4),
                "contrib_top5_share": round(float(np.sort(c)[::-1][:5].sum()), 4),
                "abscos_herfindahl": round(float((a ** 2).sum()), 4),
                "uniform_reference": round(1.0 / len(per_head), 4),
            },
            "head_probe_test_auc": meta["head_probe_test_auc"],
        }
        report[tag] = rep

        print(f"\n{tag}  (head-probe AUC {meta['head_probe_test_auc']}, "
              f"layers {sorted(layer_stats)})")
        print(f"  {'layer':>5} {'#heads':>6} {'contrib_share':>13} {'cos_mean':>9} "
              f"{'probe_w_share':>13}")
        for L, v in layer_stats.items():
            print(f"  {L:>5} {v['n_heads']:>6} {v['contrib_share']:>13.3f} "
                  f"{v['cos_mean']:>+9.4f} {v['probe_weight_share']:>13.3f}")
        print(f"  concentration: contrib top-1 share {c.max():.3f}, top-5 {np.sort(c)[::-1][:5].sum():.3f} "
              f"(uniform = {1/len(per_head):.3f})")
        top = sorted(per_head, key=lambda x: -x["contrib"])[:3]
        print("  highest-leverage heads: " +
              ", ".join(f"L{t['layer']}h{t['head']}({t['contrib']:.2e})" for t in top))

    p = os.path.join(ROOT, args.out)
    json.dump(report, open(p, "w"), indent=2)
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
