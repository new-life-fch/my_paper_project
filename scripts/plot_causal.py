#!/usr/bin/env python3
"""
Causal steering figure (paper Fig. 2). Reads causal_steer_<tag>.json.

Updated 2026-10-13 for the rewritten experiment (TODO.md P0-7 / E-1):
  * `sweep["random"]` became `sweep["random_all"]` -- a LIST of >=10 random
    directions, plotted as a mean curve with a min-max band instead of one draw.
  * the horizontal baseline is read from the JSON (mean P(yes) at alpha=0)
    instead of the hard-coded 0.793, which was only correct for the 3B run
    (TODO.md P3-7).
  * panel (c) plots the new ALIGNMENT GAP (mean P(yes)|relevant - |irrelevant).
    This is the diagnostic the old artifact could not measure: a blunt shift
    moves both groups together and leaves the gap flat, so only a widening gap
    is evidence that the injected direction carries relevance.
"""
import json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--json", default="results/causal_steer_llama32_3b_q1500.json")
ap.add_argument("--out", default=None)
ap.add_argument("--tag", default=None)
args = ap.parse_args()

d = json.load(open(args.json))
al = d["alphas"]
tag = args.tag or d.get("tag", "model")
out = args.out or f"results/figures/causal_steer_{tag}.png"

COL = {"internal": "#c0392b", "final": "#2980b9", "random": "#7f8c8d"}
has_random_list = isinstance(d["sweep"].get("random_all"), list)

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))
ax1, ax2, ax3 = axes

for k in ("internal", "final"):
    sw = d["sweep"][k]
    lab = (f"internal dir (L{d['best_internal_layer']})" if k == "internal"
           else "final-layer dir")
    ax1.plot(al, sw["mean_pyes"], "o-", color=COL[k], label=lab)
    ax2.plot(al, sw["auc"], "o-", color=COL[k], label=lab)
    ax3.plot(al, sw["alignment_gap"], "o-", color=COL[k], label=lab)

if has_random_list:
    rnd = d["sweep"]["random_all"]
    mp = np.array([r["mean_pyes"] for r in rnd])
    au = np.array([r["auc"] for r in rnd])
    gp = np.array([r["alignment_gap"] for r in rnd])
    n = mp.shape[0]
    for ax, arr in ((ax1, mp), (ax2, au), (ax3, gp)):
        ax.plot(al, arr.mean(0), "s--", color=COL["random"],
                label=f"{n} random dirs (mean)")
        ax.fill_between(al, arr.min(0), arr.max(0), color=COL["random"], alpha=0.18,
                        label="random min-max")

base = d["sweep"]["internal"]["mean_pyes"][al.index(0.0)] if 0.0 in al else None
if base is not None:
    ax1.axhline(base, color="k", ls=":", lw=0.8)
    ax1.annotate(f"natural P(yes)={base:.3f}", xy=(al[0], base), xytext=(2, 6),
                 textcoords="offset points", fontsize=7)

for ax in axes:
    ax.axvline(0, color="k", ls=":", lw=0.8)
ax1.set_xlabel(r"steering coefficient $\alpha$ (units of $\|h\|$)")
ax1.set_ylabel("mean judge P(yes)")
ax1.set_title("(a) Verdict shift under output-layer injection")
ax2.set_xlabel(r"steering coefficient $\alpha$")
ax2.set_ylabel("test AUC")
ax2.set_title("(b) Discriminative AUC under injection")
ax3.set_xlabel(r"steering coefficient $\alpha$")
ax3.set_ylabel(r"gap: P(yes)$|$rel $-$ P(yes)$|$irrel")
ax3.set_title("(c) Relevance-ALIGNED movement (the real diagnostic)")
ax3.axhline(0, color="k", lw=0.6)
for ax in axes:
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")

# one-line summary for the master log
sw = d["sweep"]
print(f"  internal: AUC {sw['internal']['auc'][0]:.3f}->{sw['internal']['auc'][-1]:.3f}, "
      f"gap {sw['internal']['alignment_gap'][0]:+.3f}->{sw['internal']['alignment_gap'][-1]:+.3f}")
print(f"  final   : AUC {sw['final']['auc'][0]:.3f}->{sw['final']['auc'][-1]:.3f}, "
      f"gap {sw['final']['alignment_gap'][0]:+.3f}->{sw['final']['alignment_gap'][-1]:+.3f}")
if has_random_list:
    print(f"  random  : {len(rnd)} dirs, gap range at min(alpha) "
          f"[{min(r['alignment_gap'][0] for r in rnd):+.3f},"
          f"{max(r['alignment_gap'][0] for r in rnd):+.3f}]")
