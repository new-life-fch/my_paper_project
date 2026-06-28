#!/usr/bin/env python3
"""Plot the causal steering sweep (paper figure). Reads causal_steer_<tag>.json."""
import json, argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--json", default="results/causal_steer_llama32_3b_q1500.json")
ap.add_argument("--out", default="results/figures/causal_steer_3b.png")
args = ap.parse_args()

d = json.load(open(args.json))
al = d["alphas"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
colors = {"internal": "#c0392b", "final": "#2980b9", "random": "#7f8c8d"}
labels = {"internal": f"internal dir (L{d['best_internal_layer']})",
          "final": "final-layer dir", "random": "random dir"}

for k in ["internal", "final", "random"]:
    sw = d["sweep"][k]
    ax1.plot(al, sw["mean_pyes"], "o-", color=colors[k], label=labels[k])
    ax2.plot(al, sw["auc"], "o-", color=colors[k], label=labels[k])
ax1.axvline(0, color="k", ls=":", lw=0.8); ax1.axhline(0.793, color="k", ls=":", lw=0.6)
ax1.set_xlabel(r"steering coefficient $\alpha$ (in units of $\|h\|$)")
ax1.set_ylabel("mean judge P(yes)")
ax1.set_title("(a) Verdict shift under output-layer injection")
ax1.legend(fontsize=8)
ax2.axvline(0, color="k", ls=":", lw=0.8)
ax2.set_xlabel(r"steering coefficient $\alpha$")
ax2.set_ylabel("test AUC")
ax2.set_title("(b) Discriminative AUC under injection")
ax2.legend(fontsize=8)
plt.tight_layout()
plt.savefig(args.out, dpi=150, bbox_inches="tight")
print(f"Saved: {args.out}")
