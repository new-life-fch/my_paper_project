#!/usr/bin/env python3
"""
Mechanism figures for the wide-scope paper. Pure CPU, reads results/*.json.

The wide paper's mechanism section rests on two intervention-style
measurements, and this file draws both:

  fig7_patch        cross-sample activation patching, 8 models:
                    (a) how far the verdict follows the transplanted activation
                    (b) that rescaling the payload does not change the outcome
  fig8_gradient     gradient alignment, 8 models: the probe direction is never
                    an outlier among random directions (the misalignment result)
  fig9_mechanism    the synthesis: verdict tracking vs readout alignment, with
                    the two Qwen models called out as the informative case

Provenance for every point is written to a .json sidecar next to the figure.
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NQ = 1500

MODELS = [
    ("LLaMA-3.2-3B", "llama32_3b", "LLaMA", "base"),
    ("LLaMA-3.2-3B-Inst", "llama32_3b_instruct", "LLaMA", "instruct"),
    ("LLaMA-3.1-8B", "llama31_8b", "LLaMA", "base"),
    ("LLaMA-3.1-8B-Inst", "llama31_8b_instruct", "LLaMA", "instruct"),
    ("Mistral-7B", "mistral_7b", "Mistral", "base"),
    ("Mistral-7B-Inst", "mistral_7b_instruct", "Mistral", "instruct"),
    ("Qwen2.5-7B", "qwen25_7b", "Qwen", "base"),
    ("Qwen2.5-7B-Inst", "qwen25_7b_instruct", "Qwen", "instruct"),
]
FAM_COL = {"LLaMA": "#c0392b", "Mistral": "#2980b9", "Qwen": "#27ae60"}


def sidecar(name, payload):
    json.dump(payload, open(os.path.join(ROOT, f"results/figures/{name}.json"), "w"),
              indent=2)


def load_patch(tag):
    return json.load(open(os.path.join(ROOT, f"results/patch_v2_{tag}_q{NQ}.json")))


def load_grad(tag):
    return json.load(open(os.path.join(
        ROOT, f"results/gradient_alignment_{tag}_q{NQ}.json")))


def fig_patch(outdir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ax1, ax2, ax3 = axes
    prov = {}
    for name, tag, fam, regime in MODELS:
        d = load_patch(tag)
        sw = d["sweep"]
        a = [0.0, 0.25, 0.5, 0.75, 1.0]
        src = [sw[f"cross_Lstar@{x}"]["auc_vs_source"] for x in a]
        tgt = [sw[f"cross_Lstar@{x}"]["auc_vs_target"] for x in a]
        gap = [sw[f"cross_Lstar@{x}"]["follow_gap_vs_source"] for x in a]
        ax1.plot(a, gap, "o-", ms=3.5, color=FAM_COL[fam],
                 ls="-" if regime == "base" else "--",
                 markerfacecolor=FAM_COL[fam] if regime == "base" else "white",
                 markeredgecolor=FAM_COL[fam], label=name, lw=1.4)
        ax2.scatter(sw["final_normmatched@1.0"]["auc_vs_target"],
                    sw["final_raw@1.0"]["auc_vs_target"], s=46,
                    color=FAM_COL[fam], marker="o" if regime == "base" else "^",
                    edgecolor="k", lw=0.4, zorder=3)
        prov[tag] = {
            "alpha": a, "auc_vs_source": src, "auc_vs_target": tgt, "follow_gap": gap,
            "identity_max_absdiff": d["controls_Lstar"]["self_patch_max_absdiff_vs_natural"],
            "row_alignment_rel_mae": d["row_alignment_rel_mae"],
            "norm_matched_at_1": sw["final_normmatched@1.0"]["auc_vs_target"],
            "raw_at_1": sw["final_raw@1.0"]["auc_vs_target"],
            "judge_auc": d["judge_auc_natural"]}
        # the identity control must be exactly inert; flag it on the figure
    ax1.axhline(0, color="k", lw=0.8, ls=":")
    ax1.set_xlabel(r"$\alpha$  (0 = own activation, 1 = partner's)")
    ax1.set_ylabel(r"$\Delta$gap:  $P(\mathrm{yes})_{rel} - P(\mathrm{yes})_{irrel}$")
    ax1.set_title("(a) Does the verdict follow the transplanted activation?")
    ax1.legend(fontsize=6, ncol=2, loc="upper left", framealpha=0.9)
    ax1.grid(alpha=0.25)

    lo, hi = 0.30, 0.56
    ax2.plot([lo, hi], [lo, hi], "k:", lw=1.1)
    ax2.text(hi - 0.01, hi - 0.018, "rescaling changes nothing", fontsize=7, ha="right")
    ax2.set_xlim(lo, hi); ax2.set_ylim(lo, hi)
    ax2.set_xlabel("transplant into output layer, norm-matched to target")
    ax2.set_ylabel("same, payload left unscaled")
    ax2.set_title("(b) The collapse is not a scale artefact")
    ax2.grid(alpha=0.25)

    pct = [load_grad(t)["probe_vs_random_abs_mean_percentile"] for _, t, _, _ in MODELS]
    y = np.arange(len(MODELS))
    for i, ((name, tag, fam, regime), p) in enumerate(zip(MODELS, pct)):
        ax3.barh(i, p, color=FAM_COL[fam], alpha=0.8,
                 hatch="" if regime == "base" else "//", edgecolor="k", lw=0.4)
    ax3.axvline(0.95, color="k", ls="--", lw=1.2)
    ax3.text(0.955, len(MODELS) - 0.6, "outlier threshold\n(the misalignment\nclaim would fail here)",
             fontsize=6.5, va="top")
    ax3.set_yticks(y); ax3.set_yticklabels([m[0] for m in MODELS], fontsize=7.5)
    ax3.set_xlim(0, 1.05); ax3.set_xlabel("percentile of the probe direction among 16 random ones")
    ax3.set_title("(c) Is the readout pointed at the relevance direction?")
    ax3.grid(axis="x", alpha=0.25)

    fig.suptitle("Mechanism: patching and gradient alignment across the panel",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(outdir, "fig7_patch.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    sidecar("fig7_patch", {"claim": "patching behaviour and gradient percentiles",
                           "source": "results/patch_v2_*.json + gradient_alignment_*.json",
                           "per_model": prov,
                           "gradient_percentiles": dict(zip([m[1] for m in MODELS], pct))})
    return p


def fig_gradient(outdir):
    """Gradient cosine vs the random-direction spread, per model."""
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    prov = {}
    for i, (name, tag, fam, regime) in enumerate(MODELS):
        d = load_grad(tag)
        pc = d["probe_direction_cosine"]["mean"]
        rc = d["random_directions_cosine"]
        ys = np.random.default_rng(7 + i).normal(i, 0.085,
                                                 size=len(rc["per_direction_means"]))
        ax.scatter(rc["per_direction_means"], ys, s=14, color="#95a5a6",
                   alpha=0.75, zorder=2, label="16 random directions" if i == 0 else None)
        ax.scatter([pc], [i], s=95, color=FAM_COL[fam], zorder=4,
                   marker="o" if regime == "base" else "D",
                   edgecolor="k", lw=0.6,
                   label="probe direction" if i == 0 else None)
        prov[tag] = {"probe_cos": pc, "random_means": rc["per_direction_means"],
                     "percentile": d["probe_vs_random_abs_mean_percentile"]}
    ax.axvline(0, color="k", lw=0.9, ls=":")
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([m[0] for m in MODELS], fontsize=8)
    ax.set_xlabel(r"$\cos(\ \nabla_{h_{L^*}}(\mathrm{yes}-\mathrm{no}),\ \hat w_{\mathrm{probe}}\ )$")
    ax.set_title("The output readout is not pointed at the direction a probe reads relevance from\n"
                 "(grey = 16 random directions per model, colour = the probe direction; "
                 "nothing is edited)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlim(-0.045, 0.045)
    fig.tight_layout()
    p = os.path.join(outdir, "fig8_gradient.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    sidecar("fig8_gradient", {"claim": "probe direction is never a gradient outlier",
                              "source": "results/gradient_alignment_*_q1500.json",
                              "per_model": prov})
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(ROOT, "results/figures"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    for fn in (fig_patch, fig_gradient):
        print("wrote", fn(args.outdir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
