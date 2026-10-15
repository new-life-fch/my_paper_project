#!/usr/bin/env python3
"""
Figures for the narrow-scope paper. Pure CPU, reads only results/*.json.

The narrow paper's single quantitative claim is: reading the top attention heads
decodes retrieval relevance better than reading the output representation
(final-layer hidden), on every model in the panel. The figures in this file are
therefore all versions of that one comparison:

  fig1_readers          all four readers, 8 models, with bootstrap CIs
  fig2_forest           the attn - final edge with CIs + BH q-values
  fig3_depth            per-layer residual curve, per-head heatmap, head-vs-layer
  fig4_attn_vs_final    the comparison itself: paired slope chart + scatter
  fig5_layer_dist       where the selected heads live, per model
  fig6_judge_gap        leniency of the zero-shot judge (why "zero training" matters)

Every number comes from a named JSON; the .json sidecar of each figure records
the provenance so a reader can re-derive the plot.
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

# display name, tag, family, regime — plotting order groups base/instruct pairs
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
READER_COL = {"judge": "#95a5a6", "final": "#7f8c8d",
              "best_resid": "#e67e22", "attn": "#c0392b"}


def load(tag):
    s = json.load(open(os.path.join(ROOT, f"results/significance_{tag}_q{NQ}.json")))
    return s[0] if isinstance(s, list) else s


def sidecar(name, payload):
    p = os.path.join(ROOT, f"results/figures/{name}.json")
    json.dump(payload, open(p, "w"), indent=2)
    return p


def fig_readers(outdir):
    """Four readers x 8 models, with query-clustered 95% CI on each."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    names = [m[0] for m in MODELS]
    x = np.arange(len(MODELS))
    w = 0.2
    prov = {}
    for i, key in enumerate(("judge", "final", "best_resid", "attn")):
        vals, los, his = [], [], []
        for _, tag, _, _ in MODELS:
            d = load(tag)
            a = d["auc"][key]
            vals.append(a["point"]); los.append(a["point"] - a["ci95"][0])
            his.append(a["ci95"][1] - a["point"])
        ax.bar(x + (i - 1.5) * w, vals, w, label=key.replace("best_resid", "best-resid"),
               color=READER_COL[key], yerr=[los, his], capsize=2,
               error_kw=dict(lw=0.8, ecolor="#333"))
        prov[key] = {tag: load(tag)["auc"][key] for _, tag, _, _ in MODELS}
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("test AUC (query-clustered 95% CI)")
    ax.set_title("Four relevance readers on the same frozen forward pass "
                 "(8 models, 1500 queries each)")
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.text(len(MODELS) - 0.5, 0.505, "chance", fontsize=7, ha="right")
    ax.legend(ncol=4, fontsize=8, loc="upper left")
    ax.set_ylim(0.45, 0.82)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p = os.path.join(outdir, "fig1_readers.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    sidecar("fig1_readers", {"claim": "four readers, 8 models, with CIs",
                             "source": "results/significance_<tag>_q1500.json .auc",
                             "values": prov})
    return p


def _fmt_p(p):
    """Bootstrap p is floored at 1/(n_boot+1); printing "0" would overclaim."""
    if p is None:
        return "n/a"
    if p <= 1e-4:
        return "$<$2e-4"
    return f"{p:.3g}"


def fig_forest(outdir):
    """The headline edge: attn - final, per model, with CI and BH q."""
    sm = json.load(open(os.path.join(ROOT, "results/significance_summary_q1500.json")))
    per = {m["tag"]: m for m in sm["edges"]["attn-final"]["per_model"]}
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ys = np.arange(len(MODELS))[::-1]
    prov = {}
    for y, (name, tag, fam, regime) in zip(ys, MODELS):
        m = per[tag]
        c = FAM_COL[fam]
        ax.errorbar(m["mean"], y,
                    xerr=[[m["mean"] - m["ci95"][0]], [m["ci95"][1] - m["mean"]]],
                    fmt="o", color=c, capsize=3, lw=1.5,
                    markerfacecolor=c if regime == "base" else "white",
                    markeredgecolor=c, markersize=7)
        ax.text(m["ci95"][1] + 0.004, y, f"{m['mean']:+.3f}  ({_fmt_p(m['q_bh'])})",
                va="center", fontsize=7)
        prov[tag] = m
    ax.axvline(0, color="k", lw=0.9, ls="--")
    ax.set_yticks(ys); ax.set_yticklabels([m[0] for m in MODELS], fontsize=8)
    ax.set_xlabel("attn $-$ final  (test AUC difference)")
    ax.set_title("Reading attention heads beats reading the output representation\n"
                 "on all 8 models (filled = base, hollow = instruct; "
                 "parentheses = BH-corrected $q$)")
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlim(-0.01, 0.245)
    fig.tight_layout()
    p = os.path.join(outdir, "fig2_forest.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    sidecar("fig2_forest", {"claim": "attn-final edge per model with CI and BH q",
                            "source": "results/significance_summary_q1500.json",
                            "values": prov})
    return p


def fig_depth(outdir, tag="llama32_3b"):
    """Per-layer residual curve + per-head heatmap + head-vs-layer view."""
    d = json.load(open(os.path.join(ROOT, f"results/figures/{tag}_layer_signal.json")))
    L = d["n_layers"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax1, ax2, ax3 = axes

    te = d["resid_auc_per_layer"]; va = d["resid_auc_per_layer_val"]
    ax1.plot(range(L), va, "o-", ms=3, color="#7f8c8d", label="residual probe (val)")
    ax1.plot(range(L), te, "o-", ms=3, color="#2980b9", label="residual probe (test)")
    ax1.axhline(d["judge"], color="#95a5a6", ls="--", lw=1.2,
                label=f"zero-shot judge = {d['judge']:.3f}")
    pk = d["resid_peak"]
    ax1.axvline(pk["layer"], color="#e67e22", ls=":", lw=1)
    ax1.annotate(f"peak L{pk['layer']}\n{pk['auc']:.3f}",
                 xy=(pk["layer"], pk["auc"]), xytext=(4, -18),
                 textcoords="offset points", fontsize=7, color="#e67e22")
    ax1.axvline(L - 1, color="k", ls=":", lw=1)
    ax1.annotate(f"output L{L-1}\n{d['resid_final']:.3f}",
                 xy=(L - 1, d["resid_final"]), xytext=(-34, 8),
                 textcoords="offset points", fontsize=7)
    ax1.set_xlabel("layer"); ax1.set_ylabel("test AUC")
    ax1.set_title("(a) Relevance decodability by depth")
    ax1.legend(fontsize=7, loc="lower right"); ax1.grid(alpha=0.25)

    g = np.array(d["head_te_grid"])
    im = ax2.imshow(g, aspect="auto", cmap="magma", vmin=0.5, vmax=0.68,
                    origin="lower")
    ax2.set_xlabel("head index"); ax2.set_ylabel("layer")
    ax2.set_title("(b) Per-head linear decodability (test AUC)")
    fig.colorbar(im, ax=ax2, fraction=0.046, label="AUC")

    ax3.plot(range(L), d["head_layer_max"], "o-", ms=3, color="#c0392b",
             label="best single head in layer")
    ax3.plot(range(L), d["head_layer_mean"], "s-", ms=3, color="#e67e22",
             label="mean over heads in layer")
    ax3.plot(range(L), te, "^--", ms=3, color="#2980b9", lw=1,
             label="residual stream (same layer)")
    ax3.axhline(d["resid_final"], color="k", ls=":", lw=1)
    ax3.set_xlabel("layer"); ax3.set_ylabel("test AUC")
    ax3.set_title("(c) Head view vs layer view")
    ax3.legend(fontsize=7, loc="lower right"); ax3.grid(alpha=0.25)

    fig.suptitle(f"{d['model']}: where the relevance signal is readable "
                 f"({L} layers, {d['n_heads']} heads)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(outdir, f"fig3_depth_{tag}.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    sidecar(f"fig3_depth_{tag}", {
        "claim": "depth curve, per-head heatmap, head-vs-layer",
        "source": f"results/figures/{tag}_layer_signal.json",
        "resid_peak": d["resid_peak"], "judge": d["judge"],
        "resid_final": d["resid_final"], "topk_head_layers": d["topk_head_layers"]})
    return p


def fig_attn_vs_final(outdir):
    """The comparison itself: slope chart + paired scatter.

    The scatter shades the half-plane ABOVE the diagonal, because that is where
    the claim lives (attn > final); shading below would visually privilege the
    opposite outcome.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax1, ax2 = axes
    prov = {}
    # hand-tuned label offsets so the eight points stay legible
    off = {"llama32_3b": (-16, -12), "llama32_3b_instruct": (6, 6),
           "llama31_8b": (-30, -14), "llama31_8b_instruct": (6, 4),
           "mistral_7b": (-26, 8), "mistral_7b_instruct": (6, -10),
           "qwen25_7b": (6, 4), "qwen25_7b_instruct": (4, -12)}
    for name, tag, fam, regime in MODELS:
        d = load(tag)
        f = d["auc"]["final"]["point"]; a = d["auc"]["attn"]["point"]
        ax1.plot([0, 1], [f, a], color=FAM_COL[fam],
                 ls="-" if regime == "base" else "--",
                 marker="o",
                 markerfacecolor=FAM_COL[fam] if regime == "base" else "white",
                 markeredgecolor=FAM_COL[fam], ms=6, lw=1.4, alpha=0.85)
        ax2.scatter(f, a, s=48, color=FAM_COL[fam],
                    marker="o" if regime == "base" else "^",
                    edgecolor="k", lw=0.4, zorder=3)
        short = (name.replace("LLaMA", "L").replace("Mistral", "M")
                 .replace("Qwen2.5", "Q").replace("-Inst", "i"))
        ax2.annotate(short, (f, a), xytext=off.get(tag, (5, -10)),
                     textcoords="offset points", fontsize=6.5)
        prov[tag] = {"final": f, "attn": a}
    lo, hi = 0.52, 0.75
    ax2.plot([lo, hi], [lo, hi], "k:", lw=1.1)
    ax2.text(hi - 0.015, hi - 0.025, "attn = final", fontsize=7, ha="right")
    ax2.fill_between([lo, hi], [lo, lo], [hi, hi], color=FAM_COL["LLaMA"],
                     alpha=0.07, zorder=0)
    ax2.text(lo + 0.006, hi - 0.012, "attn wins", fontsize=7.5, va="top",
             color="#7b241c")
    ax2.set_xlim(lo, hi); ax2.set_ylim(lo, hi)
    ax2.set_xlabel("output representation (final-layer probe), AUC")
    ax2.set_ylabel("attention heads (ours), AUC")
    ax2.set_title("(b) All eight models sit above the diagonal")
    ax2.grid(alpha=0.25)

    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["output\nrepresentation", "attention\nheads"])
    ax1.set_ylabel("test AUC"); ax1.set_ylim(0.50, 0.76)
    ax1.set_title("(a) Paired slope chart (solid = base, dashed = instruct)")
    ax1.grid(axis="y", alpha=0.25)
    fig.suptitle("The same frozen forward pass, two places to read it", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(outdir, "fig4_attn_vs_final.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    sidecar("fig4_attn_vs_final", {"claim": "attn vs final, paired",
                                   "source": "significance_<tag>_q1500.json", "values": prov})
    return p


def fig_layer_dist(outdir):
    """Where the validation-selected heads live, per model."""
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    for i, (name, tag, fam, regime) in enumerate(MODELS):
        d = json.load(open(os.path.join(ROOT, f"results/figures/{tag}_layer_signal.json")))
        L = d["n_layers"]
        heads = d["topk_head_layers"]
        cnt = np.bincount(heads, minlength=L)
        frac = np.arange(L) / (L - 1)
        ax.scatter(frac[cnt > 0], [i] * int((cnt > 0).sum()),
                   s=cnt[cnt > 0] * 26, color=FAM_COL[fam],
                   marker="o" if regime == "base" else "^",
                   edgecolor="k", lw=0.4, alpha=0.85)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([m[0] for m in MODELS], fontsize=8)
    ax.set_xlabel("relative depth of the selected head (0 = first layer, 1 = last)")
    ax.set_title("Where the top-20 validation-selected heads live\n"
                 "(marker area = number of selected heads in that layer)")
    ax.set_xlim(-0.03, 1.03); ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    p = os.path.join(outdir, "fig5_layer_dist.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    sidecar("fig5_layer_dist", {
        "claim": "depth distribution of selected heads",
        "source": "results/figures/<tag>_layer_signal.json .topk_head_layers",
        "per_model": {tag: json.load(open(os.path.join(
            ROOT, f"results/figures/{tag}_layer_signal.json")))["topk_head_layers"]
            for _, tag, _, _ in MODELS}})
    return p


def fig_judge_gap(outdir):
    """Why 'zero training' matters: the judge vs a trained probe on its own layer."""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(MODELS)); w = 0.36
    prov = {}
    j = [load(t)["auc"]["judge"]["point"] for _, t, _, _ in MODELS]
    f = [load(t)["auc"]["final"]["point"] for _, t, _, _ in MODELS]
    a = [load(t)["auc"]["attn"]["point"] for _, t, _, _ in MODELS]
    ax.bar(x - w / 2, j, w, color=READER_COL["judge"], label="zero-shot judge (no training)")
    ax.bar(x + w / 2, a, w, color=READER_COL["attn"], label="attention-head probe (trained)")
    for xi, (jj, ff) in enumerate(zip(j, f)):
        ax.hlines(ff, xi - w / 2 - 0.14, xi - w / 2 + 0.14, color="k", lw=1.4)
        ax.text(xi - w / 2, jj + 0.006, f"{jj:.2f}", ha="center", fontsize=6)
        ax.text(xi + w / 2, a[xi] + 0.006, f"{a[xi]:.2f}", ha="center", fontsize=6)
    ax.plot([], [], "k_", ms=10, label="final-layer probe (trained, output repr.)")
    ax.set_xticks(x); ax.set_xticklabels([m[0] for m in MODELS], rotation=20,
                                         ha="right", fontsize=8)
    ax.set_ylabel("test AUC"); ax.set_ylim(0.45, 0.80)
    ax.axhline(0.5, color="k", ls=":", lw=0.8)
    ax.set_title("The output verdict is the weakest reader everywhere;\n"
                 "the horizontal tick is what a trained probe gets from the output representation")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p = os.path.join(outdir, "fig6_judge_gap.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    sidecar("fig6_judge_gap", {"claim": "judge vs attn vs final, per model",
                               "source": "significance_<tag>_q1500.json",
                               "judge": dict(zip([m[1] for m in MODELS], j)),
                               "final": dict(zip([m[1] for m in MODELS], f)),
                               "attn": dict(zip([m[1] for m in MODELS], a))})
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(ROOT, "results/figures"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    made = []
    for fn in (fig_readers, fig_forest, fig_attn_vs_final, fig_layer_dist, fig_judge_gap):
        made.append(fn(args.outdir)); print("wrote", made[-1])
    made.append(fig_depth(args.outdir)); print("wrote", made[-1])
    # the depth figure for every model, since it is the per-model evidence
    for _, tag, _, _ in MODELS:
        p = fig_depth(args.outdir, tag); print("wrote", p)
    print(f"\n{len(made)} figures in {args.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
