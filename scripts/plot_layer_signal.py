#!/usr/bin/env python3
"""
Paper main figure: WHERE along depth is (query,passage) relevance decodable,
and how does it relate to the output layer?

Produces a 3-panel figure from a q*_instruct cache (model-agnostic):
  A. per-layer residual-stream probe AUC vs depth, with the LLM-judge baseline,
     the peak layer, and the final (output-representation) layer marked. This is
     the "signal forms mid-network, decays toward output" story.
  B. per-head attention probe AUC heatmap (layer x head).
  C. layer-view vs head-view: per-layer max/mean head AUC curve + a histogram of
     which layers the global top-k heads live in.

Also dumps the underlying numbers to results/figures/<tag>_layer_signal.json so
the paper can cite exact values.

Usage:
  python scripts/plot_layer_signal.py --cache results/cache/q500_instruct \
      --judge-cache results/cache/q500_judge --tag llama32_3b
"""
import os, sys, argparse, json
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

N_JOBS = max(1, min(48, (os.cpu_count() or 4) - 2))


def auc(y, s):
    try: return float(roc_auc_score(y, s))
    except ValueError: return 0.5


def lr(penalty="l2", C=1.0):
    return LogisticRegression(penalty=penalty, C=C,
                              solver="lbfgs" if penalty == "l2" else "liblinear",
                              max_iter=3000, class_weight="balanced", random_state=42)


def load(cache, scheme, split):
    d = torch.load(os.path.join(cache, f"{scheme}_{split}.pt"))
    return d["acts"].to(torch.float32).numpy(), d["labels"].numpy()


def compute(cache, judge_cache, topk):
    meta = json.load(open(os.path.join(cache, "meta.json")))
    n_layers, n_heads = meta["n_layers"], meta["n_heads"]

    # judge baseline (output, zero training)
    judge = None
    jp = os.path.join(judge_cache, "judge_test.pt")
    if os.path.exists(jp):
        jd = torch.load(jp)
        judge = round(auc(jd["labels"].numpy(), jd["scores"].numpy()), 4)

    # ---- per-layer resid probe AUC (depth curve) ----
    Rtr, ytr = load(cache, "resid", "train")
    Rte, yte = load(cache, "resid", "test")

    def layer_auc(li):
        m = make_pipeline(StandardScaler(), lr(C=0.5)).fit(Rtr[:, li, :], ytr)
        return round(auc(yte, m.predict_proba(Rte[:, li, :])[:, 1]), 4)
    resid_auc = Parallel(n_jobs=min(N_JOBS, n_layers))(
        delayed(layer_auc)(li) for li in range(n_layers))

    # ---- per-head attn probe AUC (val-selected, test-scored) ----
    Atr, _ = load(cache, "attn", "train")
    Ava, yva = load(cache, "attn", "val")
    Ate, _ = load(cache, "attn", "test")
    n_total_heads = Atr.shape[1]

    def head_auc(h):
        p = lr(penalty="l1").fit(Atr[:, h, :], ytr)
        va = auc(yva, p.predict_proba(Ava[:, h, :])[:, 1])
        te = auc(yte, p.predict_proba(Ate[:, h, :])[:, 1])
        return va, te
    res = Parallel(n_jobs=N_JOBS)(delayed(head_auc)(h) for h in range(n_total_heads))
    head_va = np.array([r[0] for r in res])
    head_te = np.array([r[1] for r in res])
    # heads were cached as flat L*nh in layer-major order
    head_te_grid = head_te.reshape(n_layers, n_heads)

    sel = np.argsort(head_va)[::-1][:topk]   # select on val (no leakage)
    sel_layers = (sel // n_heads).tolist()

    return {
        "model": meta["model"], "n_layers": n_layers, "n_heads": n_heads,
        "judge": judge,
        "resid_auc_per_layer": resid_auc,
        "resid_peak": {"layer": int(np.argmax(resid_auc)), "auc": max(resid_auc)},
        "resid_final": resid_auc[-1],
        "head_te_grid": head_te_grid.tolist(),
        "head_layer_max": head_te_grid.max(axis=1).round(4).tolist(),
        "head_layer_mean": head_te_grid.mean(axis=1).round(4).tolist(),
        "topk": topk,
        "topk_head_layers": sel_layers,
        "topk_head_val_auc": head_va[sel].round(4).tolist(),
    }


def plot(d, save_png):
    nl = d["n_layers"]
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # --- A: depth curve ---
    ax = axes[0]
    xs = list(range(nl))
    ax.plot(xs, d["resid_auc_per_layer"], "o-", color="#2c6fbb", lw=2, ms=4,
            label="resid probe (trained)")
    pk = d["resid_peak"]
    ax.scatter([pk["layer"]], [pk["auc"]], color="green", zorder=5, s=80,
               label=f"peak L{pk['layer']} = {pk['auc']:.3f}")
    ax.scatter([nl - 1], [d["resid_final"]], color="orange", zorder=5, s=80,
               label=f"final L{nl-1} = {d['resid_final']:.3f}")
    if d["judge"] is not None:
        ax.axhline(d["judge"], color="red", ls="--", alpha=0.7,
                   label=f"LLM-judge (output) = {d['judge']:.3f}")
    ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("Layer depth"); ax.set_ylabel("test ROC-AUC")
    ax.set_title("A. Relevance decodability vs depth\n(forms mid-network, decays to output)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # --- B: per-head heatmap ---
    ax = axes[1]
    grid = np.array(d["head_te_grid"])
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0.45, vmax=max(0.7, grid.max()),
                   origin="lower")
    ax.set_xlabel("Head"); ax.set_ylabel("Layer")
    ax.set_title("B. Per-head attn probe AUC\n(layer x head)")
    fig.colorbar(im, ax=ax, label="test ROC-AUC", fraction=0.046)

    # --- C: layer-view vs head-view ---
    ax = axes[2]
    ax.plot(xs, d["head_layer_max"], "s-", color="#b5651d", lw=2, ms=4,
            label="best head per layer")
    ax.plot(xs, d["head_layer_mean"], "^-", color="#888", lw=1.5, ms=3,
            label="mean head per layer")
    ax.set_xlabel("Layer depth"); ax.set_ylabel("test ROC-AUC", color="#b5651d")
    ax.set_title(f"C. Head-view vs layer-view\n(top-{d['topk']} head layer histogram)")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.hist(d["topk_head_layers"], bins=range(nl + 1), alpha=0.3, color="purple")
    ax2.set_ylabel(f"# of top-{d['topk']} heads", color="purple")
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(f"Layer signal localization — {d['model']}", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_png, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/cache/q500_instruct")
    ap.add_argument("--judge-cache", default="results/cache/q500_judge")
    ap.add_argument("--tag", default="llama32_3b")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--outdir", default="results/figures")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    d = compute(args.cache, args.judge_cache, args.topk)
    png = os.path.join(args.outdir, f"{args.tag}_layer_signal.png")
    js = os.path.join(args.outdir, f"{args.tag}_layer_signal.json")
    plot(d, png)
    json.dump(d, open(js, "w"), indent=2)
    print(f"peak L{d['resid_peak']['layer']}={d['resid_peak']['auc']:.3f}  "
          f"final L{d['n_layers']-1}={d['resid_final']:.3f}  judge={d['judge']}")
    print(f"top-{d['topk']} head layers: {sorted(d['topk_head_layers'])}")
    print(f"Saved: {png}\n       {js}")


if __name__ == "__main__":
    main()
