"""
Evaluation and visualization module for Attention-Probe-RAG.

Provides:
  - Per-head performance heatmaps (layer x head).
  - Top-k heads analysis plots.
  - Scheme A vs Scheme B comparison charts.
  - Ensemble probe performance curves.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns


def plot_per_head_heatmap(per_head_results, n_layers, n_heads, title, save_path,
                           metric="roc_auc"):
    """
    Plot a heatmap of per-head probe performance (layers x heads).

    Args:
        per_head_results: List of dicts from train_per_head_probes.
        n_layers: Number of transformer layers.
        n_heads: Number of heads per layer.
        title: Plot title.
        save_path: Path to save the figure.
        metric: Which metric to visualize ('roc_auc', 'accuracy', 'f1').
    """
    # Build matrix [n_layers, n_heads]
    matrix = np.full((n_layers, n_heads), np.nan)
    for r in per_head_results:
        matrix[r["layer"], r["head"]] = r[metric]

    fig, ax = plt.subplots(figsize=(max(12, n_heads * 0.5), max(8, n_layers * 0.3)))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="RdYlGn",
        vmin=0.4,
        vmax=0.9,
        xticklabels=range(n_heads),
        yticklabels=range(n_layers),
        cbar_kws={"label": metric},
    )
    ax.set_xlabel("Attention Head")
    ax.set_ylabel("Layer")
    ax.set_title(f"{title}\n({metric} per attention head)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved heatmap: {save_path}")


def plot_top_k_comparison(per_head_results, title, save_path, top_k_values=None):
    """
    Plot ROC-AUC distribution of per-head probes and highlight top-k heads.

    Args:
        per_head_results: List of dicts sorted by ROC-AUC.
        title: Plot title.
        save_path: Path to save figure.
        top_k_values: List of k values to highlight (default: [10, 20, 50, 100]).
    """
    if top_k_values is None:
        top_k_values = [10, 20, 50, 100]

    aucs = [r["roc_auc"] for r in per_head_results]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Sorted ROC-AUC curve
    ax = axes[0]
    ax.plot(range(len(aucs)), aucs, "b-", linewidth=1, alpha=0.7)
    ax.axhline(0.5, color="r", linestyle="--", label="Random (0.5)")

    for k in top_k_values:
        if k < len(aucs):
            ax.axhline(aucs[k], color="gray", linestyle=":", alpha=0.5)
            ax.annotate(f"top-{k}: {aucs[k]:.3f}",
                       xy=(k, aucs[k]), fontsize=8, color="green")

    ax.set_xlabel("Head Rank (sorted by ROC-AUC)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"{title}\nPer-head ROC-AUC (sorted)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Histogram of ROC-AUC values
    ax = axes[1]
    ax.hist(aucs, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(0.5, color="r", linestyle="--", label="Random (0.5)")
    ax.axvline(np.mean(aucs), color="g", linestyle="-", label=f"Mean ({np.mean(aucs):.3f})")
    ax.set_xlabel("ROC-AUC")
    ax.set_ylabel("Count")
    ax.set_title(f"{title}\nROC-AUC Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved top-k analysis: {save_path}")


def plot_scheme_comparison(scheme_a_results, scheme_b_results, save_path):
    """
    Plot head-by-head comparison between Scheme A (last_token) and Scheme B (pooling).

    Args:
        scheme_a_results: Per-head results for Scheme A.
        scheme_b_results: Per-head results for Scheme B.
        save_path: Path to save figure.
    """
    # Build lookup by head_idx
    a_by_head = {r["head_idx"]: r["roc_auc"] for r in scheme_a_results}
    b_by_head = {r["head_idx"]: r["roc_auc"] for r in scheme_b_results}

    common_heads = sorted(set(a_by_head.keys()) & set(b_by_head.keys()))
    a_vals = [a_by_head[h] for h in common_heads]
    b_vals = [b_by_head[h] for h in common_heads]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Scatter plot
    ax = axes[0]
    ax.scatter(a_vals, b_vals, alpha=0.3, s=8, color="steelblue")
    ax.plot([0.4, 0.9], [0.4, 0.9], "r--", alpha=0.5, label="y=x")
    ax.set_xlabel("Scheme A (last_token) ROC-AUC")
    ax.set_ylabel("Scheme B (pooling) ROC-AUC")
    ax.set_title("Per-Head Comparison: Scheme A vs Scheme B")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Difference histogram
    ax = axes[1]
    diffs = [b - a for a, b in zip(a_vals, b_vals)]
    ax.hist(diffs, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="r", linestyle="--", label="No difference")
    ax.axvline(np.mean(diffs), color="g", linestyle="-",
               label=f"Mean diff: {np.mean(diffs):+.4f}")
    ax.set_xlabel("ROC-AUC Difference (B - A)")
    ax.set_ylabel("Count")
    ax.set_title("Scheme B - Scheme A Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved scheme comparison: {save_path}")


def plot_ensemble_performance(ensemble_results, save_path):
    """
    Plot ensemble probe performance as a function of number of top-k heads.

    Args:
        ensemble_results: List of dicts with 'n_heads', 'accuracy', 'f1', 'roc_auc'.
        save_path: Path to save figure.
    """
    n_heads_list = [r["n_heads"] for r in ensemble_results]
    metrics = {
        "Accuracy": [r["accuracy"] for r in ensemble_results],
        "F1": [r["f1"] for r in ensemble_results],
        "ROC-AUC": [r["roc_auc"] for r in ensemble_results],
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, values in metrics.items():
        ax.plot(n_heads_list, values, "o-", label=name, linewidth=2, markersize=5)

    ax.axhline(0.5, color="r", linestyle="--", alpha=0.5, label="Random (0.5)")
    ax.set_xlabel("Number of Top-k Heads")
    ax.set_ylabel("Score")
    ax.set_title("Ensemble Probe Performance vs Number of Heads")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved ensemble curve: {save_path}")
