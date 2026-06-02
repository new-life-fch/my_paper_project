"""
Probe training module for Attention-Probe-RAG.

Implements the two-stage probe training pipeline:
  Stage 1: Per-head L1-regularized logistic regression (feature selection).
  Stage 2: Ensemble probe using top-k selected heads with L2 regularization.

Following the ITI (Li et al., 2023) methodology adapted for document relevance.
"""

import numpy as np
import torch
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm


def train_per_head_probes(activations, labels, n_layers, n_heads, head_dim,
                          C=1.0, max_iter=1000):
    """
    Train independent L1-regularized logistic regression probes for each attention head.

    Each head gets its own binary classifier:
      input:  activation vector of shape [head_dim]
      output: probability of passage being relevant

    Args:
        activations: Tensor [N, n_layers, n_heads, head_dim] or [N, n_total_heads, head_dim].
        labels: Array-like of shape [N] with binary labels (0/1).
        n_layers: Number of transformer layers.
        n_heads: Number of heads per layer.
        head_dim: Dimension per head.
        C: Inverse regularization strength for logistic regression.
        max_iter: Maximum iterations for solver.

    Returns:
        List of dicts with per-head results, sorted by validation ROC-AUC.
    """
    if isinstance(activations, torch.Tensor):
        activations = activations.numpy()
    labels = np.array(labels)

    # Handle both [N, n_layers, n_heads, d] and [N, n_total_heads, d] shapes
    if activations.ndim == 4:
        N, nl, nh, d = activations.shape
        activations = activations.reshape(N, nl * nh, d)

    n_total_heads = activations.shape[1]
    assert n_total_heads == n_layers * n_heads, \
        f"Expected {n_layers * n_heads} heads, got {n_total_heads}"

    results = []
    for head_idx in tqdm(range(n_total_heads), desc="Training per-head probes"):
        layer_idx = head_idx // n_heads
        h = head_idx % n_heads

        X = activations[:, head_idx, :]  # [N, head_dim]

        try:
            probe = LogisticRegression(
                penalty="l1",
                C=C,
                solver="saga",
                max_iter=max_iter,
                class_weight="balanced",
                random_state=42,
            )
            probe.fit(X, labels)
            y_pred = probe.predict(X)
            y_prob = probe.predict_proba(X)[:, 1]

            acc = accuracy_score(labels, y_pred)
            f1 = f1_score(labels, labels, average="binary", zero_division=0)
            # Recompute f1 properly
            f1 = f1_score(labels, y_pred, average="binary", zero_division=0)

            try:
                auc = roc_auc_score(labels, y_prob)
            except ValueError:
                auc = 0.5

            # Number of non-zero coefficients (L1 sparsity)
            n_nonzero = np.sum(probe.coef_ != 0)

            results.append({
                "layer": layer_idx,
                "head": h,
                "head_idx": head_idx,
                "accuracy": acc,
                "f1": f1,
                "roc_auc": auc,
                "n_nonzero_features": int(n_nonzero),
                "probe": probe,
            })
        except Exception as e:
            results.append({
                "layer": layer_idx,
                "head": h,
                "head_idx": head_idx,
                "accuracy": 0.0,
                "f1": 0.0,
                "roc_auc": 0.5,
                "n_nonzero_features": 0,
                "probe": None,
                "error": str(e),
            })

    # Sort by ROC-AUC descending
    results.sort(key=lambda x: x["roc_auc"], reverse=True)
    return results


def evaluate_per_head_probes(per_head_results, activations, labels,
                              n_layers, n_heads, split_name="val"):
    """
    Evaluate per-head probes on a held-out split.

    Args:
        per_head_results: List of dicts from train_per_head_probes (with 'probe' keys).
        activations: Tensor [N, n_layers, n_heads, head_dim] or [N, n_total_heads, head_dim].
        labels: Array-like of shape [N].
        n_layers: Number of layers.
        n_heads: Number of heads per layer.
        split_name: Name for logging.

    Returns:
        List of dicts with evaluation metrics per head.
    """
    if isinstance(activations, torch.Tensor):
        activations = activations.numpy()
    labels = np.array(labels)

    if activations.ndim == 4:
        N, nl, nh, d = activations.shape
        activations = activations.reshape(N, nl * nh, d)

    eval_results = []
    for r in per_head_results:
        head_idx = r["head_idx"]
        probe = r["probe"]

        if probe is None:
            eval_results.append({
                "layer": r["layer"],
                "head": r["head"],
                "head_idx": head_idx,
                f"{split_name}_accuracy": 0.0,
                f"{split_name}_f1": 0.0,
                f"{split_name}_roc_auc": 0.5,
            })
            continue

        X = activations[:, head_idx, :]
        try:
            y_pred = probe.predict(X)
            y_prob = probe.predict_proba(X)[:, 1]

            eval_results.append({
                "layer": r["layer"],
                "head": r["head"],
                "head_idx": head_idx,
                f"{split_name}_accuracy": accuracy_score(labels, y_pred),
                f"{split_name}_f1": f1_score(labels, y_pred, average="binary", zero_division=0),
                f"{split_name}_roc_auc": roc_auc_score(labels, y_prob),
            })
        except Exception as e:
            eval_results.append({
                "layer": r["layer"],
                "head": r["head"],
                "head_idx": head_idx,
                f"{split_name}_accuracy": 0.0,
                f"{split_name}_f1": 0.0,
                f"{split_name}_roc_auc": 0.5,
            })

    return eval_results


def train_ensemble_probe(activations, labels, selected_head_indices,
                         n_layers, n_heads, head_dim,
                         C=1.0, penalty="l2", max_iter=2000):
    """
    Train an ensemble probe using concatenated features from selected heads.

    Stage 2: Combine top-k heads' activations into a single feature vector,
    then train an L2-regularized logistic regression.

    Args:
        activations: Tensor [N, n_layers, n_heads, head_dim] or [N, n_total_heads, head_dim].
        labels: Array-like of shape [N].
        selected_head_indices: List of head indices to include.
        n_layers: Number of layers.
        n_heads: Number of heads per layer.
        head_dim: Dimension per head.
        C: Inverse regularization strength.
        penalty: Regularization type ('l1' or 'l2').
        max_iter: Maximum iterations.

    Returns:
        (probe, metrics_dict)
    """
    if isinstance(activations, torch.Tensor):
        activations = activations.numpy()
    labels = np.array(labels)

    if activations.ndim == 4:
        N, nl, nh, d = activations.shape
        activations = activations.reshape(N, nl * nh, d)

    # Concatenate selected heads: [N, len(selected) * head_dim]
    X = np.concatenate(
        [activations[:, idx, :] for idx in selected_head_indices],
        axis=1
    )

    probe = LogisticRegression(
        penalty=penalty,
        C=C,
        solver="lbfgs" if penalty == "l2" else "saga",
        max_iter=max_iter,
        class_weight="balanced",
        random_state=42,
    )
    probe.fit(X, labels)

    y_pred = probe.predict(X)
    y_prob = probe.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": accuracy_score(labels, y_pred),
        "f1": f1_score(labels, y_pred, average="binary", zero_division=0),
        "roc_auc": roc_auc_score(labels, y_prob),
        "n_heads": len(selected_head_indices),
        "n_features": X.shape[1],
    }

    return probe, metrics


def evaluate_ensemble_probe(probe, activations, labels, selected_head_indices,
                             n_layers, n_heads, head_dim, split_name="val"):
    """
    Evaluate ensemble probe on a held-out split.

    Args:
        probe: Trained sklearn LogisticRegression.
        activations: Tensor of activations.
        labels: Ground truth labels.
        selected_head_indices: Head indices used during training.
        split_name: Name for logging.

    Returns:
        Dict of metrics.
    """
    if isinstance(activations, torch.Tensor):
        activations = activations.numpy()
    labels = np.array(labels)

    if activations.ndim == 4:
        N, nl, nh, d = activations.shape
        activations = activations.reshape(N, nl * nh, d)

    X = np.concatenate(
        [activations[:, idx, :] for idx in selected_head_indices],
        axis=1
    )

    y_pred = probe.predict(X)
    y_prob = probe.predict_proba(X)[:, 1]

    return {
        f"{split_name}_accuracy": accuracy_score(labels, y_pred),
        f"{split_name}_f1": f1_score(labels, y_pred, average="binary", zero_division=0),
        f"{split_name}_roc_auc": roc_auc_score(labels, y_prob),
    }
