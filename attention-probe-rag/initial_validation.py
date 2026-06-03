#!/usr/bin/env python3
"""
Attention-Probe-RAG: Initial Validation Pipeline
=================================================

Milestone 1 (M1) + Milestone 2 (M2) validation script.

Validates the core hypothesis: attention head activations from LLM forward pass
can be used to train binary relevance probes that significantly outperform random.

Pipeline:
  1. Load model (LLaMA-3.2-3B via nnsight)
  2. Sanity check: verify nnsight integration and activation shapes
  3. Load MS MARCO data and split by query
  4. Extract activations (Scheme A: last_token, Scheme B: pooling)
  5. Train per-head probes (L1 logistic regression)
  6. Evaluate probes and compare schemes
  7. Train ensemble probe with top-k heads
  8. Generate visualizations

Usage:
  python initial_validation.py --model meta-llama/Llama-3.2-3B --n-queries 50
  python initial_validation.py --model meta-llama/Llama-3.2-3B --n-queries 100 --top-k 20 50 100
  python initial_validation.py --help
"""

import os
import sys
import argparse
import logging
import json
import gc
from datetime import datetime

import torch
import numpy as np

# Add src/ to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import load_ms_marco, split_by_query
from src.activations import (
    extract_activations_last_token,
    extract_activations_pooling,
    reshape_for_probes,
)
from src.probes import (
    train_per_head_probes,
    evaluate_per_head_probes,
    train_ensemble_probe,
    evaluate_ensemble_probe,
)
from src.evaluation import (
    plot_per_head_heatmap,
    plot_top_k_comparison,
    plot_scheme_comparison,
    plot_ensemble_performance,
)


def setup_logging(output_dir):
    """Configure logging to both file and stdout."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "validation.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def load_model(model_name, torch_dtype=torch.float16):
    """Load LLM via nnsight LanguageModel."""
    from nnsight import LanguageModel

    logging.info(f"Loading model: {model_name} (dtype={torch_dtype})")
    logging.info("  This may take several minutes on first run (model download)...")

    model = LanguageModel(
        model_name,
        device_map="auto",
        dispatch=True,
        torch_dtype=torch_dtype,
    )

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // n_heads

    logging.info(f"  Architecture: {n_layers} layers, {n_heads} heads/layer, "
                 f"head_dim={head_dim}, hidden_size={hidden_size}")
    logging.info(f"  Total attention heads: {n_layers * n_heads}")

    return model, n_layers, n_heads, head_dim


def sanity_check(model, n_layers, n_heads, head_dim):
    """
    Quick nnsight integration test: run a forward pass on a simple prompt,
    verify activation shapes match expectations.
    """
    logging.info("=" * 60)
    logging.info("SANITY CHECK: nnsight + model integration")
    logging.info("=" * 60)

    test_prompt = "Q: What is the capital of France?\nP: Paris is the capital and most populous city of France."

    logging.info(f"  Prompt: {test_prompt[:80]}...")

    with model.trace(test_prompt):
        attn_out = model.model.layers[0].self_attn.o_proj.output[0]
        B, S, H = attn_out.shape
        per_head = attn_out.view(B, S, n_heads, head_dim)
        shape_info = (B, S, n_heads, head_dim)
        last_token_shape = per_head[0, -1, :, :].shape
        saved = per_head[0, -1, :, :].detach().cpu().save()

    logging.info(f"  Input tokens: B={shape_info[0]}, S={shape_info[1]}")
    logging.info(f"  o_proj.output[0] shape: [{B}, {S}, {H}]")
    logging.info(f"  Reshaped to: [{B}, {S}, {n_heads}, {head_dim}]")
    logging.info(f"  Last token per-head shape: {last_token_shape}")
    logging.info(f"  Saved tensor shape: {saved.shape}")
    logging.info(f"  Sample values (head 0, first 5 dims): {saved[0, :5].tolist()}")

    assert saved.shape == (n_heads, head_dim), \
        f"Shape mismatch! Expected ({n_heads}, {head_dim}), got {saved.shape}"

    logging.info("  Sanity check PASSED!")
    del saved
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def run_activation_extraction(model, train_samples, val_samples, test_samples,
                               n_layers, n_heads, head_dim, scheme, max_length):
    """Extract activations for a given scheme across all splits."""
    logging.info(f"--- Extracting activations: {scheme} ---")

    extract_fn = (extract_activations_last_token if scheme == "last_token"
                  else extract_activations_pooling)

    logging.info(f"  Train ({len(train_samples)} samples)...")
    train_act = extract_fn(model, train_samples, n_layers, n_heads, head_dim, max_length)

    logging.info(f"  Val ({len(val_samples)} samples)...")
    val_act = extract_fn(model, val_samples, n_layers, n_heads, head_dim, max_length)

    logging.info(f"  Test ({len(test_samples)} samples)...")
    test_act = extract_fn(model, test_samples, n_layers, n_heads, head_dim, max_length)

    # Reshape to [N, n_total_heads, head_dim]
    train_act = reshape_for_probes(train_act)
    val_act = reshape_for_probes(val_act)
    test_act = reshape_for_probes(test_act)

    logging.info(f"  Shapes: train={train_act.shape}, val={val_act.shape}, test={test_act.shape}")

    # Memory cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return train_act, val_act, test_act


def run_probe_training(train_act, val_act, test_act, train_labels, val_labels,
                       test_labels, n_layers, n_heads, head_dim, scheme_name,
                       top_k_values, output_dir):
    """
    Train and evaluate per-head probes + ensemble probe for one scheme.

    Returns:
        per_head_results, ensemble_results_list
    """
    logging.info(f"--- Probe training: {scheme_name} ---")

    # Stage 1: Per-head probes on train split
    logging.info("Stage 1: Training per-head L1 logistic regression probes...")
    per_head_train = train_per_head_probes(
        train_act, train_labels, n_layers, n_heads, head_dim
    )

    # Evaluate per-head probes on val split
    logging.info("Stage 1: Evaluating on validation split...")
    per_head_val = evaluate_per_head_probes(
        per_head_train, val_act, val_labels, n_layers, n_heads, split_name="val"
    )

    # Evaluate per-head probes on test split
    logging.info("Stage 1: Evaluating on test split...")
    per_head_test = evaluate_per_head_probes(
        per_head_train, test_act, test_labels, n_layers, n_heads, split_name="test"
    )

    # Merge results
    for i in range(len(per_head_train)):
        per_head_train[i].update(per_head_val[i])
        per_head_train[i].update(per_head_test[i])

    # Log top heads
    logging.info("  Top-10 heads (by val ROC-AUC):")
    sorted_by_val = sorted(per_head_train, key=lambda x: x.get("val_roc_auc", 0), reverse=True)
    for i, r in enumerate(sorted_by_val[:10]):
        logging.info(f"    #{i+1}: Layer {r['layer']}, Head {r['head']} "
                     f"(val_auc={r.get('val_roc_auc', 0):.3f}, "
                     f"test_auc={r.get('test_roc_auc', 0):.3f}, "
                     f"train_auc={r['roc_auc']:.3f})")

    n_above_random = sum(1 for r in per_head_train if r.get("val_roc_auc", 0) > 0.55)
    logging.info(f"  Heads above random+0.05 (val): {n_above_random}/{len(per_head_train)}")

    # Visualize
    plot_per_head_heatmap(
        per_head_train, n_layers, n_heads,
        f"{scheme_name} - Per-Head ROC-AUC (Train)",
        os.path.join(output_dir, f"heatmap_{scheme_name}.png"),
        metric="roc_auc",
    )
    plot_top_k_comparison(
        per_head_train,
        f"{scheme_name} - Per-Head Probe Performance",
        os.path.join(output_dir, f"topk_{scheme_name}.png"),
        top_k_values=top_k_values,
    )

    # Stage 2: Ensemble probes with varying top-k
    logging.info("Stage 2: Training ensemble probes...")
    # Sort by val performance for head selection
    head_order = sorted(per_head_train, key=lambda x: x.get("val_roc_auc", 0), reverse=True)

    ensemble_results = []
    for k in top_k_values:
        selected = [r["head_idx"] for r in head_order[:k]]

        # Train ensemble on train split
        probe, train_metrics = train_ensemble_probe(
            train_act, train_labels, selected,
            n_layers, n_heads, head_dim
        )

        # Evaluate on val and test
        val_metrics = evaluate_ensemble_probe(
            probe, val_act, val_labels, selected,
            n_layers, n_heads, head_dim, split_name="val"
        )
        test_metrics = evaluate_ensemble_probe(
            probe, test_act, test_labels, selected,
            n_layers, n_heads, head_dim, split_name="test"
        )

        result = {"n_heads": k, **train_metrics, **val_metrics, **test_metrics}
        ensemble_results.append(result)

        logging.info(f"  top-{k}: train_auc={train_metrics['roc_auc']:.3f}, "
                     f"val_auc={val_metrics['val_roc_auc']:.3f}, "
                     f"test_auc={test_metrics['test_roc_auc']:.3f}")

    return per_head_train, ensemble_results


def main():
    parser = argparse.ArgumentParser(
        description="Attention-Probe-RAG: Initial Validation Pipeline"
    )
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.2-3B",
        help="HuggingFace model ID (default: meta-llama/Llama-3.2-3B)"
    )
    parser.add_argument(
        "--n-queries", type=int, default=50,
        help="Number of queries for Phase 1 validation (default: 50)"
    )
    parser.add_argument(
        "--max-passages", type=int, default=5,
        help="Max passages per query (default: 5)"
    )
    parser.add_argument(
        "--max-length", type=int, default=512,
        help="Max token length for truncation (default: 512)"
    )
    parser.add_argument(
        "--top-k", type=int, nargs="+", default=[10, 20, 50, 100],
        help="Top-k head counts for ensemble probes (default: 10 20 50 100)"
    )
    parser.add_argument(
        "--scheme", type=str, default="both",
        choices=["last_token", "pooling", "both"],
        help="Activation extraction scheme (default: both)"
    )
    parser.add_argument(
        "--dtype", type=str, default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype (default: float16)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results/initial_validation",
        help="Output directory for results"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    # Setup
    output_dir = args.output_dir
    logger = setup_logging(output_dir)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    # Save config
    config = vars(args)
    config["timestamp"] = datetime.now().isoformat()
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    logging.info("=" * 60)
    logging.info("Attention-Probe-RAG: Initial Validation Pipeline")
    logging.info("=" * 60)
    logging.info(f"Config: {json.dumps(config, indent=2)}")

    # ================================================================
    # Step 1: Load model
    # ================================================================
    model, n_layers, n_heads, head_dim = load_model(args.model, torch_dtype)

    # ================================================================
    # Step 2: Sanity check
    # ================================================================
    sanity_check(model, n_layers, n_heads, head_dim)

    # ================================================================
    # Step 3: Data preparation
    # ================================================================
    logging.info("=" * 60)
    logging.info("DATA PREPARATION: MS MARCO")
    logging.info("=" * 60)

    samples = load_ms_marco(
        n_queries=args.n_queries,
        max_passages_per_query=args.max_passages,
        seed=args.seed,
    )
    train_samples, val_samples, test_samples = split_by_query(samples, seed=args.seed)

    train_labels = [s["label"] for s in train_samples]
    val_labels = [s["label"] for s in val_samples]
    test_labels = [s["label"] for s in test_samples]

    logging.info(f"Label distribution — train: {sum(train_labels)}/{len(train_labels)} pos, "
                 f"val: {sum(val_labels)}/{len(val_labels)} pos, "
                 f"test: {sum(test_labels)}/{len(test_labels)} pos")

    # ================================================================
    # Step 4 & 5 & 6: Extract + Train + Evaluate
    # ================================================================
    all_per_head_results = {}
    all_ensemble_results = {}

    schemes_to_run = (["last_token", "pooling"] if args.scheme == "both"
                      else [args.scheme])

    for scheme in schemes_to_run:
        logging.info("=" * 60)
        logging.info(f"SCHEME: {scheme}")
        logging.info("=" * 60)

        # Extract activations
        train_act, val_act, test_act = run_activation_extraction(
            model, train_samples, val_samples, test_samples,
            n_layers, n_heads, head_dim, scheme, args.max_length
        )

        # Train and evaluate probes
        per_head, ensemble = run_probe_training(
            train_act, val_act, test_act,
            train_labels, val_labels, test_labels,
            n_layers, n_heads, head_dim,
            scheme, args.top_k, output_dir
        )

        all_per_head_results[scheme] = per_head
        all_ensemble_results[scheme] = ensemble

        # Save per-head results to CSV
        import pandas as pd
        df = pd.DataFrame([
            {k: v for k, v in r.items() if k != "probe"}
            for r in per_head
        ])
        csv_path = os.path.join(output_dir, f"per_head_results_{scheme}.csv")
        df.to_csv(csv_path, index=False)
        logging.info(f"  Saved per-head results: {csv_path}")

        # Free memory between schemes
        del train_act, val_act, test_act
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ================================================================
    # Step 7: Scheme comparison (if both were run)
    # ================================================================
    if "last_token" in all_per_head_results and "pooling" in all_per_head_results:
        logging.info("=" * 60)
        logging.info("SCHEME COMPARISON: last_token vs pooling")
        logging.info("=" * 60)

        plot_scheme_comparison(
            all_per_head_results["last_token"],
            all_per_head_results["pooling"],
            os.path.join(output_dir, "scheme_comparison.png"),
        )

        # Head-by-head comparison
        a_auc = {r["head_idx"]: r["roc_auc"] for r in all_per_head_results["last_token"]}
        b_auc = {r["head_idx"]: r["roc_auc"] for r in all_per_head_results["pooling"]}
        common = set(a_auc.keys()) & set(b_auc.keys())

        a_mean = np.mean([a_auc[h] for h in common])
        b_mean = np.mean([b_auc[h] for h in common])
        logging.info(f"  Mean train ROC-AUC (common heads): "
                     f"last_token={a_mean:.4f}, pooling={b_mean:.4f}")
        logging.info(f"  Winner: {'last_token' if a_mean > b_mean else 'pooling'} "
                     f"(diff={abs(a_mean - b_mean):.4f})")

        # Ensemble comparison
        for k in args.top_k:
            a_ens = next((r for r in all_ensemble_results["last_token"] if r["n_heads"] == k), None)
            b_ens = next((r for r in all_ensemble_results["pooling"] if r["n_heads"] == k), None)
            if a_ens and b_ens:
                logging.info(f"  Ensemble top-{k}: "
                             f"last_token test_auc={a_ens.get('test_roc_auc', 0):.3f}, "
                             f"pooling test_auc={b_ens.get('test_roc_auc', 0):.3f}")

        # Plot ensemble curves
        for scheme in schemes_to_run:
            plot_ensemble_performance(
                all_ensemble_results[scheme],
                os.path.join(output_dir, f"ensemble_curve_{scheme}.png"),
            )

    # ================================================================
    # Summary
    # ================================================================
    logging.info("=" * 60)
    logging.info("VALIDATION COMPLETE")
    logging.info("=" * 60)

    # Core hypothesis check
    for scheme, results in all_per_head_results.items():
        n_above = sum(1 for r in results if r.get("val_roc_auc", 0) > 0.55)
        total = len(results)
        pct = n_above / total * 100 if total > 0 else 0
        logging.info(f"  [{scheme}] Heads above 0.55 val_auc: {n_above}/{total} ({pct:.1f}%)")

        if n_above > total * 0.1:
            logging.info(f"  [{scheme}] CORE HYPOTHESIS SUPPORTED: "
                         f"Significant fraction of heads encode relevance information.")
        else:
            logging.info(f"  [{scheme}] WARNING: Few heads above threshold. "
                         f"Consider trying more data, different model, or alternative activations.")

    logging.info(f"\nResults saved to: {output_dir}/")
    logging.info("Files: validation.log, config.json, per_head_results_*.csv, *.png")


if __name__ == "__main__":
    main()
