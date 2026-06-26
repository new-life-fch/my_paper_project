#!/usr/bin/env python3
"""
Extract activations once and cache to disk.

The model forward pass is the expensive step (~1 min for 300 queries). By
caching activations, we can iterate on probe methodology in seconds without
re-running the model. Saves per-split tensors for both extraction schemes.

Usage:
  python scripts/extract_and_cache.py --n-queries 500 --out results/cache/q500
"""
import os
import sys
import argparse
import json

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_ms_marco, split_by_query
from src.activations import (
    extract_activations_last_token,
    extract_activations_pooling,
    reshape_for_probes,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--n-queries", type=int, default=500)
    ap.add_argument("--max-passages", type=int, default=5)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/cache/q500")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # --- Data ---
    samples = load_ms_marco(n_queries=args.n_queries,
                            max_passages_per_query=args.max_passages, seed=args.seed)
    train_s, val_s, test_s = split_by_query(samples, seed=args.seed)
    splits = {"train": train_s, "val": val_s, "test": test_s}
    for name, s in splits.items():
        labels = np.array([x["label"] for x in s])
        print(f"{name}: {len(s)} samples, {labels.sum()} pos / {len(labels)-labels.sum()} neg")

    # --- Model (nnsight) ---
    from nnsight import LanguageModel
    print(f"Loading {args.model} ...")
    model = LanguageModel(args.model, device_map="auto", dispatch=True, dtype=torch.float16)
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    print(f"Arch: {n_layers} layers x {n_heads} heads, head_dim={head_dim}, total={n_layers*n_heads}")

    extractors = {
        "last_token": extract_activations_last_token,
        "pooling": extract_activations_pooling,
    }

    meta = {
        "model": args.model, "n_queries": args.n_queries,
        "n_layers": n_layers, "n_heads": n_heads, "head_dim": head_dim,
        "seed": args.seed, "max_length": args.max_length,
        "splits": {k: len(v) for k, v in splits.items()},
    }

    for scheme, fn in extractors.items():
        for name, s in splits.items():
            print(f"[{scheme}/{name}] extracting {len(s)} ...")
            acts = fn(model, s, n_layers, n_heads, head_dim, args.max_length)
            acts = reshape_for_probes(acts)  # [N, total_heads, head_dim]
            labels = torch.tensor([x["label"] for x in s], dtype=torch.long)
            qids = torch.tensor([x["query_id"] for x in s], dtype=torch.long)
            # save float16 to halve disk; probes convert to float32 on load
            torch.save(
                {"acts": acts.to(torch.float16), "labels": labels, "query_ids": qids},
                os.path.join(args.out, f"{scheme}_{name}.pt"),
            )

    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nCached to {args.out}/  (meta.json + {{scheme}}_{{split}}.pt)")


if __name__ == "__main__":
    main()
