#!/usr/bin/env python3
"""
Extract activations under an INSTRUCTION prompt, capturing TWO sites at once.

Motivation: the plain "Q:..\nP:.." format never asks the model to judge
relevance, and all 6 probe methods plateaued at test AUC ~0.63 on those
activations. Probing works best when the model is poised to answer the
property being probed. So we reframe the input as an explicit relevance
question and probe the ANSWER-token activations.

Two sites captured in a single forward pass (GPU-efficient):
  attn  : per-head o_proj output at last token   -> [N, n_layers*n_heads, head_dim]
  resid : residual stream (layer output) last tok -> [N, n_layers, hidden_size]

Same seed/splits as the plain cache, so results are directly comparable.

Usage:
  python scripts/extract_instruct.py --n-queries 500 --out results/cache/q500_instruct
"""
import os
import sys
import argparse
import json

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_ms_marco, split_by_query


def build_instruct_prompt(query: str, passage: str) -> str:
    # Document first, then the question, then the judgment ask. The token after
    # "Answer:" is where the model is poised to emit a relevance verdict.
    return (f"Document: {passage}\n"
            f"Question: {query}\n"
            f"Is the document relevant to the question? Answer:")


def extract_both_sites(model, samples, n_layers, n_heads, head_dim, max_length):
    """Return (attn [N, L*nh, hd], resid [N, L, H]) at the last token."""
    attn_all, resid_all = [], []
    from tqdm import tqdm
    for s in tqdm(samples, desc="extract(instruct)"):
        prompt = build_instruct_prompt(s["query"], s["passage"])
        attn_layers, resid_layers = [], []
        with model.trace(prompt):
            for i in range(n_layers):
                # attention head output (o_proj is nn.Linear -> already [B,S,H])
                ao = model.model.layers[i].self_attn.o_proj.output
                B, S, H = ao.shape
                attn_layers.append(
                    ao.view(B, S, n_heads, head_dim)[0, -1, :, :].detach().cpu().save())
                # residual stream: in transformers 5.x the decoder layer's
                # .output is already the [B,S,H] hidden tensor (not a tuple)
                lo = model.model.layers[i].output
                resid_layers.append(lo[0, -1, :].detach().cpu().save())
        attn_all.append(torch.stack(attn_layers, dim=0))      # [L, nh, hd]
        resid_all.append(torch.stack(resid_layers, dim=0))    # [L, H]
    attn = torch.stack(attn_all, dim=0)                       # [N, L, nh, hd]
    N = attn.shape[0]
    attn = attn.reshape(N, n_layers * n_heads, head_dim)      # [N, L*nh, hd]
    resid = torch.stack(resid_all, dim=0)                     # [N, L, H]
    return attn, resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--n-queries", type=int, default=500)
    ap.add_argument("--max-passages", type=int, default=5)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/cache/q500_instruct")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    samples = load_ms_marco(n_queries=args.n_queries,
                            max_passages_per_query=args.max_passages, seed=args.seed)
    train_s, val_s, test_s = split_by_query(samples, seed=args.seed)
    splits = {"train": train_s, "val": val_s, "test": test_s}

    from nnsight import LanguageModel
    print(f"Loading {args.model} ...")
    model = LanguageModel(args.model, device_map="auto", dispatch=True, dtype=torch.float16)
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    hidden = model.config.hidden_size
    head_dim = hidden // n_heads
    print(f"Arch: {n_layers}L x {n_heads}H, head_dim={head_dim}, hidden={hidden}")

    meta = {"model": args.model, "n_queries": args.n_queries, "prompt": "instruct",
            "n_layers": n_layers, "n_heads": n_heads, "head_dim": head_dim,
            "hidden": hidden, "seed": args.seed,
            "splits": {k: len(v) for k, v in splits.items()}}

    for name, s in splits.items():
        attn, resid = extract_both_sites(model, s, n_layers, n_heads, head_dim, args.max_length)
        labels = torch.tensor([x["label"] for x in s], dtype=torch.long)
        qids = torch.tensor([x["query_id"] for x in s], dtype=torch.long)
        torch.save({"acts": attn.to(torch.float16), "labels": labels, "query_ids": qids},
                   os.path.join(args.out, f"attn_{name}.pt"))
        torch.save({"acts": resid.to(torch.float16), "labels": labels, "query_ids": qids},
                   os.path.join(args.out, f"resid_{name}.pt"))
        print(f"  {name}: attn={tuple(attn.shape)} resid={tuple(resid.shape)} "
              f"pos={int(labels.sum())}/{len(labels)}")

    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nCached to {args.out}/  (attn_*.pt, resid_*.pt, meta.json)")


if __name__ == "__main__":
    main()
