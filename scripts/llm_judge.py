#!/usr/bin/env python3
"""
LLM-judge baseline: ask the SAME model for a relevance verdict, NO probe.

This is the critical control. Our best probe (instruct attn answer-token)
reaches test AUC ~0.685. If simply reading the model's P("yes") vs P("no")
at the answer token matches or beats that, the probe adds nothing over just
asking the model — so this baseline decides whether the whole probe approach
is worth pursuing.

Same instruct prompt, same seed/splits as extract_instruct.py. Captures the
answer-token logits restricted to yes/no token variants (not the full 128k
vocab), computes a normalized P(yes) score per sample, saves scores+labels+
query_ids, and reports ROC-AUC on each split.

Usage:
  python scripts/llm_judge.py --n-queries 500 --out results/cache/q500_judge
"""
import os
import sys
import argparse
import json

import torch
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_ms_marco, split_by_query


def build_instruct_prompt(query: str, passage: str) -> str:
    # Identical to extract_instruct.py so the judge sees the same context.
    return (f"Document: {passage}\n"
            f"Question: {query}\n"
            f"Is the document relevant to the question? Answer:")


def yes_no_token_ids(tokenizer):
    """Collect first-token ids for yes/no surface variants."""
    yes_words = [" Yes", " yes", "Yes", "yes", " YES"]
    no_words = [" No", " no", "No", "no", " NO"]

    def first_ids(words):
        ids = set()
        for w in words:
            enc = tokenizer.encode(w, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
        return sorted(ids)

    return first_ids(yes_words), first_ids(no_words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--n-queries", type=int, default=500)
    ap.add_argument("--max-passages", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/cache/q500_judge")
    ap.add_argument("--dataset", default="ms_marco",
                    help="ms_marco | beir:fiqa | beir:scifact")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.dataset.startswith("beir:"):
        from src.data import load_beir_relevance
        samples = load_beir_relevance(args.dataset.split(":", 1)[1],
                                      n_queries=args.n_queries,
                                      max_passages_per_query=args.max_passages,
                                      seed=args.seed)
    else:
        samples = load_ms_marco(n_queries=args.n_queries,
                                max_passages_per_query=args.max_passages, seed=args.seed)
    train_s, val_s, test_s = split_by_query(samples, seed=args.seed)
    splits = {"train": train_s, "val": val_s, "test": test_s}

    from nnsight import LanguageModel
    print(f"Loading {args.model} ...")
    model = LanguageModel(args.model, device_map="auto", dispatch=True, dtype=torch.float16)
    tok = model.tokenizer
    yes_ids, no_ids = yes_no_token_ids(tok)
    print(f"yes token ids: {yes_ids}")
    print(f"no  token ids: {no_ids}")

    from tqdm import tqdm
    meta = {"model": args.model, "n_queries": args.n_queries, "method": "llm_judge",
            "yes_ids": yes_ids, "no_ids": no_ids, "seed": args.seed}
    auc_report = {}

    for name, s in splits.items():
        scores = []
        for x in tqdm(s, desc=f"judge/{name}"):
            prompt = build_instruct_prompt(x["query"], x["passage"])
            with model.trace(prompt):
                logits = model.lm_head.output[0, -1, :].detach().cpu().save()
            lg = logits.float()
            # logsumexp over each group, then softmax between the two groups
            yes = torch.logsumexp(lg[yes_ids], dim=0)
            no = torch.logsumexp(lg[no_ids], dim=0)
            p_yes = torch.softmax(torch.stack([yes, no]), dim=0)[0].item()
            scores.append(p_yes)
        scores = np.array(scores)
        labels = np.array([x["label"] for x in s])
        qids = np.array([x["query_id"] for x in s])
        torch.save({"scores": torch.tensor(scores), "labels": torch.tensor(labels),
                    "query_ids": torch.tensor(qids)},
                   os.path.join(args.out, f"judge_{name}.pt"))
        try:
            a = roc_auc_score(labels, scores)
        except ValueError:
            a = 0.5
        auc_report[name] = round(float(a), 3)
        print(f"  {name}: AUC={a:.3f}  (pos {labels.sum()}/{len(labels)})")

    meta["auc"] = auc_report
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nLLM-judge AUC: {auc_report}")
    print(f"Compare against best probe test AUC 0.685 (instruct attn answer-token).")


if __name__ == "__main__":
    main()
