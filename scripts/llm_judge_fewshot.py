#!/usr/bin/env python3
"""
Few-shot LLM-judge: a STRONGER output-side upper bound.

Motivation (reviewer rebuttal): the zero-shot judge might look weak only
because the prompt is weak. This variant prepends K balanced in-context
exemplars (drawn ONLY from the train split, same data the probe trains on,
so the comparison stays fair / no val-test leakage) before asking the same
yes/no relevance question. If even a few-shot judge cannot match the internal
probe, the "internals > output" gap is not a prompt-engineering artefact.

Same model / answer-token / yes-no scoring as llm_judge.py. Same seed/splits.

Usage:
  python scripts/llm_judge_fewshot.py --n-queries 1500 --k-shot 4 \
      --out results/cache/llama32_3b_q1500_judge_fs4
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

def qp_block(query: str, passage: str) -> str:
    return (f"Document: {passage}\n"
            f"Question: {query}\n"
            f"Is the document relevant to the question? Answer:")


def build_fewshot_prefix(exemplars):
    """exemplars: list of (query, passage, label). Verbalise as a worked demo."""
    parts = []
    for q, p, y in exemplars:
        ans = " Yes" if y == 1 else " No"
        parts.append(qp_block(q, p) + ans)
    return "\n\n".join(parts)


def pick_exemplars(train_s, k, seed):
    """k balanced exemplars (k/2 pos, k/2 neg) sampled from TRAIN only."""
    rng = np.random.default_rng(seed)
    pos = [x for x in train_s if x["label"] == 1]
    neg = [x for x in train_s if x["label"] == 0]
    npos = k // 2
    nneg = k - npos
    pi = rng.choice(len(pos), size=min(npos, len(pos)), replace=False)
    ni = rng.choice(len(neg), size=min(nneg, len(neg)), replace=False)
    ex = [(pos[i]["query"], pos[i]["passage"], 1) for i in pi] + \
         [(neg[i]["query"], neg[i]["passage"], 0) for i in ni]
    rng.shuffle(ex)
    return ex


def yes_no_token_ids(tokenizer):
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
    ap.add_argument("--n-queries", type=int, default=1500)
    ap.add_argument("--max-passages", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k-shot", type=int, default=4)
    ap.add_argument("--out", default="results/cache/judge_fs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    samples = load_ms_marco(n_queries=args.n_queries,
                            max_passages_per_query=args.max_passages, seed=args.seed)
    train_s, val_s, test_s = split_by_query(samples, seed=args.seed)
    splits = {"train": train_s, "val": val_s, "test": test_s}

    exemplars = pick_exemplars(train_s, args.k_shot, args.seed)
    prefix = build_fewshot_prefix(exemplars)
    ex_qids = {x["query_id"] for q, p, y in exemplars
               for x in train_s if x["query"] == q and x["passage"] == p}

    import torch as _t
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {args.model} (few-shot judge, k={args.k_shot}) ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto",
                                                 dtype=_t.float16)
    model.eval()
    dev = next(model.parameters()).device
    yes_ids, no_ids = yes_no_token_ids(tok)
    print(f"yes ids {yes_ids} | no ids {no_ids}")

    from tqdm import tqdm
    meta = {"model": args.model, "n_queries": args.n_queries, "method": "llm_judge_fewshot",
            "k_shot": args.k_shot, "yes_ids": yes_ids, "no_ids": no_ids, "seed": args.seed,
            "exemplar_query_ids": sorted(ex_qids)}
    auc_report = {}

    for name, s in splits.items():
        scores = []
        for x in tqdm(s, desc=f"fsjudge/{name}"):
            full = prefix + "\n\n" + qp_block(x["query"], x["passage"])
            ids = tok(full, return_tensors="pt").to(dev)
            with _t.no_grad():
                logits = model(**ids).logits[0, -1, :]
            lg = logits.float().cpu()
            yes = _t.logsumexp(lg[yes_ids], dim=0)
            no = _t.logsumexp(lg[no_ids], dim=0)
            p_yes = _t.softmax(_t.stack([yes, no]), dim=0)[0].item()
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
    print(f"\nFew-shot judge (k={args.k_shot}) AUC: {auc_report}")


if __name__ == "__main__":
    main()
