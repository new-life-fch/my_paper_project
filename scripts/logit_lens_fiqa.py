#!/usr/bin/env python3
"""
Logit lens for the FiQA cache, so the fixed-direction control can be compared
ON THE SAME DATASET (CPU only: one model load, all BLAS).

The main logit_lens.py hardcodes the {tag}_q{N}_instruct cache; this variant
takes a cache path so the FiQA run (tag llama32_3b_fiqa_bm25) can be measured
with the identical procedure. Reading the two on different datasets would be an
apples-to-oranges comparison, which is exactly the kind of thing this project
has been burned by before.

Usage:
  python scripts/logit_lens_fiqa.py \
      --model meta-llama/Llama-3.2-3B \
      --cache results/cache/llama32_3b_fiqa_bm25_q1500_instruct \
      --judge-cache results/cache/llama32_3b_fiqa_bm25_q1500_judge \
      --out results/logit_lens_llama32_3b_fiqa_bm25_q1500.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cache_paths import resolve_cache  # noqa: E402

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

YES_WORDS = ["yes", "Yes", "YES", " yes", " YES"]
NO_WORDS = ["no", "No", "NO", " no", " NO"]


def auc(y, s):
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--judge-cache", default=None)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()
    cache = resolve_cache(args.cache)
    jcache = resolve_cache(args.judge_cache) if args.judge_cache else None

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {args.model} (CPU) ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()

    def first(words):
        s = set()
        for w in words:
            e = tok.encode(w, add_special_tokens=False)
            if e:
                s.add(e[0])
        return sorted(s)
    yes_ids, no_ids = first(YES_WORDS), first(NO_WORDS)

    norm = model.model.norm
    head = model.lm_head

    def split(name):
        d = torch.load(os.path.join(cache, f"resid_{name}.pt"), map_location="cpu")
        R = d["acts"].to(torch.float32)
        y = d["labels"].numpy()
        nL = R.shape[1]
        out = np.zeros((R.shape[0], nL), dtype=np.float32)
        with torch.no_grad():
            for li in range(nL):
                for i in range(0, R.shape[0], args.batch):
                    h = norm(R[i:i + args.batch, li, :])
                    lg = head(h)
                    yes = torch.logsumexp(lg[:, yes_ids], dim=-1)
                    no = torch.logsumexp(lg[:, no_ids], dim=-1)
                    p = torch.softmax(torch.stack([yes, no], -1), -1)[:, 0]
                    out[i:i + args.batch, li] = p.numpy()
        return out, y

    res = {"tag": args.tag, "model": args.model, "cache": cache,
           "procedure": "final_norm + lm_head on cached layer output (zero training)"}
    pv, yv = split("val")
    pt, yt = split("test")
    val_auc = [round(auc(yv, pv[:, li]), 4) for li in range(pv.shape[1])]
    test_auc = [round(auc(yt, pt[:, li]), 4) for li in range(pt.shape[1])]
    L = int(np.argmax(val_auc))
    res["logit_lens_per_layer_auc_val"] = val_auc
    res["logit_lens_per_layer_auc"] = test_auc
    res["logit_lens_final_layer_auc"] = test_auc[-1]
    res["logit_lens_best_layer"] = {"layer": L, "auc": test_auc[L],
                                    "val_auc_at_selected": val_auc[L]}
    res["n_val"], res["n_test"] = int(len(yv)), int(len(yt))

    if jcache and os.path.exists(os.path.join(jcache, "judge_test.pt")):
        jd = torch.load(os.path.join(jcache, "judge_test.pt"), map_location="cpu")
        res["llm_judge_auc"] = round(auc(jd["labels"].numpy(),
                                         jd["scores"].numpy()), 4)
        print(f"  judge AUC = {res['llm_judge_auc']}")
    print(f"  logit lens: final layer {test_auc[-1]:.4f}, "
          f"val-selected best L{L} = {test_auc[L]:.4f}")
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"  saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
