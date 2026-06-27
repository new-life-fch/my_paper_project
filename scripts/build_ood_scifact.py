#!/usr/bin/env python3
"""
Cross-domain (OOD) test: SciFact (scientific claims) vs MS MARCO (web search).

The decisive test of the repositioned thesis. The cross-encoder
ms-marco-MiniLM-L-6-v2 is MS-MARCO-specialized; on out-of-domain SciFact it
should degrade. A probe on a FROZEN general LLM, trained with a few SciFact
labels, may transfer relatively better. If the probe closes or reverses the
in-domain gap here, that is a genuine contribution.

Pipeline (BEIR SciFact, standard IR format):
  1. Load corpus / queries / qrels.
  2. Per query: BM25 retrieve top-N candidates from the corpus -> a realistic
     rerank pool with hard negatives (mirrors a deployed RAG retriever).
  3. Build (query, passage, label) triples (label=1 if in qrels, else 0).
  4. Split by query (train/val/test) for the probe; cross-encoder/BM25 need no training.
  5. Extract LLaMA-3.2-3B instruct attn answer-token activations for all triples.
  6. Cache everything for the scoring step (compare_methods-style).

This script does the data + extraction; scoring/metrics done by compare_ood.py.

Usage:
  python scripts/build_ood_scifact.py --topn 20 --out results/cache/scifact_ood
"""
import os
import sys
import argparse
import json
import random
from collections import defaultdict

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_triples(topn, seed, max_queries):
    from datasets import load_dataset
    from rank_bm25 import BM25Okapi

    corpus = load_dataset("BeIR/scifact", "corpus", trust_remote_code=True)["corpus"]
    queries = load_dataset("BeIR/scifact", "queries", trust_remote_code=True)["queries"]
    qrels = load_dataset("BeIR/scifact-qrels")
    qrels_split = qrels["test"] if "test" in qrels else list(qrels.values())[0]

    # maps
    doc_text = {}
    for r in corpus:
        t = (r.get("title") or "").strip()
        x = (r.get("text") or "").strip()
        doc_text[str(r["_id"])] = (t + ". " + x) if t else x
    qtext = {str(r["_id"]): r["text"] for r in queries}

    # qrels: query-id -> set(relevant corpus-id)
    rel = defaultdict(set)
    for r in qrels_split:
        if int(r["score"]) > 0:
            rel[str(r["query-id"])].add(str(r["corpus-id"]))
    rel_qids = [q for q in rel.keys() if q in qtext]
    random.seed(seed)
    random.shuffle(rel_qids)
    if max_queries:
        rel_qids = rel_qids[:max_queries]
    print(f"corpus={len(doc_text)} queries_with_qrels={len(rel_qids)}")

    # BM25 over whole corpus
    doc_ids = list(doc_text.keys())
    print("Building BM25 over corpus...")
    bm = BM25Okapi([doc_text[d].lower().split() for d in doc_ids])

    triples = []
    for qid in rel_qids:
        q = qtext[qid]
        scores = bm.get_scores(q.lower().split())
        top_idx = np.argsort(scores)[::-1][:topn]
        cand_ids = [doc_ids[i] for i in top_idx]
        # ensure at least one relevant doc is in the pool (inject if BM25 missed it)
        rels_here = rel[qid]
        if not any(c in rels_here for c in cand_ids):
            # add one relevant doc (replace worst candidate)
            inj = next(iter(rels_here))
            if inj in doc_text:
                cand_ids[-1] = inj
        for c in cand_ids:
            triples.append({"query_id": int(qid) if qid.isdigit() else hash(qid) & 0x7fffffff,
                            "query": q, "passage": doc_text[c],
                            "label": 1 if c in rels_here else 0,
                            "bm25": float(scores[doc_ids.index(c)])})
    pos = sum(t["label"] for t in triples)
    print(f"triples={len(triples)} pos={pos} neg={len(triples)-pos}")
    return triples


def split_by_query(triples, seed=42, ratios=(0.5, 0.2)):
    qids = sorted(set(t["query_id"] for t in triples))
    random.seed(seed); random.shuffle(qids)
    n = len(qids); n_tr = int(n*ratios[0]); n_va = int(n*ratios[1])
    tr = set(qids[:n_tr]); va = set(qids[n_tr:n_tr+n_va])
    sp = {"train": [], "val": [], "test": []}
    for t in triples:
        s = "train" if t["query_id"] in tr else ("val" if t["query_id"] in va else "test")
        sp[s].append(t)
    for k, v in sp.items():
        print(f"  {k}: {len(v)} ({sum(x['label'] for x in v)} pos)")
    return sp


def build_instruct_prompt(query, passage):
    return (f"Document: {passage}\n"
            f"Question: {query}\n"
            f"Is the document relevant to the question? Answer:")


def extract_attn(model, samples, n_layers, n_heads, head_dim):
    from tqdm import tqdm
    out = []
    for s in tqdm(samples, desc="extract"):
        prompt = build_instruct_prompt(s["query"], s["passage"])
        layers = []
        with model.trace(prompt):
            for i in range(n_layers):
                ao = model.model.layers[i].self_attn.o_proj.output
                B, S, H = ao.shape
                layers.append(ao.view(B, S, n_heads, head_dim)[0, -1, :, :].detach().cpu().save())
        out.append(torch.stack(layers, dim=0))
    a = torch.stack(out, dim=0)
    N = a.shape[0]
    return a.reshape(N, n_layers*n_heads, head_dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--max-queries", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/cache/scifact_ood")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    triples = build_triples(args.topn, args.seed, args.max_queries)
    sp = split_by_query(triples, seed=args.seed)

    # save raw triples (text + bm25 + labels) for scoring baselines
    json.dump(sp, open(os.path.join(args.out, "triples.json"), "w"))

    from nnsight import LanguageModel
    print(f"Loading {args.model} ...")
    model = LanguageModel(args.model, device_map="auto", dispatch=True, dtype=torch.float16)
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads

    for name, s in sp.items():
        if not s:
            continue
        acts = extract_attn(model, s, n_layers, n_heads, head_dim)
        torch.save({"acts": acts.to(torch.float16),
                    "labels": torch.tensor([x["label"] for x in s]),
                    "query_ids": torch.tensor([x["query_id"] for x in s])},
                   os.path.join(args.out, f"attn_{name}.pt"))
        print(f"  saved attn_{name}: {tuple(acts.shape)}")

    json.dump({"model": args.model, "dataset": "BeIR/scifact", "topn": args.topn,
               "n_layers": n_layers, "n_heads": n_heads, "head_dim": head_dim,
               "splits": {k: len(v) for k, v in sp.items()}},
              open(os.path.join(args.out, "meta.json"), "w"), indent=2)
    print(f"\nCached to {args.out}/")


if __name__ == "__main__":
    main()
