#!/usr/bin/env python3
"""
Generalized cross-domain OOD builder for ANY BEIR dataset (BeIR/<name>).

Generalizes build_ood_scifact.py: BM25 top-N rerank pools per query, build
(query, passage, label) triples, split by query, extract LLaMA-3.2-3B instruct
attn answer-token activations, cache for scoring (zeroshot_transfer / compare_ood
/ strong_reranker all read the same layout).

Tested targets: BeIR/fiqa (finance), BeIR/nfcorpus (medical), BeIR/trec-covid.

Usage:
  python scripts/build_ood.py --dataset BeIR/fiqa     --out results/cache/fiqa_ood
  python scripts/build_ood.py --dataset BeIR/nfcorpus --out results/cache/nfcorpus_ood
"""
import os, sys, argparse, json, random
from collections import defaultdict
import torch, numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_triples(dataset, topn, seed, max_queries, qrels_split):
    from datasets import load_dataset
    from rank_bm25 import BM25Okapi

    corpus = load_dataset(dataset, "corpus", trust_remote_code=True)["corpus"]
    queries = load_dataset(dataset, "queries", trust_remote_code=True)["queries"]
    qrels = load_dataset(f"{dataset}-qrels")
    if qrels_split == "auto":
        qrels_split = "test" if "test" in qrels else list(qrels.keys())[0]
    qr = qrels[qrels_split]
    print(f"qrels split = {qrels_split} ({len(qr)} rows)")

    doc_text = {}
    for r in corpus:
        t = (r.get("title") or "").strip()
        x = (r.get("text") or "").strip()
        doc_text[str(r["_id"])] = (t + ". " + x) if t else x
    qtext = {str(r["_id"]): r["text"] for r in queries}

    rel = defaultdict(set)
    for r in qr:
        if int(r["score"]) > 0:
            rel[str(r["query-id"])].add(str(r["corpus-id"]))
    rel_qids = [q for q in rel.keys() if q in qtext]
    random.seed(seed); random.shuffle(rel_qids)
    if max_queries:
        rel_qids = rel_qids[:max_queries]
    print(f"corpus={len(doc_text)} queries_with_qrels={len(rel_qids)}")

    doc_ids = list(doc_text.keys())
    print("Building BM25 over corpus...")
    bm = BM25Okapi([doc_text[d].lower().split() for d in doc_ids])
    id2idx = {d: i for i, d in enumerate(doc_ids)}

    triples = []
    for qid in rel_qids:
        q = qtext[qid]
        scores = bm.get_scores(q.lower().split())
        top_idx = np.argsort(scores)[::-1][:topn]
        cand_ids = [doc_ids[i] for i in top_idx]
        rels_here = rel[qid]
        if not any(c in rels_here for c in cand_ids):
            inj = next(iter(rels_here))
            if inj in doc_text:
                cand_ids[-1] = inj
        for c in cand_ids:
            triples.append({"query_id": int(qid) if qid.isdigit() else hash(qid) & 0x7fffffff,
                            "query": q, "passage": doc_text[c],
                            "label": 1 if c in rels_here else 0,
                            "bm25": float(scores[id2idx[c]])})
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
    ap.add_argument("--dataset", required=True, help="e.g. BeIR/fiqa")
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--max-queries", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--qrels-split", default="auto")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    triples = build_triples(args.dataset, args.topn, args.seed, args.max_queries, args.qrels_split)
    sp = split_by_query(triples, seed=args.seed)
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

    json.dump({"model": args.model, "dataset": args.dataset, "topn": args.topn,
               "n_layers": n_layers, "n_heads": n_heads, "head_dim": head_dim,
               "splits": {k: len(v) for k, v in sp.items()}},
              open(os.path.join(args.out, "meta.json"), "w"), indent=2)
    print(f"\nCached to {args.out}/")


if __name__ == "__main__":
    main()
