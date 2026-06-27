#!/usr/bin/env python3
"""
Re-run the reranker baseline with a STRONG modern general-domain reranker
(BAAI/bge-reranker-v2-m3), replacing the 2021 ms-marco-MiniLM-L-6-v2.

Motivation: the in-domain "probe loses to reranker" and the OOD comparisons
were measured against a weak, dated cross-encoder. To make the verdict honest
we must compare against the current best open general reranker. bge-reranker-v2-m3
is a top open-source general/multilingual reranker (XLM-RoBERTa backbone),
loaded directly via transformers (sigmoid over a single relevance logit).

Scores BOTH test sets it can reach without re-extraction:
  - in-domain MS MARCO test (reload text in deterministic split order)
  - SciFact test (from triples.json)
and prints alongside the probe numbers already on record.

Usage:
  python scripts/strong_reranker.py --which both
"""
import os, sys, argparse, json
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RERANKER = "BAAI/bge-reranker-v2-m3"


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    try: return round(float(roc_auc_score(y, s)), 3)
    except ValueError: return 0.5


def dcg(rels): return sum(r / np.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(lab, k=10):
    ideal = sorted(lab, reverse=True); idcg = dcg(ideal[:k])
    return dcg(lab[:k]) / idcg if idcg > 0 else 0.0


def per_query(scores, labels, qids):
    by_q = defaultdict(lambda: ([], []))
    for s, l, q in zip(scores, labels, qids):
        by_q[q][0].append(s); by_q[q][1].append(l)
    out = []
    for s, l in by_q.values():
        order = np.argsort(s)[::-1]; lab = np.array(l)[order]
        if lab.sum() == 0: continue
        first = np.argmax(lab == 1) + 1
        out.append({"mrr": 1.0/first, "ndcg@10": ndcg_at_k(lab.tolist(), 10),
                    "recall@3": float(lab[:3].sum())/lab.sum(),
                    "recall@5": float(lab[:5].sum())/lab.sum()})
    keys = ["mrr", "ndcg@10", "recall@3", "recall@5"]
    return {k: round(float(np.mean([m[k] for m in out])), 3) for k in keys}, len(out)


@torch.no_grad()
def rerank_scores(model, tok, pairs, bs=32, max_len=512):
    scores = []
    for i in range(0, len(pairs), bs):
        batch = pairs[i:i+bs]
        enc = tok([p[0] for p in batch], [p[1] for p in batch],
                  padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt").to(DEVICE)
        logits = model(**enc).logits.view(-1).float()
        scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return np.array(scores)


def report_one(tag, scores, labels, qids, probe_ref):
    a = auc(labels, scores); agg, nq = per_query(scores, labels, qids)
    print(f"\n=== {tag} ===")
    print(f"  {'bge-reranker-v2-m3':22s} AUC={a:.3f} MRR={agg['mrr']:.3f} "
          f"NDCG@10={agg['ndcg@10']:.3f} R@3={agg['recall@3']:.3f} (q={nq})")
    print(f"  {'probe (on record)':22s} {probe_ref}")
    return {"auc": a, "n_queries": nq, **agg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["msmarco", "scifact", "both"], default="both")
    ap.add_argument("--out", default="results/strong_reranker.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    print(f"Loading {RERANKER} on {DEVICE}...")
    tok = AutoTokenizer.from_pretrained(RERANKER)
    model = AutoModelForSequenceClassification.from_pretrained(RERANKER).to(DEVICE).eval()

    out = {"reranker": RERANKER, "results": {}}

    if args.which in ("msmarco", "both"):
        from src.data import load_ms_marco, split_by_query
        samples = load_ms_marco(n_queries=500, max_passages_per_query=5, seed=42)
        _, _, te = split_by_query(samples, seed=42)
        labels = np.array([x["label"] for x in te]); qids = np.array([x["query_id"] for x in te])
        pairs = [[x["query"], x["passage"]] for x in te]
        s = rerank_scores(model, tok, pairs)
        out["results"]["msmarco"] = report_one(
            "IN-DOMAIN MS MARCO (test)", s, labels, qids,
            "AUC=0.685 MRR=0.686 NDCG@10=0.758 R@3=0.836 (old CE=0.759)")

    if args.which in ("scifact", "both"):
        sp = json.load(open("results/cache/scifact_ood/triples.json"))["test"]
        labels = np.array([t["label"] for t in sp]); qids = np.array([t["query_id"] for t in sp])
        pairs = [[t["query"], t["passage"]] for t in sp]
        s = rerank_scores(model, tok, pairs)
        out["results"]["scifact"] = report_one(
            "SciFact (test, zero-shot for reranker)", s, labels, qids,
            "zeroshot=0.795 | adapted=0.964 (old CE=0.858)")

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
