#!/usr/bin/env python3
"""
Cross-methodology comparison: probe vs RAG reranker baselines.

The user's real goal is to beat the RAG reranker (cross-encoder). This harness
compares, on IDENTICAL query-disjoint test passages, four relevance scorers
with per-QUERY ranking metrics (not just pooled AUC):

  probe        : our method — LR on instruct attn answer-token activations,
                 top-k heads by val AUC (trained on train split)
  llm_judge    : same model, P("yes") at answer token, zero training (cached)
  cross_encoder: cross-encoder/ms-marco-MiniLM-L-6-v2 (the reranker to beat)
  bm25         : sparse lexical baseline (per-query)

Alignment: caches were written in deterministic split order, so reloading
MS MARCO with the same seed reproduces the exact per-split sample order; we
align cached activations/scores to passage text by index (asserted via labels).

Metrics per query (queries with >=1 relevant passage):
  MRR, NDCG@10, Recall@1/3/5  + pooled ROC-AUC across all test samples.

Usage:
  python scripts/compare_methods.py --n-queries 500 --topk 20
"""
import os
import sys
import argparse
import json
from collections import defaultdict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_ms_marco, split_by_query


def lr(C=1.0, penalty="l2"):
    return LogisticRegression(penalty=penalty, C=C,
                              solver="lbfgs" if penalty == "l2" else "liblinear",
                              max_iter=2000, class_weight="balanced", random_state=42)


def load_acts(cache, scheme, split):
    d = torch.load(os.path.join(cache, f"{scheme}_{split}.pt"))
    return d["acts"].to(torch.float32).numpy(), d["labels"].numpy(), d["query_ids"].numpy()


# ---------- ranking metrics ----------
def dcg(rels):
    return sum(r / np.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(sorted_labels, k=10):
    ideal = sorted(sorted_labels, reverse=True)
    idcg = dcg(ideal[:k])
    return dcg(sorted_labels[:k]) / idcg if idcg > 0 else 0.0


def query_metrics(scores, labels):
    """Per-query MRR, NDCG@10, Recall@1/3/5 for one query's candidates."""
    order = np.argsort(scores)[::-1]
    lab = np.array(labels)[order]
    n_rel = lab.sum()
    if n_rel == 0:
        return None
    # MRR: reciprocal rank of first relevant
    first = np.argmax(lab == 1) + 1
    mrr = 1.0 / first
    ndcg = ndcg_at_k(lab.tolist(), k=10)
    rec = {f"recall@{k}": float(lab[:k].sum()) / n_rel for k in (1, 3, 5)}
    return {"mrr": mrr, "ndcg@10": ndcg, **rec}


def aggregate(per_query):
    pq = [m for m in per_query if m is not None]
    keys = ["mrr", "ndcg@10", "recall@1", "recall@3", "recall@5"]
    return {k: round(float(np.mean([m[k] for m in pq])), 3) for k in keys}, len(pq)


def eval_method(name, scores, labels, qids, report):
    # pooled AUC
    try:
        auc = round(float(roc_auc_score(labels, scores)), 3)
    except ValueError:
        auc = 0.5
    # per-query ranking
    by_q = defaultdict(lambda: ([], []))
    for s, l, q in zip(scores, labels, qids):
        by_q[q][0].append(s); by_q[q][1].append(l)
    per_query = [query_metrics(s, l) for s, l in by_q.values()]
    agg, n_q = aggregate(per_query)
    report[name] = {"auc": auc, "n_queries": n_q, **agg}
    print(f"  {name:14s} AUC={auc:.3f}  MRR={agg['mrr']:.3f}  NDCG@10={agg['ndcg@10']:.3f}  "
          f"R@1={agg['recall@1']:.3f} R@3={agg['recall@3']:.3f}  (q={n_q})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=500)
    ap.add_argument("--max-passages", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--instruct-cache", default="results/cache/q500_instruct")
    ap.add_argument("--judge-cache", default="results/cache/q500_judge")
    ap.add_argument("--scheme", default="attn")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--out", default="results/method_comparison.json")
    args = ap.parse_args()

    # --- reload text in deterministic split order ---
    samples = load_ms_marco(n_queries=args.n_queries,
                            max_passages_per_query=args.max_passages, seed=args.seed)
    tr_s, va_s, te_s = split_by_query(samples, seed=args.seed)
    te_labels_txt = np.array([x["label"] for x in te_s])
    te_qids_txt = np.array([x["query_id"] for x in te_s])
    queries = [x["query"] for x in te_s]
    passages = [x["passage"] for x in te_s]

    report = {}

    # ============ PROBE ============
    print("Training probe (instruct attn, top-k val-AUC, L2)...")
    Xtr, ytr, _ = load_acts(args.instruct_cache, args.scheme, "train")
    Xva, yva, _ = load_acts(args.instruct_cache, args.scheme, "val")
    Xte, yte, qte = load_acts(args.instruct_cache, args.scheme, "test")
    # sanity: cached test order matches reloaded text order
    assert np.array_equal(yte, te_labels_txt), "cache/text misalignment (labels)"
    assert np.array_equal(qte, te_qids_txt), "cache/text misalignment (qids)"

    n_heads = Xtr.shape[1]

    def head_val_auc(h):
        p = lr(penalty="l1").fit(Xtr[:, h, :], ytr)
        try:
            return roc_auc_score(yva, p.predict_proba(Xva[:, h, :])[:, 1])
        except ValueError:
            return 0.5
    val_aucs = np.array(Parallel(n_jobs=min(48, os.cpu_count() - 2))(
        delayed(head_val_auc)(h) for h in range(n_heads)))
    sel = np.argsort(val_aucs)[::-1][:args.topk]

    def feats(X):
        return np.concatenate([X[:, h, :] for h in sel], axis=1)
    probe = lr().fit(feats(Xtr), ytr)
    probe_scores = probe.predict_proba(feats(Xte))[:, 1]
    eval_method("probe", probe_scores, yte, qte, report)

    # ============ LLM-JUDGE (cached) ============
    jd = torch.load(os.path.join(args.judge_cache, "judge_test.pt"))
    j_scores, j_labels, j_qids = jd["scores"].numpy(), jd["labels"].numpy(), jd["query_ids"].numpy()
    assert np.array_equal(j_labels, te_labels_txt), "judge cache misaligned"
    eval_method("llm_judge", j_scores, j_labels, j_qids, report)

    # ============ CROSS-ENCODER ============
    print("Scoring cross-encoder ms-marco-MiniLM-L-6-v2...")
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    ce_scores = ce.predict([[q, p] for q, p in zip(queries, passages)],
                            batch_size=64, show_progress_bar=False)
    eval_method("cross_encoder", np.asarray(ce_scores), te_labels_txt, te_qids_txt, report)

    # ============ BM25 (per-query) ============
    print("Scoring BM25 (per-query)...")
    from rank_bm25 import BM25Okapi
    bm_scores = np.zeros(len(te_s))
    by_q = defaultdict(list)
    for i, q in enumerate(te_qids_txt):
        by_q[q].append(i)
    for q, idxs in by_q.items():
        corpus = [passages[i].lower().split() for i in idxs]
        bm = BM25Okapi(corpus)
        query_tokens = queries[idxs[0]].lower().split()
        s = bm.get_scores(query_tokens)
        for j, i in enumerate(idxs):
            bm_scores[i] = s[j]
    eval_method("bm25", bm_scores, te_labels_txt, te_qids_txt, report)

    # --- summary ---
    print("\n=== METHOD COMPARISON (MS MARCO v1.1, 500q, query-disjoint test) ===")
    hdr = f"{'method':14s} {'AUC':>6s} {'MRR':>6s} {'NDCG@10':>8s} {'R@1':>6s} {'R@3':>6s}"
    print(hdr); print("-" * len(hdr))
    for m, r in report.items():
        print(f"{m:14s} {r['auc']:6.3f} {r['mrr']:6.3f} {r['ndcg@10']:8.3f} "
              f"{r['recall@1']:6.3f} {r['recall@3']:6.3f}")

    json.dump({"config": vars(args), "results": report}, open(args.out, "w"), indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
