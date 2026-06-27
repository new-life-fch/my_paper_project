#!/usr/bin/env python3
"""
Score the cross-domain (OOD) SciFact experiment: probe vs cross-encoder vs BM25.

Consumes the cache from build_ood_scifact.py. Trains the probe on SciFact's
train split (few-shot domain adaptation of the FROZEN LLM's probe), and scores
all three methods on the held-out SciFact test queries with per-query ranking
metrics (MRR, NDCG@10, Recall@k).

The key comparison: the cross-encoder is MS-MARCO-specialized and gets NO
SciFact training; the probe gets a small SciFact train split. If the probe
beats / closes the gap with the cross-encoder here (unlike in-domain MS MARCO
where it lost 0.685 vs 0.759), that demonstrates cross-domain robustness.

Usage:
  python scripts/compare_ood.py --cache results/cache/scifact_ood --topk 20
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

N_JOBS = max(1, min(48, os.cpu_count() - 2))


def lr(C=1.0, penalty="l2"):
    return LogisticRegression(penalty=penalty, C=C,
                              solver="lbfgs" if penalty == "l2" else "liblinear",
                              max_iter=2000, class_weight="balanced", random_state=42)


def load_acts(cache, split):
    d = torch.load(os.path.join(cache, f"attn_{split}.pt"))
    return d["acts"].to(torch.float32).numpy(), d["labels"].numpy(), d["query_ids"].numpy()


def auc(y, s):
    try:
        return roc_auc_score(y, s)
    except ValueError:
        return 0.5


def dcg(rels):
    return sum(r / np.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(labels_in_rank, k=10):
    ideal = sorted(labels_in_rank, reverse=True)
    idcg = dcg(ideal[:k])
    return dcg(labels_in_rank[:k]) / idcg if idcg > 0 else 0.0


def query_metrics(scores, labels):
    order = np.argsort(scores)[::-1]
    lab = np.array(labels)[order]
    n_rel = lab.sum()
    if n_rel == 0:
        return None
    first = np.argmax(lab == 1) + 1
    return {"mrr": 1.0 / first, "ndcg@10": ndcg_at_k(lab.tolist(), 10),
            "recall@1": float(lab[:1].sum())/n_rel, "recall@3": float(lab[:3].sum())/n_rel,
            "recall@5": float(lab[:5].sum())/n_rel}


def eval_method(name, scores, labels, qids, report):
    a = round(float(auc(labels, scores)), 3)
    by_q = defaultdict(lambda: ([], []))
    for s, l, q in zip(scores, labels, qids):
        by_q[q][0].append(s); by_q[q][1].append(l)
    pq = [query_metrics(s, l) for s, l in by_q.values()]
    pq = [m for m in pq if m]
    keys = ["mrr", "ndcg@10", "recall@1", "recall@3", "recall@5"]
    agg = {k: round(float(np.mean([m[k] for m in pq])), 3) for k in keys}
    report[name] = {"auc": a, "n_queries": len(pq), **agg}
    print(f"  {name:14s} AUC={a:.3f} MRR={agg['mrr']:.3f} NDCG@10={agg['ndcg@10']:.3f} "
          f"R@3={agg['recall@3']:.3f} R@5={agg['recall@5']:.3f} (q={len(pq)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/cache/scifact_ood")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--out", default="results/ood_comparison.json")
    args = ap.parse_args()

    sp = json.load(open(os.path.join(args.cache, "triples.json")))
    test = sp["test"]
    te_labels = np.array([t["label"] for t in test])
    te_qids = np.array([t["query_id"] for t in test])
    queries = [t["query"] for t in test]
    passages = [t["passage"] for t in test]
    bm25_scores = np.array([t["bm25"] for t in test])

    report = {}

    # ===== PROBE (train on SciFact train split) =====
    print("Training probe on SciFact train split...")
    Xtr, ytr, _ = load_acts(args.cache, "train")
    Xva, yva, _ = load_acts(args.cache, "val")
    Xte, yte, qte = load_acts(args.cache, "test")
    assert np.array_equal(yte, te_labels), "cache/text misalignment"

    n_heads = Xtr.shape[1]

    def hauc(h):
        p = lr(penalty="l1").fit(Xtr[:, h, :], ytr)
        return auc(yva, p.predict_proba(Xva[:, h, :])[:, 1])
    vaucs = np.array(Parallel(n_jobs=N_JOBS)(delayed(hauc)(h) for h in range(n_heads)))
    sel = np.argsort(vaucs)[::-1][:args.topk]

    def feats(X):
        return np.concatenate([X[:, h, :] for h in sel], axis=1)
    probe = lr().fit(feats(Xtr), ytr)
    eval_method("probe", probe.predict_proba(feats(Xte))[:, 1], yte, qte, report)

    # ===== BM25 (retrieval score, no training) =====
    eval_method("bm25", bm25_scores, te_labels, te_qids, report)

    # ===== CROSS-ENCODER (MS-MARCO specialized, NO SciFact training) =====
    print("Scoring cross-encoder (MS-MARCO-trained, zero SciFact adaptation)...")
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    ce_scores = ce.predict([[q, p] for q, p in zip(queries, passages)],
                           batch_size=64, show_progress_bar=False)
    eval_method("cross_encoder", np.asarray(ce_scores), te_labels, te_qids, report)

    print("\n=== OOD (SciFact) METHOD COMPARISON ===")
    print("Reference in-domain (MS MARCO): probe AUC 0.685 vs cross_encoder 0.759")
    hdr = f"{'method':14s} {'AUC':>6s} {'MRR':>6s} {'NDCG@10':>8s} {'R@3':>6s} {'R@5':>6s}"
    print(hdr); print("-"*len(hdr))
    for m, r in report.items():
        print(f"{m:14s} {r['auc']:6.3f} {r['mrr']:6.3f} {r['ndcg@10']:8.3f} "
              f"{r['recall@3']:6.3f} {r['recall@5']:6.3f}")

    json.dump({"dataset": "BeIR/scifact", "config": vars(args), "results": report},
              open(args.out, "w"), indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
