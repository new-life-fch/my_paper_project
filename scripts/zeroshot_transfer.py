#!/usr/bin/env python3
"""
ZERO-SHOT cross-domain transfer: the FAIR version of the SciFact OOD test.

Motivation (the gap compare_ood.py left open):
  compare_ood.py trained the probe on SciFact's own train split, while the
  cross-encoder saw zero SciFact data. That confounds "the probe transfers
  better" with "the probe got to peek at the target domain". A reviewer will
  reject 0.964 on that basis.

This script removes the confound. BOTH methods are zero-shot wrt SciFact:
  - PROBE: select heads on MS MARCO val, train the ensemble on MS MARCO train
           (instruct cache), then score SciFact test DIRECTLY. Never touches
           any SciFact label.
  - CROSS-ENCODER: ms-marco-MiniLM-L-6-v2, MS-MARCO-specialized, scored on
           SciFact test with zero adaptation (same as before).
  - BM25: retrieval score, no training.

Both probe and cross-encoder are trained only on MS MARCO and applied to a new
domain. If the probe still wins/ties, the cross-domain-robustness claim is
clean. We also report an "oracle / in-domain-adapted" probe (trained on SciFact,
from compare_ood.py = 0.964) as the upper bound, to bracket the effect.

Feature alignment: both caches use the SAME model / prompt / 672 heads, so a
probe fit on MS MARCO heads applies directly to SciFact activations. We DO fit
a StandardScaler on the MS MARCO train features and apply it to SciFact, because
the two domains have different activation magnitudes and an unscaled linear
probe transfers poorly.

Usage:
  python scripts/zeroshot_transfer.py \
      --src results/cache/q500_instruct \
      --tgt results/cache/scifact_ood \
      --topk 20
"""
import os
import sys
import argparse
import json
from collections import defaultdict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N_JOBS = max(1, min(48, (os.cpu_count() or 4) - 2))


def lr(C=1.0, penalty="l2"):
    return LogisticRegression(penalty=penalty, C=C,
                              solver="lbfgs" if penalty == "l2" else "liblinear",
                              max_iter=2000, class_weight="balanced", random_state=42)


def load_acts(cache, split):
    d = torch.load(os.path.join(cache, f"attn_{split}.pt"))
    return (d["acts"].to(torch.float32).numpy(), d["labels"].numpy(),
            d["query_ids"].numpy())


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
            "recall@1": float(lab[:1].sum()) / n_rel,
            "recall@3": float(lab[:3].sum()) / n_rel,
            "recall@5": float(lab[:5].sum()) / n_rel}


def eval_method(name, scores, labels, qids, report):
    a = round(float(auc(labels, scores)), 3)
    by_q = defaultdict(lambda: ([], []))
    for s, l, q in zip(scores, labels, qids):
        by_q[q][0].append(s)
        by_q[q][1].append(l)
    pq = [query_metrics(s, l) for s, l in by_q.values()]
    pq = [m for m in pq if m]
    keys = ["mrr", "ndcg@10", "recall@1", "recall@3", "recall@5"]
    agg = {k: round(float(np.mean([m[k] for m in pq])), 3) for k in keys}
    report[name] = {"auc": a, "n_queries": len(pq), **agg}
    print(f"  {name:24s} AUC={a:.3f} MRR={agg['mrr']:.3f} "
          f"NDCG@10={agg['ndcg@10']:.3f} R@3={agg['recall@3']:.3f} (q={len(pq)})")
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="results/cache/q500_instruct",
                    help="source-domain cache (probe trains here)")
    ap.add_argument("--tgt", default="results/cache/scifact_ood",
                    help="target-domain cache (probe evaluated here, zero-shot)")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--out", default="results/zeroshot_transfer.json")
    args = ap.parse_args()

    # ---- source domain (MS MARCO): select heads + fit ensemble ----
    print(f"[src] loading {args.src}")
    Xtr, ytr, _ = load_acts(args.src, "train")
    Xva, yva, _ = load_acts(args.src, "val")
    n_heads = Xtr.shape[1]
    print(f"[src] train={Xtr.shape} val={Xva.shape} heads={n_heads}")

    print("[src] selecting heads by MS MARCO val AUC (per-head L1)...")

    def hauc(h):
        p = lr(penalty="l1").fit(Xtr[:, h, :], ytr)
        return auc(yva, p.predict_proba(Xva[:, h, :])[:, 1])

    vaucs = np.array(Parallel(n_jobs=N_JOBS)(delayed(hauc)(h) for h in range(n_heads)))
    sel = np.argsort(vaucs)[::-1][:args.topk]
    print(f"[src] top-{args.topk} head val AUC range: "
          f"{vaucs[sel].min():.3f}..{vaucs[sel].max():.3f}")

    def feats(X):
        return np.concatenate([X[:, h, :] for h in sel], axis=1)

    # StandardScaler fit on SOURCE only, then frozen — critical for cross-domain
    # linear transfer (the two domains have different activation magnitudes).
    probe = make_pipeline(StandardScaler(), lr()).fit(feats(Xtr), ytr)
    src_val_auc = auc(yva, probe.predict_proba(feats(Xva))[:, 1])
    print(f"[src] probe MS MARCO val AUC (sanity, in-domain): {src_val_auc:.3f}")

    # ---- target domain (SciFact): zero-shot eval ----
    print(f"[tgt] loading {args.tgt} test")
    Xte, yte, qte = load_acts(args.tgt, "test")
    sp = json.load(open(os.path.join(args.tgt, "triples.json")))
    test = sp["test"]
    te_labels = np.array([t["label"] for t in test])
    assert np.array_equal(yte, te_labels), "tgt cache/text misalignment"
    queries = [t["query"] for t in test]
    passages = [t["passage"] for t in test]
    bm25_scores = np.array([t["bm25"] for t in test])

    report = {"_meta": {"src_val_auc_in_domain": round(float(src_val_auc), 3),
                        "topk": args.topk}}

    print("\n=== ZERO-SHOT TRANSFER to SciFact (no method sees SciFact labels) ===")
    eval_method("probe(MSMARCO->SciFact)",
                probe.predict_proba(feats(Xte))[:, 1], yte, qte, report)
    eval_method("bm25", bm25_scores, te_labels, qte, report)

    print("[tgt] scoring cross-encoder (MS-MARCO-specialized, zero SciFact adapt)...")
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    ce_scores = ce.predict([[q, p] for q, p in zip(queries, passages)],
                           batch_size=64, show_progress_bar=False)
    eval_method("cross_encoder", np.asarray(ce_scores), yte, qte, report)

    print("\n--- brackets ---")
    print("in-domain MS MARCO (compare_methods): probe 0.685 vs cross_encoder 0.759")
    print("SciFact-ADAPTED probe (compare_ood):  probe 0.964 (upper bound, peeked)")
    print("This table: probe trained ONLY on MS MARCO, applied zero-shot.")

    json.dump({"src": args.src, "tgt": args.tgt, "results": report},
              open(args.out, "w"), indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
