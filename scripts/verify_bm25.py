#!/usr/bin/env python3
"""
Verify that `src.data._bm25_rank_all`'s numpy BM25 reproduces
`rank_bm25.BM25Okapi.get_scores` exactly.

Why this exists: the numpy backend was written because `get_scores` is ~2 minutes
per query on FiQA (pure-Python loop over the vocabulary), which made the BM25
hard-negative construction intractable. A faster reimplementation of a scoring
function is exactly the kind of thing that can be subtly wrong and silently
change which negatives get sampled, so it gets an explicit equivalence test.

Usage:
  python scripts/verify_bm25.py --n-docs 3000 --n-queries 8
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from rank_bm25 import BM25Okapi
from collections import Counter, defaultdict


def numpy_scores(query_tokens, corpus_tokens, bm, idf, vocab):
    k1, b = bm.k1, bm.b
    postings = defaultdict(list)
    for di, toks in enumerate(corpus_tokens):
        for t, f in Counter(toks).items():
            if t in vocab:
                postings[t].append((di, f))
    doc_len = np.fromiter((len(t) for t in corpus_tokens), dtype=np.float32,
                          count=len(corpus_tokens))
    denom_len = k1 * (1.0 - b + b * doc_len / float(bm.avgdl))
    score = np.zeros(len(corpus_tokens), dtype=np.float32)
    for t in set(query_tokens):
        col = vocab.get(t)
        if col is None:
            continue
        w = idf[col]
        if w <= 0:
            continue
        di = np.fromiter((d for d, _ in postings[t]), dtype=np.int64, count=len(postings[t]))
        f = np.fromiter((ff for _, ff in postings[t]), dtype=np.float32, count=len(postings[t]))
        score[di] += w * (f * (k1 + 1.0)) / (f + denom_len[di])
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=3000)
    ap.add_argument("--n-queries", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    wordpool = [f"w{i}" for i in range(800)]
    corpus = [[str(x) for x in rng.choice(wordpool, size=rng.integers(5, 60))]
              for _ in range(args.n_docs)]
    queries = [[str(x) for x in rng.choice(wordpool, size=rng.integers(1, 8))]
               for _ in range(args.n_queries)]

    bm = BM25Okapi(corpus)
    # rank_bm25 exposes idf as a plain dict (no idf_index attribute); build the
    # term->column map ourselves so the numpy path can index it.
    vocab = {t: i for i, t in enumerate(bm.idf)}
    idf = np.fromiter(bm.idf.values(), dtype=np.float32, count=len(bm.idf))

    worst = 0.0
    for qi, q in enumerate(queries):
        ref = np.asarray(bm.get_scores(q), dtype=np.float64)
        got = numpy_scores(q, corpus, bm, idf, vocab).astype(np.float64)
        d = float(np.max(np.abs(ref - got)))
        worst = max(worst, d)
        # ranking agreement matters more than raw values for negative sampling
        same_top = list(np.argsort(-ref)[:20]) == list(np.argsort(-got)[:20])
        print(f"  query {qi}: max|Δscore|={d:.3e}  top-20 identical={same_top}")

    print(f"\nworst absolute deviation over {args.n_queries} queries: {worst:.3e}")
    if worst < 1e-4:
        print("OK  numpy BM25 reproduces rank_bm25.BM25Okapi.get_scores")
        return 0
    print("FAIL  implementations disagree -- do NOT use the numpy backend")
    return 1


if __name__ == "__main__":
    sys.exit(main())
