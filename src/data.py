"""
Data preparation module for Attention-Probe-RAG.

Loads MS MARCO Passage Ranking dataset and constructs (query, passage, label) triples.
Supports query-based splitting to prevent data leakage.
"""

import random
import zlib

import numpy as np

from datasets import load_dataset
from tqdm import tqdm


def stable_query_id(text: str) -> int:
    """Deterministic positive int id for a query string.

    Why not `hash(str)`: Python randomises str hashing per process (PYTHONHASHSEED),
    so the same corpus produced a DIFFERENT id map on every run -- the FiQA judge
    and the FiQA probe could then train/test on different query sets.  Documented
    as TODO.md P1-16; zlib.crc32 is stable across processes and machines.
    """
    return zlib.crc32(str(text).encode("utf-8")) % (10 ** 9)


def load_ms_marco(n_queries: int = 100, max_passages_per_query: int = 5, seed: int = 42):
    """
    Load MS MARCO v1.1 and construct (query, passage, label) triples.

    Args:
        n_queries: Number of queries to sample (Phase 1 uses small subset).
        max_passages_per_query: Max passages per query (controls data size).
        seed: Random seed for reproducibility.

    Returns:
        List of dicts: {query_id, query, passage, label}
    """
    print("Loading MS MARCO v1.1 dataset...")
    # NOTE 2026-10-13: `trust_remote_code=True` was removed -- datasets 5.x no
    # longer accepts it and RAISES instead of ignoring it.  MS MARCO v1.1 is
    # plain Parquet on the Hub (dataset card points at v1.1/*.parquet), so no
    # loading script is needed and the data/labels are unchanged.
    ds = load_dataset("microsoft/ms_marco", "v1.1", split="train")

    random.seed(seed)
    # Sample a subset of indices for efficiency
    sample_size = min(n_queries, len(ds))
    sample_indices = random.sample(range(len(ds)), sample_size)

    samples = []
    for idx in tqdm(sample_indices, desc="Building triples"):
        item = ds[idx]
        query = item["query"]
        query_id = item["query_id"]
        passages = item["passages"]["passage_text"]
        is_selected = item["passages"]["is_selected"]

        # Collect positive and negative passages
        positives = []
        negatives = []
        for p, s in zip(passages, is_selected):
            if s == 1:
                positives.append(p)
            else:
                negatives.append(p)

        # Balance: up to max_passages_per_query total, keeping natural ratio
        n_pos = min(len(positives), max_passages_per_query // 2)
        n_neg = min(len(negatives), max_passages_per_query - n_pos)

        for p in positives[:n_pos]:
            samples.append({
                "query_id": query_id,
                "query": query,
                "passage": p,
                "label": 1,
            })
        for p in negatives[:n_neg]:
            samples.append({
                "query_id": query_id,
                "query": query,
                "passage": p,
                "label": 0,
            })

    print(f"Built {len(samples)} samples from {sample_size} queries "
          f"(pos: {sum(1 for s in samples if s['label'] == 1)}, "
          f"neg: {sum(1 for s in samples if s['label'] == 0)})")
    return samples


def _bm25_rank_all(qids, qid2text, all_cids, cid2text, pool=300,
                   dataset="fiqa", seed=42):
    """Top-`pool` BM25 passages per query, computed with numpy and cached.

    Why not `rank_bm25.BM25Okapi.get_scores`: it loops over the bag-of-words
    vocabulary in pure Python -- measured ~2 MINUTES per query on FiQA's 57k
    passages, i.e. tens of hours for 1500 queries. BM25 only scores terms that
    occur in the query, so we build inverted posting lists ONCE and then evaluate
    the identical formula touching only the query's own terms:

        score(q,d) = sum_{t in q} idf[t] * f(t,d)*(k1+1)
                                 / (f(t,d) + k1*(1 - b + b*|d|/avgdl))

    `idf`, `k1=1.5`, `b=0.75` are taken from BM25Okapi itself, and non-positive
    idf values are zeroed exactly as the library does. The result is verified
    bit-identical against `BM25Okapi.get_scores` in scripts/verify_bm25.py.

    The ranking is deterministic, so it is cached under results/cache/bm25/ and
    reused by both extraction and judging.
    """
    import os, json
    cache = os.path.join("results", "cache", "bm25", f"{dataset}_pool{pool}_seed{seed}.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)

    from rank_bm25 import BM25Okapi
    from collections import Counter, defaultdict

    def tok(t):
        return [w for w in str(t).lower().split() if w]

    corpus_tokens = [tok(cid2text[c]) for c in all_cids]
    print(f"  BM25: tokenising {len(all_cids)} passages ...")
    bm = BM25Okapi(corpus_tokens)             # only for idf / avgdl / defaults
    k1, b = bm.k1, bm.b
    # rank_bm25 exposes idf as a plain dict (no idf_index attribute); build the
    # term->column map ourselves so the numpy path can index it.
    vocab = {t: i for i, t in enumerate(bm.idf)}
    idf = np.fromiter(bm.idf.values(), dtype=np.float32, count=len(bm.idf))

    postings = defaultdict(list)              # term -> [(doc_idx, tf), ...]
    for di, toks in enumerate(corpus_tokens):
        for t, f in Counter(toks).items():
            if t in vocab:
                postings[t].append((di, f))
    postings = {t: (np.fromiter((d for d, _ in v), dtype=np.int64, count=len(v)),
                    np.fromiter((f for _, f in v), dtype=np.float32, count=len(v)))
                for t, v in postings.items()}
    doc_len = np.fromiter((len(t) for t in corpus_tokens), dtype=np.float32,
                          count=len(corpus_tokens))
    denom_len = k1 * (1.0 - b + b * doc_len / float(bm.avgdl))
    D = len(corpus_tokens)
    print(f"  BM25: {sum(len(v[0]) for v in postings.values())} (doc,term) pairs, "
          f"vocab {len(vocab)}; ranking {len(qids)} queries ...")

    ranked = {}
    for i, q in enumerate(qids):
        score = np.zeros(D, dtype=np.float32)
        for t in set(tok(qid2text[q])):
            col = vocab.get(t)
            if col is None:
                continue
            w = idf[col]
            if w <= 0:                        # library zeroes non-positive idf
                continue
            di, f = postings[t]
            score[di] += w * (f * (k1 + 1.0)) / (f + denom_len[di])
        n = min(pool, D)
        top = np.argpartition(-score, n - 1)[:n]
        top = top[np.argsort(-score[top])]
        ranked[q] = [all_cids[int(j)] for j in top]
        if (i + 1) % 300 == 0:
            print(f"    ranked {i+1}/{len(qids)} queries", flush=True)

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "w") as f:
        json.dump(ranked, f)
    print(f"  BM25 rankings cached -> {cache}")
    return ranked


def load_beir_relevance(name: str = "fiqa", n_queries: int = 1500,
                        max_passages_per_query: int = 5, seed: int = 42,
                        negatives: str = "bm25", bm25_pool: int = 100):
    """
    Load a BeIR relevance dataset (e.g. fiqa, scifact) and build
    (query_id, query, passage, label) triples with the SAME schema as
    load_ms_marco, so the whole extraction/probe pipeline works unchanged.

    This is for the SECOND in-domain relevance experiment: a probe is trained
    AND tested within this dataset (no cross-domain transfer — that dead plan
    lives in workspace B). Goal: show the internals>output gap is not specific
    to MS MARCO.

    qrels mark only positives; negatives come from corpus passages not judged
    relevant for that query.

    `negatives` (2026-10-14, user decision — see TODO.md「已决定」):
      "random" — original behaviour: uniformly sample the corpus. Correct but
                 EASY; on FiQA it drove the final-layer probe to ~0.976 AUC, i.e.
                 a ceiling effect that leaves no room for an internal>output edge
                 and therefore makes the second dataset weak evidence.
      "bm25"   — DEFAULT now: retrieve the top `bm25_pool` passages with BM25 for
                 the query and sample negatives from those that qrels do NOT mark
                 relevant. This mirrors MS MARCO's own hard negatives (which are
                 BM25-retrieved but unselected passages, see load_ms_marco) and
                 makes the two datasets comparable in difficulty. BM25 ranking is
                 deterministic, so the cache stays reproducible.
    """
    print(f"Loading BeIR/{name} relevance dataset (negatives={negatives})...")
    # `trust_remote_code` removed for datasets 5.x (see load_ms_marco).
    corpus = load_dataset(f"BeIR/{name}", "corpus")["corpus"]
    queries = load_dataset(f"BeIR/{name}", "queries")["queries"]
    qrels_all = load_dataset(f"BeIR/{name}-qrels")
    # Pool qrels across splits (we re-split by query ourselves downstream).
    import itertools
    qrels = list(itertools.chain.from_iterable(qrels_all[s] for s in qrels_all))

    # id -> text maps
    cid2text = {str(r["_id"]): (r["text"] or "") for r in corpus}
    qid2text = {str(r["_id"]): (r["text"] or "") for r in queries}
    all_cids = list(cid2text.keys())

    # positives grouped by query
    pos_by_q = {}
    for r in qrels:
        if int(r["score"]) <= 0:
            continue
        q = str(r["query-id"]); c = str(r["corpus-id"])
        if q in qid2text and c in cid2text:
            pos_by_q.setdefault(q, []).append(c)

    rng = random.Random(seed)
    qids = [q for q in pos_by_q if pos_by_q[q]]
    rng.shuffle(qids)
    qids = qids[:n_queries]

    # ---- BM25 retrieval (optional, but the default for hard negatives) ----
    # NOTE (2026-10-14): `rank_bm25.BM25Okapi.get_scores` loops over the whole
    # vocabulary in pure Python -- measured ~2 MINUTES per query on FiQA's 57k
    # passages, i.e. tens of hours for 1500 queries. `_bm25_rank_all` below
    # computes the identical formula with numpy (only the query's own terms are
    # touched) and caches the ranking, which makes this tractable. The two
    # implementations are verified to agree exactly (see scripts/verify_bm25.py).
    bm25_ranked = {}
    if negatives == "bm25":
        bm25_ranked = _bm25_rank_all(qids, qid2text, all_cids, cid2text,
                                     pool=bm25_pool, dataset=name, seed=seed)

    samples = []
    n_pos_per = max(1, max_passages_per_query // 2)
    for q in qids:
        pos_cids = pos_by_q[q][:n_pos_per]
        pos_set = set(pos_by_q[q])
        n_neg = max_passages_per_query - len(pos_cids)
        negs = []
        if negatives == "bm25":
            for c in bm25_ranked.get(q, []):
                if c not in pos_set:
                    negs.append(c)
                if len(negs) >= n_neg:
                    break
            if len(negs) < n_neg:
                print(f"  WARNING query {q}: only {len(negs)}/{n_neg} BM25 negatives "
                      f"(pool exhausted)")
        else:
            tries = 0
            while len(negs) < n_neg and tries < n_neg * 20:
                c = rng.choice(all_cids)
                if c not in pos_set and c not in negs:   # also de-dup (R-7)
                    negs.append(c)
                tries += 1
        # numeric query_id for downstream torch.tensor(query_ids)
        try:
            qid_num = int(q)
        except ValueError:
            # stable across processes (was abs(hash(q)) -> random per run, TODO.md P1-16)
            qid_num = stable_query_id(q)
        for c in pos_cids:
            samples.append({"query_id": qid_num, "query": qid2text[q],
                            "passage": cid2text[c], "label": 1})
        for c in negs:
            samples.append({"query_id": qid_num, "query": qid2text[q],
                            "passage": cid2text[c], "label": 0})

    print(f"Built {len(samples)} samples from {len(qids)} queries "
          f"(pos: {sum(s['label'] for s in samples)}, "
          f"neg: {sum(1 for s in samples if s['label'] == 0)})")
    return samples


def split_by_query(samples, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Split samples by query_id to prevent data leakage.

    All passages for the same query go to the same split.

    Args:
        samples: List of sample dicts from load_ms_marco.
        train_ratio: Fraction of queries for training.
        val_ratio: Fraction of queries for validation.
        seed: Random seed.

    Returns:
        (train_samples, val_samples, test_samples)
    """
    random.seed(seed)

    # Group by query_id
    query_ids = list(set(s["query_id"] for s in samples))
    random.shuffle(query_ids)

    n = len(query_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_qids = set(query_ids[:n_train])
    val_qids = set(query_ids[n_train:n_train + n_val])
    # Remaining queries go to test

    train_samples = [s for s in samples if s["query_id"] in train_qids]
    val_samples = [s for s in samples if s["query_id"] in val_qids]
    test_samples = [s for s in samples if s["query_id"] not in train_qids and s["query_id"] not in val_qids]

    print(f"Split: train={len(train_samples)} samples ({len(train_qids)} queries), "
          f"val={len(val_samples)} samples ({len(val_qids)} queries), "
          f"test={len(test_samples)} samples ({n - n_train - n_val} queries)")

    return train_samples, val_samples, test_samples
