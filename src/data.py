"""
Data preparation module for Attention-Probe-RAG.

Loads MS MARCO Passage Ranking dataset and constructs (query, passage, label) triples.
Supports query-based splitting to prevent data leakage.
"""

import random
from datasets import load_dataset
from tqdm import tqdm


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
    ds = load_dataset("microsoft/ms_marco", "v1.1", split="train", trust_remote_code=True)

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


def load_beir_relevance(name: str = "fiqa", n_queries: int = 1500,
                        max_passages_per_query: int = 5, seed: int = 42):
    """
    Load a BeIR relevance dataset (e.g. fiqa, scifact) and build
    (query_id, query, passage, label) triples with the SAME schema as
    load_ms_marco, so the whole extraction/probe pipeline works unchanged.

    This is for the SECOND in-domain relevance experiment: a probe is trained
    AND tested within this dataset (no cross-domain transfer — that dead plan
    lives in workspace B). Goal: show the internals>output gap is not specific
    to MS MARCO.

    qrels mark only positives; negatives are sampled from corpus passages not
    judged relevant for that query (standard relevance-set construction).
    """
    print(f"Loading BeIR/{name} relevance dataset...")
    corpus = load_dataset(f"BeIR/{name}", "corpus", trust_remote_code=True)["corpus"]
    queries = load_dataset(f"BeIR/{name}", "queries", trust_remote_code=True)["queries"]
    qrels_all = load_dataset(f"BeIR/{name}-qrels", trust_remote_code=True)
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

    samples = []
    n_pos_per = max(1, max_passages_per_query // 2)
    for q in qids:
        pos_cids = pos_by_q[q][:n_pos_per]
        pos_set = set(pos_by_q[q])
        n_neg = max_passages_per_query - len(pos_cids)
        # sample negatives not judged relevant for this query
        negs = []
        tries = 0
        while len(negs) < n_neg and tries < n_neg * 20:
            c = rng.choice(all_cids)
            if c not in pos_set:
                negs.append(c)
            tries += 1
        # numeric query_id for downstream torch.tensor(query_ids)
        try:
            qid_num = int(q)
        except ValueError:
            qid_num = abs(hash(q)) % (10 ** 9)
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
