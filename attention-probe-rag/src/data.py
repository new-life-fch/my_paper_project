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
