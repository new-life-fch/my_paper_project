#!/usr/bin/env python3
"""
internals>output CONTROL: is the probe's edge over LLM-judge "reading internals"
or merely "being trained"?

Hierarchy of relevance readers on the SAME model/prompt/answer-token (MS MARCO
in-domain, 500q instruct cache):
  1. LLM-judge      : model's literal output P("yes") vs P("no"), ZERO training. 0.535 on record.
  2. output-repr probe: TRAINED LR on the FINAL hidden state (resid layer L-1, the
     representation that is layernormed+projected to logits) — a trained reader of
     the OUTPUT representation.
  3. per-layer resid probe: TRAINED LR on each residual-stream layer (3072-d) — where
     along depth is relevance most linearly decodable?
  4. attn-head probe: TRAINED LR on top-k attention heads (our method). 0.685 on record.

Claim "internals know more than output" holds iff attn-head / mid-layer probes
beat BOTH the zero-shot judge AND the trained output-representation probe. If the
final-layer probe ~matches the attn-head probe, the gap vs judge is just
training, not depth — that would WEAKEN the claim and must be reported honestly.

Usage:
  python scripts/internals_vs_output.py --cache results/cache/llama32_3b_q500_instruct
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cache_paths import resolve_cache, warn_if_legacy_exists  # noqa: E402  (docs/CACHE_NAMING.md)

import argparse, json
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

N_JOBS = max(1, min(48, (os.cpu_count() or 4) - 2))


def lr(C=1.0, penalty="l2"):
    return LogisticRegression(penalty=penalty, C=C,
                              solver="lbfgs" if penalty == "l2" else "liblinear",
                              max_iter=3000, class_weight="balanced", random_state=42)


def auc(y, s):
    try: return roc_auc_score(y, s)
    except ValueError: return 0.5


def load(cache, scheme, split):
    d = torch.load(os.path.join(cache, f"{scheme}_{split}.pt"))
    qids = d["query_ids"].numpy() if "query_ids" in d else None
    return d["acts"].to(torch.float32).numpy(), d["labels"].numpy(), qids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/cache/llama32_3b_q500_instruct")
    ap.add_argument("--judge-cache", default="results/cache/llama32_3b_q500_judge")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--out", default="results/internals_vs_output.json")
    args = ap.parse_args()

    # honour pre-2026-10-12 cache names too (docs/CACHE_NAMING.md)
    args.cache = resolve_cache(args.cache)
    args.judge_cache = resolve_cache(args.judge_cache)

    rep = {}

    # --- 0. the two caches must describe the SAME rows -------------------
    # Labels alone are NOT sufficient: a different dataset revision / seed /
    # --max-passages produces the same label vector over different (query, passage)
    # pairs, which silently misaligns judge scores against probe features.
    # TODO.md P1-15; significance.py got this check on 2026-10-13, this script
    # was the one missing it (TODO.md R-2).
    _Rte_chk, yte_chk, qte_chk = load(args.cache, "resid", "test")
    jd = torch.load(os.path.join(args.judge_cache, "judge_test.pt"))
    assert np.array_equal(jd["labels"].numpy(), yte_chk), "judge/resid label mismatch"
    if "query_ids" in jd and qte_chk is not None:
        assert np.array_equal(jd["query_ids"].numpy(), qte_chk), \
            "judge/resid QUERY_ID mismatch -- caches describe different rows (TODO.md P1-15)"
    else:
        print("WARNING: judge cache has no query_ids; only labels were checked")

    # --- 1. LLM-judge (zero training) ---
    rep["llm_judge_zeroshot"] = round(float(auc(jd["labels"].numpy(), jd["scores"].numpy())), 3)

    # --- 2/3. resid per-layer trained probes ---
    Rtr, ytr, _ = load(args.cache, "resid", "train")
    Rva, yva, _ = load(args.cache, "resid", "val")
    Rte, yte, _ = load(args.cache, "resid", "test")
    n_layers = Rtr.shape[1]

    def layer_auc(li):
        m = make_pipeline(StandardScaler(), lr(C=0.5)).fit(Rtr[:, li, :], ytr)
        return round(float(auc(yte, m.predict_proba(Rte[:, li, :])[:, 1])), 3)
    # Layer selection MUST happen on validation, never on test (TODO.md P0-6/R-2).
    # This script used to argmax TEST AUCs, which inflated resid_best_layer by
    # ~+0.01-0.02 and contradicted the paper's own val-selected Table 1. Fit on
    # train, score val AND test for every layer, then select on val.
    def layer_aucs_both(li):
        m = make_pipeline(StandardScaler(), lr(C=0.5)).fit(Rtr[:, li, :], ytr)
        return (round(float(auc(yva, m.predict_proba(Rva[:, li, :])[:, 1])), 3),
                round(float(auc(yte, m.predict_proba(Rte[:, li, :])[:, 1])), 3))
    both = Parallel(n_jobs=min(N_JOBS, n_layers))(
        delayed(layer_aucs_both)(li) for li in range(n_layers))
    val_aucs = [b[0] for b in both]
    test_aucs = [b[1] for b in both]
    rep["resid_per_layer_val_auc"] = val_aucs
    rep["resid_per_layer_test_auc"] = test_aucs
    rep["resid_final_layer(output_repr)"] = test_aucs[-1]
    best_layer = int(np.argmax(val_aucs))
    rep["resid_best_layer"] = {
        "layer": best_layer,
        "auc": test_aucs[best_layer],
        "selection": "val",
        "val_auc_at_selected": val_aucs[best_layer],
    }
    # kept for transparency only -- DO NOT report this as a result, it is
    # selection-on-test and therefore optimistic (this is what the old field was).
    rep["resid_best_layer_test_argmax_DO_NOT_REPORT"] = {
        "layer": int(np.argmax(test_aucs)), "auc": max(test_aucs)}

    # --- 4. attn-head probe (our method) ---
    Atr, _, _ = load(args.cache, "attn", "train")
    Ava, _, _ = load(args.cache, "attn", "val")
    Ate, _, _ = load(args.cache, "attn", "test")
    n_heads = Atr.shape[1]

    def hauc(h):
        p = lr(penalty="l1").fit(Atr[:, h, :], ytr)
        return auc(yva, p.predict_proba(Ava[:, h, :])[:, 1])
    vaucs = np.array(Parallel(n_jobs=N_JOBS)(delayed(hauc)(h) for h in range(n_heads)))
    sel = np.argsort(vaucs)[::-1][:args.topk]
    def feats(X): return np.concatenate([X[:, h, :] for h in sel], axis=1)
    probe = lr().fit(feats(Atr), ytr)
    rep["attn_head_probe(ours)"] = round(float(auc(yte, probe.predict_proba(feats(Ate))[:, 1])), 3)

    print(json.dumps(rep, indent=2))
    print("\n--- reading ---")
    print(f"judge (output, 0-train)      : {rep['llm_judge_zeroshot']}")
    print(f"final-layer probe (output)   : {rep['resid_final_layer(output_repr)']}")
    print(f"best resid layer L{rep['resid_best_layer']['layer']:<2d} (internal) : {rep['resid_best_layer']['auc']}")
    print(f"attn-head probe (ours)       : {rep['attn_head_probe(ours)']}")
    print("internals>output holds iff internal probes > BOTH judge AND final-layer probe.")
    json.dump(rep, open(args.out, "w"), indent=2)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
