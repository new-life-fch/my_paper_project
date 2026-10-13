#!/usr/bin/env python3
"""
CAPACITY / INDUCTIVE-BIAS CONTROL for the headline claim (TODO.md R-4).

The obvious reviewer attack on "reading attention heads beats reading the output
representation" is that the head probe simply has more freedom: it runs a search
over 672-784 candidate heads, picks 20 on validation, and concatenates them.

What the measured configs actually say (docs/MODEL_ARCHITECTURES.md):
    top-20 heads x head_dim 128 = 2560 features   <   final resid 3072/3584/4096
so the head probe wins with FEWER features. The remaining difference is
  (a) the val-time search over heads, and
  (b) the inductive bias: 20 separately-standardised 128-d chunks concatenated,
      versus one 3072-d vector.
This script closes (a) and (b) in turn, on the SAME frozen caches:

  M0 final_linear   : LR on the final-layer residual                     (baseline)
  M1 attn_linear    : LR on top-k heads by val AUC, concatenated         (headline)
  M2 final_select20 : pick 20 of 128-d CHUNKS of the final residual by val
                      AUC, concatenate -> same feature count (2560), same
                      search procedure, same inductive bias, only the layer
                      differs. THE decisive control.
  M3 final_mlp      : MLP on the final residual (same param count ~ vanilla)
  M4 final_mlp_wide : MLP on the final residual (hidden = hidden_size)
  M5 final_headmean : average of 20 per-chunk LRs fit at the final layer
                      (ensemble readout instead of concatenation)

delta = M1 - M2 is the honest "reading depth, not readout flexibility" number.
If M2 catches M1, the paper's attn-final edge is an artefact of the readout
function, not of depth -- and the thesis must be rewritten to rest on best_resid.

Everything is fit on train, selected on val, reported on test, with a
query-clustered paired bootstrap for the deltas.

Usage:
  python scripts/capacity_control.py --cache results/cache/llama32_3b_q1500_instruct \
      --judge-cache results/cache/llama32_3b_q1500_judge --tag llama32_3b \
      --out results/capacity_control_llama32_3b_q1500.json
"""
import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cache_paths import resolve_cache  # noqa: E402

import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
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
    try: return float(roc_auc_score(y, s))
    except ValueError: return 0.5


def load(cache, scheme, split):
    d = torch.load(os.path.join(cache, f"{scheme}_{split}.pt"))
    return (d["acts"].to(torch.float32).numpy(), d["labels"].numpy(),
            d["query_ids"].numpy() if "query_ids" in d else None)


def boot_indices(qids, n_boot, rng):
    """Query-clustered bootstrap index sets (shared across readers = paired)."""
    idx_by_q = {q: np.where(qids == q)[0] for q in np.unique(qids)}
    uq = np.unique(qids)
    return [np.concatenate([idx_by_q[q] for q in rng.choice(uq, size=len(uq), replace=True)])
            for _ in range(n_boot)]


def boot_auc(y, s, sets):
    return np.array([auc(y[i], s[i]) for i in sets])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--judge-cache", default=None)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--topk", type=int, default=20, help="heads/chunks for M1/M2/M5")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--chunk-perms", type=int, default=8,
                    help="random chunk-boundary permutations for M2 robustness")
    ap.add_argument("--out", default="results/capacity_control.json")
    args = ap.parse_args()
    args.cache = resolve_cache(args.cache)
    if args.judge_cache:
        args.judge_cache = resolve_cache(args.judge_cache)

    Rtr, ytr, _ = load(args.cache, "resid", "train")
    Rva, yva, _ = load(args.cache, "resid", "val")
    Rte, yte, qte = load(args.cache, "resid", "test")
    Atr, _, _ = load(args.cache, "attn", "train")
    Ava, _, _ = load(args.cache, "attn", "val")
    Ate, _, _ = load(args.cache, "attn", "test")
    n_layers, hidden = Rtr.shape[1], Rtr.shape[2]
    n_heads = Atr.shape[1]

    rep = {"tag": args.tag, "cache": args.cache, "n_train": int(len(ytr)),
           "n_val": int(len(yva)), "n_test": int(len(yte)), "n_boot": int(args.boot),
           "n_layers": int(n_layers), "hidden": int(hidden),
           "n_attn_heads": int(n_heads), "topk": int(args.topk),
           "feature_counts": {"final_linear": int(hidden),
                              "attn_linear": int(args.topk * Ate.shape[2])},
           "readers": {}}

    # ---- reference: zero-shot judge -------------------------------------
    if args.judge_cache:
        jd = torch.load(os.path.join(args.judge_cache, "judge_test.pt"))
        assert np.array_equal(jd["labels"].numpy(), yte), "judge/resid label mismatch"
        if "query_ids" in jd and qte is not None:
            assert np.array_equal(jd["query_ids"].numpy(), qte), \
                "judge/resid QUERY_ID mismatch -- caches describe different rows"
        rep["judge_auc"] = round(auc(yte, jd["scores"].numpy()), 3)

    def score_model(fit_predict):
        """fit_predict() -> test scores (fit on train, select on val inside)."""
        s = fit_predict()
        return s

    # ---- M0: final-layer linear ------------------------------------------
    def m0():
        m = make_pipeline(StandardScaler(), lr()).fit(Rtr[:, -1, :], ytr)
        return m.predict_proba(Rte[:, -1, :])[:, 1]
    s0 = score_model(m0)

    # ---- M1: attn head probe (top-k heads by val) ------------------------
    def head_vauc(h):
        p = lr(penalty="l1").fit(Atr[:, h, :], ytr)
        return h, auc(yva, p.predict_proba(Ava[:, h, :])[:, 1])
    res = Parallel(n_jobs=N_JOBS)(delayed(head_vauc)(h) for h in range(n_heads))
    sel_heads = [int(h) for h, _ in sorted(res, key=lambda t: -t[1])[:args.topk]]

    def feats_attn(X):
        return np.concatenate([X[:, h, :] for h in sel_heads], axis=1)
    m1 = lr().fit(feats_attn(Atr), ytr)
    s1 = m1.predict_proba(feats_attn(Ate))[:, 1]

    # ---- M2: same procedure, but chunks of the FINAL residual ------------
    # The final residual is hidden-dim; split it into chunks of the same size as
    # a head (128) so the FEATURE COUNT (topk*128), the val-search and the
    # concatenation are identical to M1. Only "which layer" changes.
    #
    # Honest caveat: the selection POOL differs. M1 picks 20 of ~672-784 heads
    # (3%), M2 picks 20 of 24-32 chunks (63-83%), so M2's search is weaker even
    # at equal feature count. M2 is therefore a *friendly* control for the final
    # layer (it cannot be accused of giving the output representation too little
    # capacity); a gap that survives M2 is conservative evidence for depth.
    chunk = Ate.shape[2]                       # head_dim = 128
    n_chunks = hidden // chunk
    spans = [(i * chunk, (i + 1) * chunk) for i in range(n_chunks)]

    def chunk_vauc(ci):
        a, b = spans[ci]
        p = lr(penalty="l1").fit(Rtr[:, -1, a:b], ytr)
        return ci, auc(yva, p.predict_proba(Rva[:, -1, a:b])[:, 1])
    cres = Parallel(n_jobs=N_JOBS)(delayed(chunk_vauc)(c) for c in range(n_chunks))
    sel_chunks = [int(c) for c, _ in sorted(cres, key=lambda t: -t[1])[:args.topk]]

    def feats_final_chunks(X):
        return np.concatenate([X[:, -1, spans[c][0]:spans[c][1]] for c in sel_chunks], axis=1)
    m2 = lr().fit(feats_final_chunks(Rtr), ytr)
    s2 = m2.predict_proba(feats_final_chunks(Rte))[:, 1]

    # ---- M3/M4: nonlinear readouts on the final residual ----------------
    # M3 matches the vanilla probe's parameter count (2560 -> 2 hidden units);
    # M4 is generous (hidden = hidden_size) to test "give it real capacity".
    mlps = {}
    for name, h in (("final_mlp_sameparams", 2), ("final_mlp_wide", hidden)):
        p = make_pipeline(StandardScaler(),
                          MLPClassifier(hidden_layer_sizes=(h,), max_iter=400,
                                        early_stopping=True, n_iter_no_change=15,
                                        random_state=42))
        p.fit(Rtr[:, -1, :], ytr)
        mlps[name] = p.predict_proba(Rte[:, -1, :])[:, 1]

    # ---- M2b: robustness to the arbitrary chunk boundaries ----------------
    # Splitting a 3072-d vector into 128-d chunks depends on an arbitrary
    # coordinate order, so report the distribution over several permutations.
    perms = {}
    for pi in range(args.chunk_perms):
        prng = np.random.default_rng(1000 + pi)
        perm = prng.permutation(hidden)
        bound = [(perm[i * chunk:(i + 1) * chunk]) for i in range(n_chunks)]

        def p_vauc(ci):
            idx = bound[ci]
            p = lr(penalty="l1").fit(Rtr[:, -1, idx], ytr)
            return ci, auc(yva, p.predict_proba(Rva[:, -1, idx])[:, 1])
        pres = Parallel(n_jobs=N_JOBS)(delayed(p_vauc)(c) for c in range(n_chunks))
        psel = [int(c) for c, _ in sorted(pres, key=lambda t: -t[1])[:args.topk]]
        ftr = np.concatenate([Rtr[:, -1, bound[c]] for c in psel], axis=1)
        fte = np.concatenate([Rte[:, -1, bound[c]] for c in psel], axis=1)
        pm = lr().fit(ftr, ytr)
        perms[f"perm{pi}"] = round(auc(yte, pm.predict_proba(fte)[:, 1]), 4)
    rep["final_select20_chunks_perm_auc"] = perms
    if perms:
        vals = list(perms.values())
        rep["final_select20_chunks_perm_summary"] = {
            "mean": round(float(np.mean(vals)), 4),
            "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4),
        }

    # ---- M5: ensemble of per-chunk LRs at the final layer ---------------
    parts_te = []
    for c in sel_chunks:
        a, b = spans[c]
        p = lr().fit(Rtr[:, -1, a:b], ytr)
        parts_te.append(p.predict_proba(Rte[:, -1, a:b])[:, 1])
    s5 = np.mean(parts_te, axis=0)

    readers = {"final_linear": s0, "attn_linear": s1, "final_select20_chunks": s2,
               "final_headmean_ensemble": s5}
    readers.update(mlps)

    rng = np.random.default_rng(42)
    if qte is None:
        raise SystemExit("cache has no query_ids; cannot cluster the bootstrap")
    idx_sets = boot_indices(qte, args.boot, rng)
    boots = {k: boot_auc(yte, v, idx_sets) for k, v in readers.items()}
    for k, v in readers.items():
        b = boots[k]
        rep["readers"][k] = {
            "auc": round(auc(yte, v), 4),
            "ci95": [round(float(np.percentile(b, 2.5)), 4),
                     round(float(np.percentile(b, 97.5)), 4)],
        }

    def edge(hi, lo):
        d = boots[hi] - boots[lo]
        return {"mean": round(float(d.mean()), 4),
                "ci95": [round(float(np.percentile(d, 2.5)), 4),
                         round(float(np.percentile(d, 97.5)), 4)],
                "p_le0": round(float((d <= 0).mean()), 4)}
    rep["edges"] = {
        "attn-final (paper headline)":        edge("attn_linear", "final_linear"),
        "attn-select20chunks (KEY CONTROL)":  edge("attn_linear", "final_select20_chunks"),
        "select20chunks-final":               edge("final_select20_chunks", "final_linear"),
        "attn-final_headmean_ensemble":       edge("attn_linear", "final_headmean_ensemble"),
        "final_mlp_sameparams-final_linear":  edge("final_mlp_sameparams", "final_linear"),
        "final_mlp_wide-final_linear":        edge("final_mlp_wide", "final_linear"),
    }
    rep["selected_heads"] = sel_heads
    # heads are cached layer-major: h = layer * heads_per_layer + head
    heads_per_layer = n_heads // n_layers
    rep["selected_head_layers"] = [int(h // heads_per_layer) for h in sel_heads]
    rep["selected_chunks"] = sel_chunks

    # ---- top-k sweep: is attn - final stable across k? ------------------
    sweep = {}
    for k in [1, 5, 10, 20, 40, 80]:
        if k > n_heads: continue
        hs = [int(h) for h, _ in sorted(res, key=lambda t: -t[1])[:k]]
        mk = lr().fit(np.concatenate([Atr[:, h, :] for h in hs], axis=1), ytr)
        ak = auc(yte, mk.predict_proba(np.concatenate([Ate[:, h, :] for h in hs], axis=1))[:, 1])
        sweep[str(k)] = round(float(ak), 4)
    rep["attn_topk_sweep_auc"] = sweep

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(rep, open(args.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in rep.items()
                      if k not in ("selected_heads", "selected_head_layers", "selected_chunks")},
                     indent=2))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
