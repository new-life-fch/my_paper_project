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
  python scripts/internals_vs_output.py --cache results/cache/q500_instruct
"""
import os, sys, argparse, json
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
    return d["acts"].to(torch.float32).numpy(), d["labels"].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/cache/q500_instruct")
    ap.add_argument("--judge-cache", default="results/cache/q500_judge")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--out", default="results/internals_vs_output.json")
    args = ap.parse_args()

    rep = {}

    # --- 1. LLM-judge (zero training) ---
    jd = torch.load(os.path.join(args.judge_cache, "judge_test.pt"))
    rep["llm_judge_zeroshot"] = round(float(auc(jd["labels"].numpy(), jd["scores"].numpy())), 3)

    # --- 2/3. resid per-layer trained probes ---
    Rtr, ytr = load(args.cache, "resid", "train")
    Rva, yva = load(args.cache, "resid", "val")
    Rte, yte = load(args.cache, "resid", "test")
    n_layers = Rtr.shape[1]

    def layer_auc(li):
        m = make_pipeline(StandardScaler(), lr(C=0.5)).fit(Rtr[:, li, :], ytr)
        return round(float(auc(yte, m.predict_proba(Rte[:, li, :])[:, 1])), 3)
    layer_aucs = Parallel(n_jobs=min(N_JOBS, n_layers))(
        delayed(layer_auc)(li) for li in range(n_layers))
    rep["resid_per_layer_test_auc"] = layer_aucs
    rep["resid_final_layer(output_repr)"] = layer_aucs[-1]
    rep["resid_best_layer"] = {"layer": int(np.argmax(layer_aucs)), "auc": max(layer_aucs)}

    # --- 4. attn-head probe (our method) ---
    Atr, _ = load(args.cache, "attn", "train")
    Ava, _ = load(args.cache, "attn", "val")
    Ate, _ = load(args.cache, "attn", "test")
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
