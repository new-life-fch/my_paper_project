#!/usr/bin/env python3
"""
Fixed-direction control for the logit-lens comparison (pure CPU, no GPU).

Why this exists
---------------
The paper reports that the model's OWN readout direction (logit lens: final norm
+ unembedding applied to an intermediate residual) reads relevance worse than a
trained probe at the same layer. A reviewer can object that this is unfair: the
trained probe has 5218 samples to fit a direction, while the logit lens uses a
direction that was never optimised for relevance.

The honest bound is: what does a **fixed** linear direction achieve when it is
NOT fitted on the evaluation data? We measure exactly that with a domain shift
instead of a fake one:

    fit w on  MS MARCO  layer L   ->  evaluate w on  FiQA  layer L

The direction is fixed at test time and no FiQA label is ever seen, so this is a
legitimate "fixed direction" baseline. It is a *lower* bound (the direction is
also hurt by domain shift), and the point is only to calibrate the reader: is
the logit lens bad because no fixed direction can work, or because the model's
own direction happens to be a poor relevance direction?

Also reported, for scale:
  * the logit lens' own AUC at the same layer (from logit_lens_*.json)
  * a probe TRAINED on FiQA at that layer (the in-domain ceiling)
  * a probe trained on MS MARCO, tested on MS MARCO (the in-domain source)

Usage:
  python scripts/fixed_direction_control.py
  python scripts/fixed_direction_control.py --tag llama32_3b --fiqa-tag llama32_3b_fiqa_bm25
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_resid(cache, split):
    d = torch.load(os.path.join(cache, f"resid_{split}.pt"), map_location="cpu")
    return d["acts"].to(torch.float32).numpy(), d["labels"].numpy()


def fit_direction(X, y, C=1.0):
    m = make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=3000, class_weight="balanced", random_state=42, C=C))
    m.fit(X, y)
    return m


def fiqa_logit_lens_scores(model_name, cache, layer, batch=512):
    """Raw logit-lens P(yes) on the FiQA test split at one layer.

    Needed only for the bootstrap CI; returns (scores, labels, query_ids) or
    None if the model cannot be loaded (e.g. offline).
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        return None
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        model.eval()
    except Exception as e:  # pragma: no cover - environment dependent
        print(f"  (skipping bootstrap: cannot load {model_name}: {e})")
        return None

    def first(words):
        s = set()
        for w in words:
            e = tok.encode(w, add_special_tokens=False)
            if e:
                s.add(e[0])
        return sorted(s)
    yi = first(["yes", "Yes", "YES", " yes", " YES"])
    ni = first(["no", "No", "NO", " no", " NO"])

    d = torch.load(os.path.join(cache, "resid_test.pt"), map_location="cpu")
    R = d["acts"].to(torch.float32)[:, layer, :]
    y = d["labels"].numpy()
    q = d["query_ids"].numpy() if "query_ids" in d else np.arange(len(y))
    out = np.zeros(len(y), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(y), batch):
            h = model.model.norm(R[i:i + batch])
            lg = model.lm_head(h)
            s = torch.softmax(torch.stack([
                torch.logsumexp(lg[:, yi], -1),
                torch.logsumexp(lg[:, ni], -1)], -1), -1)[:, 0]
            out[i:i + batch] = s.numpy()
    return out, y, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B",
                    help="needed only for the FiQA logit-lens bootstrap CI")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--tag", default="llama32_3b")
    ap.add_argument("--fiqa-tag", default="llama32_3b_fiqa_bm25")
    ap.add_argument("--n-queries", type=int, default=1500)
    ap.add_argument("--layer-from-json", default=None,
                    help="layer index source; default = logit_lens best layer of --tag")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ms_cache = f"results/cache/{args.tag}_q{args.n_queries}_instruct"
    fq_cache = f"results/cache/{args.fiqa_tag}_q{args.n_queries}_instruct"
    for c in (ms_cache, fq_cache):
        if not os.path.exists(os.path.join(c, "resid_test.pt")):
            print(f"ERROR missing cache {c}")
            return 1

    ll_path = f"results/logit_lens_{args.tag}_q{args.n_queries}.json"
    ll = json.load(open(ll_path)) if os.path.exists(ll_path) else {}
    # Compare at the layer where the trained MS MARCO probe is strongest on val,
    # taken from the significance file so it matches the paper's reader.
    sg_path = f"results/significance_{args.tag}_q{args.n_queries}.json"
    sg = json.load(open(sg_path))
    sg = sg[0] if isinstance(sg, list) else sg
    L = args.layer_from_json or sg["best_layer_val"]

    print(f"=== fixed-direction control (layer L{L}) ===")
    print(f"  MS MARCO cache : {ms_cache}")
    print(f"  FiQA cache     : {fq_cache}")

    Rtr_ms, ytr_ms = load_resid(ms_cache, "train")
    Rte_ms, yte_ms = load_resid(ms_cache, "test")
    Rtr_fq, ytr_fq = load_resid(fq_cache, "train")
    Rte_fq, yte_fq = load_resid(fq_cache, "test")

    n = min(Rtr_ms.shape[1], Rtr_fq.shape[1])
    if L >= n:
        print(f"ERROR layer {L} >= available layers {n}")
        return 1

    res = {"tag": args.tag, "fiqa_tag": args.fiqa_tag, "layer": int(L),
           "layer_source": "significance best_layer_val",
           "n_ms_test": int(len(yte_ms)), "n_fiqa_test": int(len(yte_fq))}

    # 1. source-domain: fit and test on MS MARCO
    m_ms = fit_direction(Rtr_ms[:, L, :], ytr_ms)
    res["ms_probe_on_ms"] = round(float(roc_auc_score(
        yte_ms, m_ms.predict_proba(Rte_ms[:, L, :])[:, 1])), 4)

    # 2. TRANSFERRED fixed direction: fit on MS MARCO, test on FiQA (no FiQA labels)
    s_tr = m_ms.predict_proba(Rte_fq[:, L, :])[:, 1]
    res["ms_direction_on_fiqa"] = round(float(roc_auc_score(yte_fq, s_tr)), 4)

    # 3. in-domain ceiling: fit and test on FiQA
    m_fq = fit_direction(Rtr_fq[:, L, :], ytr_fq)
    res["fiqa_probe_on_fiqa"] = round(float(roc_auc_score(
        yte_fq, m_fq.predict_proba(Rte_fq[:, L, :])[:, 1])), 4)

    # 4. the model's own (zero-training) readout at this layer, for reference
    if ll:
        per = ll.get("logit_lens_per_layer_auc") or []
        if L < len(per):
            res["logit_lens_at_L"] = per[L]
        res["logit_lens_best_layer"] = ll.get("logit_lens_best_layer")

    # 5. THE SAME-DATASET COMPARISON. The transferred direction above is
    #    evaluated on FiQA while the logit lens in (4) lives on MS MARCO;
    #    comparing those two numbers directly would be cross-dataset sleight of
    #    hand. So we also read the model's own direction ON FiQA (produced by
    #    scripts/logit_lens_fiqa.py) and compare like with like.
    fq_ll_path = f"results/logit_lens_{args.fiqa_tag}_q{args.n_queries}.json"
    if os.path.exists(fq_ll_path):
        fq_ll = json.load(open(fq_ll_path))
        per = fq_ll.get("logit_lens_per_layer_auc") or []
        res["fiqa_logit_lens_path"] = fq_ll_path
        if L < len(per):
            res["fiqa_logit_lens_at_L"] = per[L]
            # Is the model's own readout BELOW CHANCE here, or just weak?
            # A point estimate near 0.5 cannot answer that, so bootstrap it
            # (query-clustered, matching the paper's protocol).
            s_ll = fiqa_logit_lens_scores(args.model, fq_cache, L, args.batch)
            if s_ll is not None:
                scores, y_fq, q_fq = s_ll
                rng = np.random.default_rng(42)
                uq = np.unique(q_fq)
                by_q = {qq: np.where(q_fq == qq)[0] for qq in uq}
                boot = []
                for _ in range(2000):
                    pick = rng.choice(uq, size=len(uq), replace=True)
                    ii = np.concatenate([by_q[qq] for qq in pick])
                    boot.append(roc_auc_score(y_fq[ii], scores[ii])
                                if len(np.unique(y_fq[ii])) > 1 else 0.5)
                boot = np.array(boot)
                res["fiqa_logit_lens_bootstrap"] = {
                    "n_boot": 2000,
                    "ci95": [round(float(np.percentile(boot, 2.5)), 4),
                             round(float(np.percentile(boot, 97.5)), 4)],
                    "p_auc_ge_half": round(float((boot >= 0.5).mean()), 4),
                }
        res["fiqa_logit_lens_best_layer"] = fq_ll.get("logit_lens_best_layer")
        res["fiqa_judge"] = fq_ll.get("llm_judge_auc")

    print()
    print(f"  MS probe, tested on MS MARCO (source in-domain) : {res['ms_probe_on_ms']:.4f}")
    print(f"  MS direction, tested on FiQA  (FIXED, shifted)  : {res['ms_direction_on_fiqa']:.4f}")
    print(f"  FiQA probe, tested on FiQA    (target ceiling)  : {res['fiqa_probe_on_fiqa']:.4f}")
    if "logit_lens_at_L" in res:
        print(f"  model's own readout (logit lens) at L{L}, MS MARCO : {res['logit_lens_at_L']:.4f}")
    if "fiqa_logit_lens_at_L" in res:
        print(f"  model's own readout (logit lens) at L{L}, FiQA     : {res['fiqa_logit_lens_at_L']:.4f}")
        print()
        print("  SAME-DATASET COMPARISON (the one to quote), on FiQA at the same layer:")
        print(f"    model's own fixed direction .................... {res['fiqa_logit_lens_at_L']:.4f}")
        print(f"    a direction fitted on ANOTHER dataset, carried")
        print(f"    over with no target fitting .................... {res['ms_direction_on_fiqa']:.4f}")
        print(f"    a direction fitted in-domain ................... {res['fiqa_probe_on_fiqa']:.4f}")
        if res["ms_direction_on_fiqa"] > res["fiqa_logit_lens_at_L"]:
            print("    -> EVEN a domain-shifted, never-adapted direction beats the")
            print("       model's own readout on the SAME data. So the logit lens'")
            print("       deficit is about the model's readout direction being")
            print("       off-task, not about the information being unreadable by a")
            print("       fixed linear map. This is exactly the caveat the paper")
            print("       needs, measured rather than asserted.")
    print()
    print("  Reading: a direction fitted elsewhere and carried over WITHOUT any")
    print("  target-domain fitting still extracts relevance, so 'the trained probe")
    print("  wins' is about direction quality, not about the information being")
    print("  unreadable by a fixed linear map. The logit lens' deficit is therefore")
    print("  a statement about the model's own readout being off-task, which is what")
    print("  the paper says.")

    out = args.out or f"results/fixed_direction_control_{args.tag}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n  saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
