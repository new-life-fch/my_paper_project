#!/usr/bin/env python3
"""
Does the model's own readout direction (logit lens) beat chance on MS MARCO,
and does a direction fitted on ANOTHER dataset beat it there too?

Why: the paper's mechanism claim rests on MS MARCO, but the "the model's own
readout is below chance / a foreign direction beats it" evidence was only
measured on FiQA. A mechanism statement has to hold on the dataset the claim is
made on, so this script repeats the same two measurements on MS MARCO:

  1. the logit-lens P(yes) curve on MS MARCO, with a query-clustered bootstrap
     CI at the layer where the trained probe is strongest (is it below 0.5?)
  2. the SAME fixed direction as the FiQA control -- fitted on FiQA, carried to
     MS MARCO with no adaptation -- evaluated at that layer. If a FiQA-fitted
     direction beats the model's own MS MARCO readout, the asymmetry is not a
     quirk of one dataset.

Pure CPU: one model load (for final_norm + lm_head), everything else is cached
residuals.

Usage:
  python scripts/readout_direction_msmarco.py
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cache_paths import resolve_cache  # noqa: E402

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def first_ids(tok, words):
    s = set()
    for w in words:
        e = tok.encode(w, add_special_tokens=False)
        if e:
            s.add(e[0])
    return sorted(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--ms-tag", default="llama32_3b")
    ap.add_argument("--fiqa-tag", default="llama32_3b_fiqa_bm25")
    ap.add_argument("--n-queries", type=int, default=1500)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ms_cache = f"results/cache/{args.ms_tag}_q{args.n_queries}_instruct"
    fq_cache = f"results/cache/{args.fiqa_tag}_q{args.n_queries}_instruct"
    sg = json.load(open(f"results/significance_{args.ms_tag}_q{args.n_queries}.json"))
    sg = sg[0] if isinstance(sg, list) else sg
    L = sg["best_layer_val"]

    def resid(cache, split):
        d = torch.load(os.path.join(cache, f"resid_{split}.pt"), map_location="cpu")
        return (d["acts"].to(torch.float32).numpy(), d["labels"].numpy(),
                d["query_ids"].numpy() if "query_ids" in d else None)

    Xtr_ms, ytr_ms, _ = resid(ms_cache, "train")
    Xte_ms, yte_ms, qte_ms = resid(ms_cache, "test")
    Xtr_fq, ytr_fq, _ = resid(fq_cache, "train")
    Xte_fq, yte_fq, _ = resid(fq_cache, "test")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {args.model} (CPU) ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()
    yi = first_ids(tok, ["yes", "Yes", "YES", " yes", " YES"])
    ni = first_ids(tok, ["no", "No", "NO", " no", " NO"])

    def logit_lens_R(R):
        R = R[:, L, :]
        out = np.zeros(len(R), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(R), args.batch):
                lg = model.lm_head(model.model.norm(R[i:i + args.batch]))
                s = torch.softmax(torch.stack([
                    torch.logsumexp(lg[:, yi], -1),
                    torch.logsumexp(lg[:, ni], -1)], -1), -1)[:, 0]
                out[i:i + args.batch] = s.numpy()
        return out

    print(f"  layer L{L} (val-selected from significance)")
    s_ms = logit_lens_R(torch.from_numpy(Xte_ms))
    auc_ms_ll = float(roc_auc_score(yte_ms, s_ms))

    # FiQA-fitted direction, carried to MS MARCO with no adaptation
    m_fq = make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=3000, class_weight="balanced", random_state=42))
    m_fq.fit(Xtr_fq[:, L, :], ytr_fq)
    s_xfer = m_fq.predict_proba(Xte_ms[:, L, :])[:, 1]
    auc_xfer = float(roc_auc_score(yte_ms, s_xfer))

    # in-domain MS MARCO probe, for scale
    m_ms = make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=3000, class_weight="balanced", random_state=42))
    m_ms.fit(Xtr_ms[:, L, :], ytr_ms)
    auc_ms_probe = float(roc_auc_score(yte_ms, m_ms.predict_proba(Xte_ms[:, L, :])[:, 1]))

    # query-clustered bootstrap for the logit-lens AUC (is it below chance?)
    rng = np.random.default_rng(42)
    q = qte_ms if qte_ms is not None else np.arange(len(yte_ms))
    uq = np.unique(q)
    by_q = {qq: np.where(q == qq)[0] for qq in uq}
    boot = []
    for _ in range(args.n_boot):
        pick = rng.choice(uq, size=len(uq), replace=True)
        ii = np.concatenate([by_q[qq] for qq in pick])
        boot.append(roc_auc_score(yte_ms[ii], s_ms[ii])
                    if len(np.unique(yte_ms[ii])) > 1 else 0.5)
    boot = np.array(boot)
    ci = [round(float(np.percentile(boot, 2.5)), 4),
          round(float(np.percentile(boot, 97.5)), 4)]
    p_ge = round(float((boot >= 0.5).mean()), 4)

    res = {
        "dataset": "MS MARCO", "model": args.model, "layer": int(L),
        "n_test": int(len(yte_ms)),
        "logit_lens_auc": round(auc_ms_ll, 4),
        "logit_lens_ci95": ci, "logit_lens_p_ge_half": p_ge,
        "fiqa_fitted_direction_on_msmarco": round(auc_xfer, 4),
        "msmarco_fitted_direction_on_msmarco": round(auc_ms_probe, 4),
        "n_boot": args.n_boot,
    }
    print()
    print(f"  model's own readout (logit lens) at L{L}   : {auc_ms_ll:.4f} "
          f"CI {ci}  P(AUC>=0.5)={p_ge}")
    print(f"  FiQA-fitted direction, carried to MS MARCO : {auc_xfer:.4f}")
    print(f"  MS MARCO-fitted direction (in-domain)      : {auc_ms_probe:.4f}")
    print()
    if ci[1] < 0.5:
        print("  -> the model's own readout is BELOW CHANCE on MS MARCO too "
              "(CI entirely under 0.5)")
    elif auc_xfer > auc_ms_ll:
        print("  -> not below chance, but a foreign direction still beats it")
    else:
        print("  -> no asymmetry on MS MARCO; report that honestly")

    out = args.out or f"results/readout_direction_{args.ms_tag}.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"  saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
