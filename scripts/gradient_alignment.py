#!/usr/bin/env python3
"""
Gradient alignment: is the model's own verdict SENSITIVE to the direction a
probe uses to read relevance at L*?

The problem with every activation-editing test
----------------------------------------------
Cross-example patching, probe-direction injection, and random-direction
injection all move the activation somewhere the network never goes for this
input. So a null result ("the verdict does not follow") is always open to the
objection that the model simply ignores out-of-distribution activations, not
relevance.

A gradient has no such problem. The directional derivative of the model's own
verdict with respect to the L* activation is a LOCAL property of the network at
the activation it actually produced. We therefore measure

    cos( g , w_hat )        where  g = d(yes_no_logit_difference)/d h_L*
                                   w_hat = the probe's unit relevance direction

and compare it against the same cosine for random directions. This asks
directly: is the model's output readout pointed at the relevance direction, or
away from it? It needs one backward pass per example, so it is cheap and it
cannot be confounded by the size of an edit.

Interpretation to fix in advance
--------------------------------
  * cos ~ 0 for the probe direction, and no higher than random directions
      -> the model's verdict is locally INSENSITIVE to the very direction a
         probe uses: the signal is decodable but not on the path the output
         readout attends to. This is the misalignment claim, measured without
         editing anything.
  * cos clearly positive and above the random spread
      -> the verdict IS locally sensitive to the relevance direction, and any
         patching null must be explained some other way.
Both outcomes are reportable. Random directions give the null spread, so the
claim is a rank/percentile statement, not a bare number.

Usage:
  python scripts/gradient_alignment.py --model meta-llama/Llama-3.2-3B \
      --cache results/cache/llama32_3b_q1500_instruct --tag llama32_3b \
      --out results/gradient_alignment_llama32_3b_q1500.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cache_paths import resolve_cache  # noqa: E402

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_prompt(query, passage):
    """Byte-identical to extract_instruct.py / llm_judge.py (selfcheck check 6)."""
    return (f"Document: {passage}\nQuestion: {query}\n"
            f"Is the document relevant to the question? Answer:")


def yes_no_ids(tok):
    def first(words):
        s = set()
        for w in words:
            e = tok.encode(w, add_special_tokens=False)
            if e:
                s.add(e[0])
        return sorted(s)
    return (first(["yes", "Yes", "YES", " yes", " YES"]),
            first(["no", "No", "NO", " no", " NO"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--n-queries", type=int, default=1500)
    ap.add_argument("--max-passages", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-random", type=int, default=16)
    ap.add_argument("--n-rows", type=int, default=256,
                    help="test rows to differentiate (backward pass each); "
                         "0 = all")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--out", default="results/gradient_alignment.json")
    args = ap.parse_args()
    args.cache = resolve_cache(args.cache)

    trd = torch.load(os.path.join(args.cache, "resid_train.pt"))
    Rtr = trd["acts"].to(torch.float32).numpy()
    ytr = trd["labels"].numpy()
    vd = torch.load(os.path.join(args.cache, "resid_val.pt"))
    Rva = vd["acts"].to(torch.float32).numpy()
    yva = vd["labels"].numpy()
    td = torch.load(os.path.join(args.cache, "resid_test.pt"))
    yte = td["labels"].numpy()
    qte = td["query_ids"].numpy()
    n_layers, H = Rtr.shape[1], Rtr.shape[2]

    if args.layer is None:
        from joblib import Parallel, delayed

        def val_auc(li):
            m = make_pipeline(StandardScaler(), LogisticRegression(
                max_iter=3000, class_weight="balanced", random_state=42))
            m.fit(Rtr[:, li, :], ytr)
            return roc_auc_score(yva, m.predict_proba(Rva[:, li, :])[:, 1])
        n_jobs = max(1, min(32, (os.cpu_count() or 4) - 2))
        vaucs = Parallel(n_jobs=n_jobs)(delayed(val_auc)(li) for li in range(n_layers))
        L = int(np.argmax(vaucs))
    else:
        L = args.layer

    # probe direction in RAW activation space
    scaler = StandardScaler().fit(Rtr[:, L, :])
    clf = LogisticRegression(max_iter=3000, class_weight="balanced",
                             random_state=42).fit(scaler.transform(Rtr[:, L, :]), ytr)
    w_std = clf.coef_[0]
    w_raw = w_std / scaler.scale_
    w_hat = w_raw / np.linalg.norm(w_raw)
    print(f"[{args.tag}] L*={L}  ||w_raw||={np.linalg.norm(w_raw):.4f}")

    # ---- model ------------------------------------------------------------
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto",
                                                 dtype=torch.float32)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    yes_ids, no_ids = yes_no_ids(tok)
    dev = next(model.model.parameters()).device
    assert n_layers == len(model.model.layers)

    from src.data import load_ms_marco, split_by_query
    samples = load_ms_marco(n_queries=args.n_queries,
                            max_passages_per_query=args.max_passages, seed=args.seed)
    _, _, test_s = split_by_query(samples, seed=args.seed)
    assert len(test_s) == len(yte), f"row mismatch {len(test_s)} vs {len(yte)}"
    labels = np.array([x["label"] for x in test_s])
    assert np.array_equal(labels, yte), "label mismatch"

    idx = np.arange(len(test_s))
    if args.n_rows and args.n_rows < len(idx):
        rng0 = np.random.default_rng(args.seed)
        # keep the class balance so the cosine average is not label-skewed
        pos = idx[labels == 1]
        neg = idx[labels == 0]
        half = args.n_rows // 2
        idx = np.concatenate([rng0.choice(pos, min(half, len(pos)), replace=False),
                              rng0.choice(neg, min(half, len(neg)), replace=False)])
        idx = np.sort(idx)
    print(f"  differentiating {len(idx)} rows (layer {L})")

    captured = {}

    def cap_hook(_m, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h
        return out

    handle = model.model.layers[L].register_forward_hook(cap_hook)
    grads = []
    t0 = time.time()
    try:
        for n, i in enumerate(idx):
            x = test_s[i]
            enc = tok([build_prompt(x["query"], x["passage"])],
                      return_tensors="pt").to(dev)
            captured.clear()
            model.zero_grad(set_to_none=True)
            out = model(**enc)
            lg = out.logits[0, -1, :]
            yes = torch.logsumexp(lg[yes_ids], 0)
            no = torch.logsumexp(lg[no_ids], 0)
            (yes - no).backward()
            h = captured.get("h")
            if h is None or h.grad is None:
                continue
            grads.append(h.grad[0, -1, :].detach().float().cpu().numpy())
            del out, lg, enc
            if (n + 1) % 64 == 0:
                print(f"    {n+1}/{len(idx)} ({time.time()-t0:.0f}s)")
    finally:
        handle.remove()
    G = np.array(grads)
    print(f"  gradient matrix {G.shape} in {time.time()-t0:.0f}s")

    rng = np.random.default_rng(args.seed)
    def cos_all(v):
        v = v / (np.linalg.norm(v) + 1e-12)
        num = G @ v
        den = np.linalg.norm(G, axis=1) + 1e-12
        return num / den

    probe_cos = cos_all(w_hat)
    rand_cos = [cos_all(rng.normal(size=H)) for _ in range(args.n_random)]
    rand_mat = np.stack(rand_cos)              # [n_random, n_rows]

    def summ(a):
        return {"mean": round(float(np.mean(a)), 5),
                "median": round(float(np.median(a)), 5),
                "frac_positive": round(float((a > 0).mean()), 4)}

    report = {
        "tag": args.tag, "model": args.model, "layer": int(L),
        "n_rows": int(G.shape[0]), "n_random": args.n_random,
        "probe_direction_cosine": summ(probe_cos),
        "random_directions_cosine": {
            "mean_of_means": round(float(np.mean([np.mean(r) for r in rand_mat])), 5),
            "mean_abs_of_means": round(float(np.mean([abs(np.mean(r)) for r in rand_mat])), 5),
            "max_abs_mean": round(float(np.max([abs(np.mean(r)) for r in rand_mat])), 5),
            "per_direction_means": [round(float(np.mean(r)), 5) for r in rand_mat],
        },
        "probe_vs_random_abs_mean_percentile": round(float(
            (np.abs([np.mean(r) for r in rand_mat]) < abs(np.mean(probe_cos))).mean()), 4),
        "protocol": (
            "g = grad of (yes-no logit difference) wrt the L* answer-token residual; "
            "cosine with the probe's unit relevance direction vs random unit directions. "
            "No activation is edited, so no out-of-distribution concern."),
    }
    print()
    print(f"  cosine(model gradient, PROBE direction) : mean {report['probe_direction_cosine']['mean']:+.5f} "
          f"median {report['probe_direction_cosine']['median']:+.5f} "
          f"frac>0 {report['probe_direction_cosine']['frac_positive']}")
    print(f"  cosine with RANDOM directions           : |mean| up to "
          f"{report['random_directions_cosine']['max_abs_mean']:.5f}, "
          f"mean of |means| {report['random_directions_cosine']['mean_abs_of_means']:.5f}")
    print(f"  probe |mean cos| percentile vs random   : "
          f"{report['probe_vs_random_abs_mean_percentile']:.2f}")
    print()
    print("  Reading (fixed in advance): cos near zero and no higher than the random")
    print("  spread -> the verdict is locally insensitive to the relevance direction,")
    print("  i.e. misalignment, with no activation edited. A clearly positive cos above")
    print("  the random spread would instead mean the readout IS pointed at it.")

    report["elapsed_sec"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
