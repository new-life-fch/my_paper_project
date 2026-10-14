#!/usr/bin/env python3
"""
Probe-direction injection: does the model's verdict follow the direction a
trained probe uses to decode relevance from L*?

Why this is the clean version of the patching test
--------------------------------------------------
`patch_internal_v2.py` replaces a sample's L* activation with ANOTHER sample's
activation. That is informative but confounded: the transplanted vector is an
activation the network never produced for this input, so "the verdict doesn't
follow" could mean the model ignores out-of-distribution content rather than
ignoring relevance.

Here the perturbation is instead constrained to the direction the probe itself
uses, and its magnitude is set from the probe's own scale:

    h_L*  <-  h_L*  +  m * (s_probe_pivot) * w_hat

where w_hat is the probe's unit weight direction, `pivot` is the activation
where the probe's decision flips (the point with logit 0), and `m` is a sweep in
units of "probe logits". At m = +2 the probe should read the example as much
more relevant; at m = -2 as much less. Both the direction and the magnitude are
therefore meaningful to the probe, and the perturbation stays within the
subspace the probe reads.

The measurement is then simply: as m moves, does the MODEL's verdict move the
same way the PROBE's prediction does? We report, at each m,
  * auc_vs_probe  : agreement between model verdict and the probe's own score
  * follow_gap    : mean P(yes) on rows the probe calls relevant minus irrelevant
and compare against random directions of the same magnitude (same null logic as
the causal experiment).

Pure GPU (needs the model), one load. CPU-side probe fitting on cached
activations.

Usage:
  python scripts/patch_probe_direction.py --model meta-llama/Llama-3.2-3B \
      --cache results/cache/llama32_3b_q1500_instruct --tag llama32_3b \
      --out results/patch_probe_dir_llama32_3b_q1500.json
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


def probe_axes(Xtr, ytr):
    """Unit class-mean-difference direction and the probe's decision scale.

    Returns (w_hat, pivot) where moving h by `t * w_hat` changes the probe's
    decision variable by t (in units of the standardised feature space).
    """
    scaler = StandardScaler().fit(Xtr)
    Z = scaler.transform(Xtr)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced",
                             random_state=42).fit(Z, ytr)
    w = clf.coef_[0]                       # in standardised space
    b = float(clf.intercept_[0])
    w_hat = w / (np.linalg.norm(w) + 1e-12)
    # in standardised space the decision variable is w.z + b; the activation
    # whose standardised decision variable is 0 lies on the probe's boundary
    pivot = -b / (np.linalg.norm(w) ** 2 + 1e-12)
    return scaler, w_hat, pivot, clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--judge-cache", default=None)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--n-queries", type=int, default=1500)
    ap.add_argument("--max-passages", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ms", default="0,0.5,1,2,4",
                    help="perturbation magnitudes in probe-logit units")
    ap.add_argument("--n-random", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--out", default="results/patch_probe_dir.json")
    args = ap.parse_args()
    args.cache = resolve_cache(args.cache)
    ms = [float(x) for x in args.ms.split(",")]

    trd = torch.load(os.path.join(args.cache, "resid_train.pt"))
    Rtr = trd["acts"].to(torch.float32).numpy()
    ytr = trd["labels"].numpy()
    vd = torch.load(os.path.join(args.cache, "resid_val.pt"))
    Rva = vd["acts"].to(torch.float32).numpy()
    yva = vd["labels"].numpy()
    td = torch.load(os.path.join(args.cache, "resid_test.pt"))
    Rte = td["acts"].to(torch.float32).numpy()
    yte = td["labels"].numpy()
    qte = td["query_ids"].numpy()
    n_layers = Rtr.shape[1]
    H = Rtr.shape[2]

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

    m_probe = make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=3000, class_weight="balanced", random_state=42)).fit(Rtr[:, L, :], ytr)
    probe_tr = m_probe.predict_proba(Rte[:, L, :])[:, 1]
    probe_auc = float(roc_auc_score(yte, probe_tr))
    scaler, w_hat_std, _, _ = probe_axes(Rtr[:, L, :], ytr)
    # convert the standardised-space direction into raw activation space
    scale = scaler.scale_ if hasattr(scaler, "scale_") else scaler[-1].scale_
    w_hat_raw = (w_hat_std / scale)
    w_hat_raw = w_hat_raw / (np.linalg.norm(w_hat_raw) + 1e-12)
    w_hat_raw = torch.tensor(w_hat_raw, dtype=torch.float32)
    print(f"[{args.tag}] L*={L} probe test AUC={probe_auc:.4f}")

    # ---- data / model -----------------------------------------------------
    from src.data import load_ms_marco, split_by_query
    samples = load_ms_marco(n_queries=args.n_queries,
                            max_passages_per_query=args.max_passages, seed=args.seed)
    _, _, test_s = split_by_query(samples, seed=args.seed)
    assert len(test_s) == len(yte), f"row mismatch {len(test_s)} vs {len(yte)}"
    labels = np.array([x["label"] for x in test_s])
    assert np.array_equal(labels, yte), "label mismatch"
    prompts = [build_prompt(x["query"], x["passage"]) for x in test_s]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto",
                                                 dtype=torch.float16)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    yes_ids, no_ids = yes_no_ids(tok)
    dev = next(model.model.parameters()).device
    layers = model.model.layers
    assert n_layers == len(layers), f"cache {n_layers} layers, model {len(layers)}"

    vec_raw = w_hat_raw.to(dev)
    rng = np.random.default_rng(args.seed)
    randoms = []
    for _ in range(args.n_random):
        d = torch.randn(H, device=dev, dtype=torch.float32)
        randoms.append(d / d.norm())

    state = {"vec": None, "m": 0.0}
    hook_handle = None

    def make_hook():
        def hook(_m, _inp, out):
            if state["vec"] is None or state["m"] == 0.0:
                return out
            h = out[0] if isinstance(out, tuple) else out
            h[:, -1, :] = h[:, -1, :] + state["m"] * state["vec"].to(h.dtype)
            return out
        return hook

    hook_handle = layers[L].register_forward_hook(make_hook())

    @torch.no_grad()
    def run(m, vec):
        state["m"], state["vec"] = m, vec
        scores, logitdiffs = [], []
        try:
            for i in range(0, len(prompts), args.batch_size):
                enc = tok(prompts[i:i + args.batch_size], return_tensors="pt",
                          padding=True).to(dev)
                lg = model(**enc).logits
                for r in range(len(enc["input_ids"])):
                    v = lg[r, -1, :].float()
                    yes = torch.logsumexp(v[yes_ids], 0)
                    no = torch.logsumexp(v[no_ids], 0)
                    scores.append(torch.softmax(torch.stack([yes, no]), 0)[0].item())
                    logitdiffs.append((yes - no).item())
                del enc, lg
        finally:
            state["m"], state["vec"] = 0.0, None
            torch.cuda.empty_cache()
        return np.array(scores), np.array(logitdiffs)

    t0 = time.time()
    nat, nat_ld = run(0.0, None)
    judge_auc = float(roc_auc_score(yte, nat))
    print(f"  natural judge AUC={judge_auc:.4f} (baseline)")
    report = {"tag": args.tag, "model": args.model, "layer": int(L),
              "probe_test_auc": round(probe_auc, 4), "judge_auc": round(judge_auc, 4),
              "n_test": int(len(yte)), "n_random": args.n_random,
              "ms": ms, "natural_logitdiff_mean": round(float(nat_ld.mean()), 3),
              "sweep": {}, "note": (
                  "Perturbation is m * w_hat (probe's unit weight direction) at L*, "
                  "with m in probe-logit units; the probe reads the row as more "
                  "relevant for m>0. Compare auc_vs_probe (model agreement with the "
                  "probe) against the m=0 baseline.")}
    for m in ms:
        if m == 0.0:
            s, ld = nat, nat_ld
        else:
            s, ld = run(m, vec_raw)
        agree = float(roc_auc_score(probe_tr, s))
        report["sweep"][f"probe@{m}"] = {
            "auc_vs_true_label": round(float(roc_auc_score(yte, s)), 4),
            "auc_vs_probe_score": round(agree, 4),
            "follow_gap_vs_probe": round(float(
                s[probe_tr >= 0.5].mean() - s[probe_tr < 0.5].mean()), 4)
            if (probe_tr >= 0.5).any() and (probe_tr < 0.5).any() else None,
            "mean_pyes": round(float(s.mean()), 4),
            "mean_logitdiff": round(float(ld.mean()), 3),
        }
        print(f"    m={m:+.1f} auc_true={report['sweep'][f'probe@{m}']['auc_vs_true_label']:.4f} "
              f"agree_with_probe={agree:.4f} "
              f"gap={report['sweep'][f'probe@{m}']['follow_gap_vs_probe']}")

    # random directions of largest magnitude, for the null
    mmax = max(ms)
    rnd = []
    for d in randoms:
        s, _ = run(mmax, d)
        rnd.append(float(roc_auc_score(probe_tr, s)))
    report["random_directions"] = {
        "m": mmax,
        "auc_vs_probe_scores": [round(x, 4) for x in rnd],
        "mean": round(float(np.mean(rnd)), 4),
        "min": round(float(np.min(rnd)), 4),
        "max": round(float(np.max(rnd)), 4),
    }
    print(f"  random dirs at m={mmax}: agreement with probe "
          f"mean={np.mean(rnd):.4f} range[{np.min(rnd):.4f},{np.max(rnd):.4f}]")
    report["elapsed_sec"] = round(time.time() - t0, 1)
    hook_handle.remove()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
