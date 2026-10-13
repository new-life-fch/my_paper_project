#!/usr/bin/env python3
"""
Logit-lens control: is the mid-network relevance signal just "the output readout
applied to an earlier layer"?

For every layer's cached residual stream we apply the model's OWN final norm +
lm_head and read P(yes) vs P(no).  Zero training.  If the mid-layer signal were
merely attenuated on its way to the logits, the model's own readout should
recover at least part of it at *some* depth; instead every layer stays near
chance, far below the trained probes.

Usage:
  python scripts/logit_lens.py --model meta-llama/Llama-3.2-3B \
      --cache results/cache/llama32_3b_q1500_instruct \
      --judge-cache results/cache/llama32_3b_q1500_judge \
      --out results/logit_lens_llama32_3b_q1500.json

NOTE on layer selection (2026-10-13, TODO.md P0-6):
  The reported peak used to be `argmax` over the EVALUATED split (default test)
  -- selection-on-test -- while the paper claimed "no label-based layer
  selection".  Now the peak is chosen on **val** and reported on **test**, and
  the per-layer curves for both splits are dumped so any claim can be audited.
  The lens itself remains zero-training; only the choice of *which layer to
  quote* uses labels, and that choice is now made out-of-sample.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cache_paths import resolve_cache, warn_if_legacy_exists  # noqa: E402

import argparse, json
import numpy as np, torch
from sklearn.metrics import roc_auc_score

YES_WORDS = ["yes", "Yes", "YES", " yes", " YES"]
NO_WORDS = ["no", "No", "NO", " no", " NO"]


def auc(y, s):
    try: return float(roc_auc_score(y, s))
    except ValueError: return 0.5


def first_ids(tok, words):
    ids = []
    for w in words:
        t = tok.encode(w, add_special_tokens=False)
        if t: ids.append(t[0])
    return sorted(set(ids))


def yes_no_token_ids(tok):
    return first_ids(tok, YES_WORDS), first_ids(tok, NO_WORDS)


def load_resid(cache, split):
    d = torch.load(os.path.join(cache, f"resid_{split}.pt"))
    qids = d["query_ids"].numpy() if "query_ids" in d else None
    return d["acts"].float(), d["labels"].numpy(), qids  # [N, L, H], [N]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--cache", required=True, help="*_instruct cache dir with resid_*.pt")
    ap.add_argument("--judge-cache", default=None, help="optional, for sanity vs judge")
    ap.add_argument("--tag", default="model")
    ap.add_argument("--split", default="test",
                    help="split the PEAK is quoted on; val always drives the choice")
    ap.add_argument("--out", default="results/logit_lens.json")
    args = ap.parse_args()
    args.cache = resolve_cache(args.cache)   # docs/CACHE_NAMING.md
    if args.judge_cache:
        args.judge_cache = resolve_cache(args.judge_cache)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {args.model} (norm + lm_head only used) ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda")
    model.eval()

    yes_ids, no_ids = yes_no_token_ids(tok)
    print(f"yes ids: {yes_ids}\nno  ids: {no_ids}")

    final_norm = model.model.norm
    lm_head = model.lm_head
    dev = next(model.parameters()).device
    yes_t = torch.tensor(yes_ids, device=dev)
    no_t = torch.tensor(no_ids, device=dev)

    def lens_auc(split):
        """Per-layer logit-lens AUC on `split`, plus the raw P(yes) matrix."""
        R, y, qids = load_resid(args.cache, split)
        N, L, H = R.shape
        print(f"resid {split}: N={N} L={L} H={H}  pos={int(y.sum())}/{N}")
        pyes, aucs = [], []
        with torch.no_grad():
            for li in range(L):
                hn = final_norm(R[:, li, :].to(dev).half())
                logits = lm_head(hn).float()
                yes = torch.logsumexp(logits[:, yes_t], dim=1)
                no = torch.logsumexp(logits[:, no_t], dim=1)
                p = torch.softmax(torch.stack([yes, no], dim=1), dim=1)[:, 0].cpu().numpy()
                pyes.append(p)
                aucs.append(round(auc(y, p), 3))
        return np.array(pyes), aucs, y, qids, L

    val_pyes, val_aucs, val_y, val_qids, L = lens_auc("val")
    # selection is made OUT OF SAMPLE (val); the quoted curve is --split (test)
    best_val = int(np.argmax(val_aucs))
    test_pyes, test_aucs, test_y, test_qids, _ = lens_auc(args.split)

    rep = {
        "tag": args.tag,
        "model": args.model,
        "split": args.split,
        "n": int(len(test_y)),
        "n_layers": int(L),
        "selection": "val",
        "logit_lens_per_layer_auc_val": val_aucs,
        "logit_lens_per_layer_auc": test_aucs,          # curve on --split (test)
        "logit_lens_final_layer_auc": test_aucs[-1],
        "logit_lens_best_layer": {
            "layer": best_val,                          # chosen on val
            "auc": test_aucs[best_val],                 # reported on test
            "val_auc_at_selected": val_aucs[best_val],
        },
        # transparency only -- selection-on-test, never report as a result
        "logit_lens_best_layer_test_argmax_DO_NOT_REPORT": {
            "layer": int(np.argmax(test_aucs)), "auc": max(test_aucs)},
        "env": {"numpy": np.__version__, "torch": torch.__version__},
    }

    if args.judge_cache:
        jd = torch.load(os.path.join(args.judge_cache, f"judge_{args.split}.pt"))
        assert np.array_equal(jd["labels"].numpy(), test_y), "judge/resid label mismatch"
        if "query_ids" in jd and test_qids is not None:
            assert np.array_equal(jd["query_ids"].numpy(), test_qids), \
                "judge/resid QUERY_ID mismatch -- caches describe different rows"
        ja = float(auc(jd["labels"].numpy(), jd["scores"].numpy()))
        rep["llm_judge_auc"] = round(ja, 3)
        rep["sanity_lastlayer_vs_judge_absdiff"] = round(abs(test_aucs[-1] - ja), 3)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps({k: v for k, v in rep.items()
                      if not k.startswith("logit_lens_per_layer")}, indent=2))
    b = rep["logit_lens_best_layer"]
    print(f"\nval-selected peak: test AUC {b['auc']} @L{b['layer']} "
          f"(val {b['val_auc_at_selected']}) vs final layer {test_aucs[-1]}"
          + (f" vs judge {rep.get('llm_judge_auc')}" if args.judge_cache else ""))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
