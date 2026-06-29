#!/usr/bin/env python3
"""
Logit-lens baseline: an UNSUPERVISED, zero-training output-side reader at EVERY
depth. Reviewer-requested control (P3).

The paper's "output representation" reader is a *trained* probe on the final
hidden state. A reviewer can object: maybe relevance is decodable mid-network
only because a *trained* probe reads it, and an untrained read-out (what the
model could actually surface) is flat. The logit lens settles this: project
each layer's cached residual through the model's own final norm + unembedding
(final_norm @ lm_head), read normalized P("yes") vs P("no") at the answer
token, and compute ROC-AUC per layer. No probe, no training, no layer
selection on labels.

If the logit-lens AUC is high mid-network and decays toward the output, that is
direct, training-free evidence for the attenuation story. If it is flat near
chance everywhere, it shows the signal needs a trained read-out (which we then
state honestly). Either way it closes the alternative explanation.

Sanity check: the LAST-layer logit-lens AUC should ~match the LLM-judge AUC,
since both apply lm_head to the (essentially) final hidden state.

Usage:
  python scripts/logit_lens.py --model meta-llama/Llama-3.2-3B \
      --cache results/cache/llama32_3b_q1500_instruct \
      --judge-cache results/cache/llama32_3b_q1500_judge \
      --tag llama32_3b_q1500 --out results/logit_lens_llama32_3b_q1500.json
"""
import os
import sys
import argparse
import json

import torch
import numpy as np
from sklearn.metrics import roc_auc_score as auc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def yes_no_token_ids(tokenizer):
    """First-token ids for yes/no surface variants (identical to llm_judge.py)."""
    yes_words = [" Yes", " yes", "Yes", "yes", " YES"]
    no_words = [" No", " no", "No", "no", " NO"]

    def first_ids(words):
        ids = set()
        for w in words:
            enc = tokenizer.encode(w, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
        return sorted(ids)

    return first_ids(yes_words), first_ids(no_words)


def load_resid(cache, split):
    d = torch.load(os.path.join(cache, f"resid_{split}.pt"))
    return d["acts"].float(), d["labels"].numpy()  # [N, L, H], [N]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--cache", required=True, help="*_instruct cache dir with resid_*.pt")
    ap.add_argument("--judge-cache", default=None, help="optional, for sanity vs judge")
    ap.add_argument("--tag", default="model")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="results/logit_lens.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {args.model} (norm + lm_head only used) ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda")
    model.eval()

    yes_ids, no_ids = yes_no_token_ids(tok)
    print(f"yes ids: {yes_ids}\nno  ids: {no_ids}")

    # final norm + unembedding (LLaMA/Mistral: model.model.norm, model.lm_head)
    final_norm = model.model.norm
    lm_head = model.lm_head

    R, y = load_resid(args.cache, args.split)  # [N, L, H]
    N, L, H = R.shape
    print(f"resid {args.split}: N={N} L={L} H={H}  pos={int(y.sum())}/{N}")

    dev = next(model.parameters()).device
    yes_t = torch.tensor(yes_ids, device=dev)
    no_t = torch.tensor(no_ids, device=dev)

    layer_pyes = []   # [L][N]
    layer_auc = []
    with torch.no_grad():
        for li in range(L):
            h = R[:, li, :].to(dev).half()             # [N, H]
            hn = final_norm(h)                          # RMSNorm
            logits = lm_head(hn).float()                # [N, vocab]
            yes = torch.logsumexp(logits[:, yes_t], dim=1)
            no = torch.logsumexp(logits[:, no_t], dim=1)
            p_yes = torch.softmax(torch.stack([yes, no], dim=1), dim=1)[:, 0]
            p = p_yes.cpu().numpy()
            layer_pyes.append(p)
            try:
                a = float(auc(y, p))
            except ValueError:
                a = 0.5
            layer_auc.append(round(a, 3))

    best = int(np.argmax(layer_auc))
    rep = {
        "tag": args.tag,
        "model": args.model,
        "split": args.split,
        "n": int(N),
        "n_layers": int(L),
        "logit_lens_per_layer_auc": layer_auc,
        "logit_lens_final_layer_auc": layer_auc[-1],
        "logit_lens_best_layer": {"layer": best, "auc": layer_auc[best]},
    }

    if args.judge_cache:
        jd = torch.load(os.path.join(args.judge_cache, f"judge_{args.split}.pt"))
        ja = float(auc(jd["labels"].numpy(), jd["scores"].numpy()))
        rep["llm_judge_auc"] = round(ja, 3)
        rep["sanity_lastlayer_vs_judge_absdiff"] = round(abs(layer_auc[-1] - ja), 3)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))
    print(f"\nSaved {args.out}")
    print(f"peak logit-lens AUC {layer_auc[best]} @L{best} vs final-layer "
          f"{layer_auc[-1]}" + (f" vs judge {rep.get('llm_judge_auc')}" if args.judge_cache else ""))


if __name__ == "__main__":
    main()
