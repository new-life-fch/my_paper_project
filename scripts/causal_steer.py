#!/usr/bin/env python3
"""
CAUSAL test for "internals know more than the model can say".

Probing (AUC) only shows the relevance signal is DECODABLE internally — it is
correlational. Reviewers will ask: is that internal signal CAUSALLY connected to
the output verdict, or an epiphenomenal trace? This script answers with a
steering / activation-injection intervention.

Idea
----
1. From a residual cache, take the relevance DIRECTION at the best internal layer
   as the difference of class means d = mean(resid | relevant) - mean(resid | irrelevant),
   unit-normalized (a standard, probe-free steering vector).
2. During a forward pass on each test prompt, ADD alpha * |h| * d to the residual
   stream at the OUTPUT (final) layer, where |h| scales by the per-token residual
   norm so alpha is dimensionless. Then read the judge score P(yes) at the answer
   token (same yes/no token sets as llm_judge.py).
3. Sweep alpha in [-A..A]. If the INTERNAL direction, injected AT THE OUTPUT, makes
   mean P(yes) move monotonically, then the output pathway CAN express this signal —
   it is merely under-expressed in the natural forward pass (attenuation, not
   absence). That is the causal core of the thesis.

Controls
--------
  * random unit direction (matched norm)  -> expect flat / no systematic shift
  * final-layer's OWN diff-of-means direction -> the output representation's native
    direction; comparing slopes tells us how much the internal direction "carries"
    at the output relative to the output's own.

Outputs results/causal_steer_<tag>.json (+ optional png).

Usage (run AFTER scaleup frees the GPU to avoid OOM):
  python scripts/causal_steer.py --model meta-llama/Llama-3.2-3B \
      --cache results/cache/q500_instruct --tag llama32_3b
"""
import os, sys, argparse, json
import numpy as np, torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import load_ms_marco, split_by_query


def build_prompt(query, passage):
    return (f"Document: {passage}\nQuestion: {query}\n"
            f"Is the document relevant to the question? Answer:")


def yes_no_ids(tok):
    def first(words):
        s = set()
        for w in words:
            e = tok.encode(w, add_special_tokens=False)
            if e: s.add(e[0])
        return sorted(s)
    return first([" Yes"," yes","Yes","yes"]), first([" No"," no","No","no"])


def diff_means_dir(acts, labels):
    """Unit diff-of-means direction at one layer. acts [N,H], labels [N]."""
    pos = acts[labels == 1].mean(0)
    neg = acts[labels == 0].mean(0)
    d = pos - neg
    return d / (np.linalg.norm(d) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--cache", default="results/cache/q500_instruct")
    ap.add_argument("--tag", default="llama32_3b")
    ap.add_argument("--n-queries", type=int, default=500)
    ap.add_argument("--max-passages", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alphas", default="-8,-4,-2,-1,0,1,2,4,8")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"results/causal_steer_{args.tag}.json"
    alphas = [float(a) for a in args.alphas.split(",")]

    # --- directions from cache (CPU, no GPU) -------------------------------
    Rtr = torch.load(os.path.join(args.cache, "resid_train.pt"))
    Atr = Rtr["acts"].to(torch.float32).numpy(); ytr = Rtr["labels"].numpy()
    n_layers = Atr.shape[1]
    # best internal layer = max train-set diff-of-means AUC via 1-d projection
    def proj_auc(li):
        d = diff_means_dir(Atr[:, li, :], ytr)
        s = Atr[:, li, :] @ d
        try: return roc_auc_score(ytr, s)
        except ValueError: return 0.5
    laucs = [proj_auc(li) for li in range(n_layers)]
    best = int(np.argmax(laucs[:-1]))  # exclude final layer itself
    d_internal = diff_means_dir(Atr[:, best, :], ytr).astype(np.float32)
    d_final    = diff_means_dir(Atr[:, -1, :], ytr).astype(np.float32)
    rng = np.random.default_rng(0)
    d_random   = rng.standard_normal(d_internal.shape).astype(np.float32)
    d_random  /= np.linalg.norm(d_random)
    print(f"[{args.tag}] best internal layer L{best} (proj-AUC {laucs[best]:.3f}); "
          f"final-layer proj-AUC {laucs[-1]:.3f}")

    # --- model + test prompts ---------------------------------------------
    samples = load_ms_marco(n_queries=args.n_queries, max_passages_per_query=args.max_passages, seed=args.seed)
    _, _, test_s = split_by_query(samples, seed=args.seed)
    labels = np.array([x["label"] for x in test_s])

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {args.model} (native transformers + forward hook) ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", dtype=torch.float16)
    model.eval()
    yes_ids, no_ids = yes_no_ids(tok)
    dev = next(model.model.parameters()).device
    dirs = {"internal": torch.tensor(d_internal, device=dev, dtype=torch.float16),
            "final":    torch.tensor(d_final, device=dev, dtype=torch.float16),
            "random":   torch.tensor(d_random, device=dev, dtype=torch.float16)}
    last = model.model.layers[-1]

    # steering injected via a forward hook on the final decoder layer: add
    # alpha*|h_last| * dvec to the last-token residual. State lives in a mutable
    # dict so we can toggle it per (dir, alpha) without re-registering.
    steer = {"vec": None, "alpha": 0.0}

    def hook(_m, _inp, out):
        if steer["alpha"] == 0.0 or steer["vec"] is None:
            return out
        h = out[0] if isinstance(out, tuple) else out      # [B,S,H]
        norm = h[:, -1, :].norm(dim=-1, keepdim=True)        # [B,1]
        h[:, -1, :] = h[:, -1, :] + (steer["alpha"] * norm) * steer["vec"]
        return out
    last.register_forward_hook(hook)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # last real token sits at position -1 for every row

    @torch.no_grad()
    def judge_scores(dvec, alpha):
        steer["vec"], steer["alpha"] = dvec, alpha
        scores = []
        bs = 16
        for i in range(0, len(test_s), bs):
            batch = test_s[i:i + bs]
            prompts = [build_prompt(x["query"], x["passage"]) for x in batch]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=512).to(dev)
            logits = model(**enc).logits                     # [B,S,V]; left-padded -> -1 is last real tok
            for r in range(len(batch)):
                lg = logits[r, -1, :].float()
                yes = torch.logsumexp(lg[yes_ids], 0); no = torch.logsumexp(lg[no_ids], 0)
                scores.append(torch.softmax(torch.stack([yes, no]), 0)[0].item())
            del enc, logits
        return np.array(scores)

    from tqdm import tqdm
    report = {"tag": args.tag, "model": args.model, "best_internal_layer": best,
              "proj_auc_internal": round(laucs[best], 3), "proj_auc_final": round(laucs[-1], 3),
              "alphas": alphas, "n_test": len(test_s), "sweep": {}}
    for name, dvec in dirs.items():
        meanp, aucs = [], []
        for a in tqdm(alphas, desc=f"steer/{name}"):
            s = judge_scores(dvec, a)
            meanp.append(round(float(s.mean()), 4))
            try: aucs.append(round(float(roc_auc_score(labels, s)), 3))
            except ValueError: aucs.append(0.5)
            torch.cuda.empty_cache()
        report["sweep"][name] = {"mean_pyes": meanp, "auc": aucs}
        print(f"  {name:9}: mean P(yes) vs alpha = {meanp}")

    json.dump(report, open(out, "w"), indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
