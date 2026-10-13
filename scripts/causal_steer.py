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
  * N random unit directions (matched norm)  -> a DISTRIBUTION, not one draw.
  * final-layer's OWN diff-of-means direction -> the output representation's native
    direction; comparing slopes tells us how much the internal direction "carries"
    at the output relative to the output's own.

Fixes applied 2026-10-13 (TODO.md P0-7 / P1-19 / R-5) — read before trusting numbers
----------------------------------------------------------------------------------
  a) **Per-label P(yes).** The old artifact recorded only the overall mean P(yes),
     so "relevance-ALIGNED movement" (the thing the causal claim is about) was
     literally unmeasurable: a random direction that adds a constant to everyone's
     logit produced the same mean shift as a real one. Now every (direction, alpha)
     records mean P(yes) over TRUE-POSITIVE and TRUE-NEGATIVE rows separately, and
     the headline diagnostic is the alignment gap
     `gap(alpha) = mean_pyes[rel](alpha) - mean_pyes[irrel](alpha)`.
     Steering that carries relevance should WIDEN the gap as alpha grows; a blunt
     shift moves both groups together and leaves the gap flat.
  b) **>=10 random directions** (`--n-random`), giving a null distribution for the
     gap so the internal direction can be called an outlier or not.
  c) **Direction layer chosen on VAL**, not on train (it used to argmax TRAIN
     projection AUC, which is why the paper's "best internal layer" disagreed with
     its own val-selected layer L9 vs the script's L13).
  d) **No length truncation.** It used `truncation=True, max_length=512`, which
     silently cuts the "Question … Answer:" suffix off long MS MARCO prompts, so
     the judged position was mid-document rather than the answer token. The
     extractor never truncated; padding to the batch max is exact and now the only
     thing that happens.
  e) **Same yes/no token set as llm_judge.py / logit_lens.py** (5 variants incl.
     " Yes"/" No"); the old set had 4. ~0.001 AUC effect, but it made the docstring
     a lie.

Outputs results/causal_steer_<tag>.json (+ optional png).

Usage:
  python scripts/causal_steer.py --model meta-llama/Llama-3.2-3B \
      --cache results/cache/llama32_3b_q1500_instruct --tag llama32_3b \
      --n-queries 1500 --out results/causal_steer_llama32_3b_q1500.json
"""
import os, sys, argparse, json
import numpy as np, torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cache_paths import resolve_cache, warn_if_legacy_exists   # docs/CACHE_NAMING.md
from src.data import load_ms_marco, split_by_query


def build_prompt(query, passage):
    return (f"Document: {passage}\nQuestion: {query}\n"
            f"Is the document relevant to the question? Answer:")


def yes_no_ids(tok):
    """MUST stay identical to llm_judge.py / logit_lens.py (5 variants each)."""
    def first(words):
        s = set()
        for w in words:
            e = tok.encode(w, add_special_tokens=False)
            if e: s.add(e[0])
        return sorted(s)
    return (first(["yes", "Yes", "YES", " yes", " YES"]),
            first(["no", "No", "NO", " no", " NO"]))


def diff_means_dir(acts, labels):
    """Unit diff-of-means direction at one layer. acts [N,H], labels [N]."""
    pos = acts[labels == 1].mean(0)
    neg = acts[labels == 0].mean(0)
    d = pos - neg
    return d / (np.linalg.norm(d) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--cache", default="results/cache/llama32_3b_q1500_instruct")
    ap.add_argument("--tag", default="llama32_3b")
    ap.add_argument("--n-queries", type=int, default=1500)
    ap.add_argument("--max-passages", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alphas", default="-8,-4,-2,-1,0,1,2,4,8")
    ap.add_argument("--n-random", type=int, default=12,
                    help="random control directions (null distribution for the gap)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.cache = resolve_cache(args.cache)   # docs/CACHE_NAMING.md
    out = args.out or f"results/causal_steer_{args.tag}.json"
    alphas = [float(a) for a in args.alphas.split(",")]

    # --- directions from cache (CPU) ---------------------------------------
    # diff-of-means is fit on TRAIN (it is a probe; fitting on train is correct),
    # but the LAYER is chosen on VAL, matching significance.py (TODO.md P1-19c).
    tr = torch.load(os.path.join(args.cache, "resid_train.pt"))
    va = torch.load(os.path.join(args.cache, "resid_val.pt"))
    Atr = tr["acts"].to(torch.float32).numpy(); ytr = tr["labels"].numpy()
    Ava = va["acts"].to(torch.float32).numpy(); yva = va["labels"].numpy()
    n_layers = Atr.shape[1]

    def proj_auc(X, y, li, d=None):
        d = diff_means_dir(Atr[:, li, :], ytr) if d is None else d
        s = X[:, li, :] @ d
        try: return roc_auc_score(y, s)
        except ValueError: return 0.5

    train_laucs = [proj_auc(Atr, ytr, li) for li in range(n_layers)]
    val_laucs = [proj_auc(Ava, yva, li) for li in range(n_layers)]
    best = int(np.argmax(val_laucs[:-1]))       # exclude the final layer itself
    d_internal = diff_means_dir(Atr[:, best, :], ytr).astype(np.float32)
    d_final = diff_means_dir(Atr[:, -1, :], ytr).astype(np.float32)
    proj_internal = {"train": round(train_laucs[best], 3), "val": round(val_laucs[best], 3)}
    proj_final = {"train": round(train_laucs[-1], 3), "val": round(val_laucs[-1], 3)}

    # E-8 / R-5: a direction's own relevance quality must be reported alongside its
    # steering effect. "The final direction destroys AUC while the internal one
    # keeps it" is partly explained by the two directions simply being different
    # quality relevance directions (internal proj-AUC > final proj-AUC). Without
    # this number the AUC contrast cannot be interpreted.
    def dir_proj_auc(d, X=Ava, y=yva):
        s = X[:, -1, :] @ d                      # random dirs live in the same space
        try: return float(roc_auc_score(y, s))
        except ValueError: return 0.5
    print(f"[{args.tag}] best internal layer L{best} (val proj-AUC {val_laucs[best]:.3f}, "
          f"train {train_laucs[best]:.3f}); final-layer val proj-AUC {val_laucs[-1]:.3f}")

    rng = np.random.default_rng(0)
    randoms = []
    for _ in range(args.n_random):
        d = rng.standard_normal(d_internal.shape).astype(np.float32)
        randoms.append(d / np.linalg.norm(d))
    rnd_proj = [round(dir_proj_auc(d), 4) for d in randoms]
    print(f"[{args.tag}] direction quality (val proj-AUC): internal {proj_internal['val']:.3f} | "
          f"final {proj_final['val']:.3f} | random mean {np.mean(rnd_proj):.3f} "
          f"(min {min(rnd_proj):.3f}, max {max(rnd_proj):.3f})")

    # --- model + test prompts ---------------------------------------------
    samples = load_ms_marco(n_queries=args.n_queries,
                            max_passages_per_query=args.max_passages, seed=args.seed)
    _, _, test_s = split_by_query(samples, seed=args.seed)
    labels = np.array([x["label"] for x in test_s])

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {args.model} (native transformers + forward hook) ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto",
                                                 dtype=torch.float16)
    model.eval()
    yes_ids, no_ids = yes_no_ids(tok)
    print(f"yes ids {yes_ids} / no ids {no_ids}")
    dev = next(model.model.parameters()).device
    last = model.model.layers[-1]

    steer = {"vec": None, "alpha": 0.0}

    def hook(_m, _inp, out):
        if steer["alpha"] == 0.0 or steer["vec"] is None:
            return out
        h = out[0] if isinstance(out, tuple) else out      # [B,S,H]
        norm = h[:, -1, :].norm(dim=-1, keepdim=True)      # [B,1]
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
        for i in range(0, len(test_s), args.batch_size):
            batch = test_s[i:i + args.batch_size]
            prompts = [build_prompt(x["query"], x["passage"]) for x in batch]
            # NO truncation: pad to batch max only (TODO.md P1-19b). Truncating
            # would cut the "… Answer:" suffix and judge a mid-document token.
            enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
            logits = model(**enc).logits
            for r in range(len(batch)):
                lg = logits[r, -1, :].float()
                yes = torch.logsumexp(lg[yes_ids], 0)
                no = torch.logsumexp(lg[no_ids], 0)
                scores.append(torch.softmax(torch.stack([yes, no]), 0)[0].item())
            del enc, logits
        return np.array(scores)

    def sweep(dvec, name):
        """Run the alpha sweep for one direction; return per-alpha curves."""
        pyes_all, pyes_pos, pyes_neg, aucs = [], [], [], []
        for a in alphas:
            s = judge_scores(dvec, a)
            pyes_all.append(round(float(s.mean()), 4))
            pyes_pos.append(round(float(s[labels == 1].mean()), 4))
            pyes_neg.append(round(float(s[labels == 0].mean()), 4))
            try: aucs.append(round(float(roc_auc_score(labels, s)), 3))
            except ValueError: aucs.append(0.5)
            torch.cuda.empty_cache()
            print(f"    {name:10s} alpha={a:+.0f}  all={pyes_all[-1]:.3f} "
                  f"rel={pyes_pos[-1]:.3f} irrel={pyes_neg[-1]:.3f} auc={aucs[-1]:.3f}")
        gap = [round(p - n, 4) for p, n in zip(pyes_pos, pyes_neg)]
        return {"mean_pyes": pyes_all, "mean_pyes_relevant": pyes_pos,
                "mean_pyes_irrelevant": pyes_neg, "alignment_gap": gap, "auc": aucs}

    from tqdm import tqdm
    report = {
        "tag": args.tag, "model": args.model,
        "best_internal_layer": best, "best_layer_selection": "val",
        "proj_auc_internal": proj_internal, "proj_auc_final": proj_final,
        # direction-quality control (E-8 / R-5): if the random directions' proj-AUC
        # is comparable to internal's, then "internal keeps AUC" cannot be explained
        # by direction quality alone; if internal >> random, it can, and the paper
        # must say so.
        "proj_auc_random": {"mean": round(float(np.mean(rnd_proj)), 4),
                            "min": min(rnd_proj), "max": max(rnd_proj),
                            "values": rnd_proj},
        "alphas": alphas, "n_test": len(test_s),
        "n_test_relevant": int((labels == 1).sum()),
        "n_random": args.n_random,
        "protocol": {"truncation": "none (pad to batch max)",
                     "yes_no_variants": 5,
                     "gap_definition": "mean_pyes[relevant] - mean_pyes[irrelevant]"},
        "sweep": {},
    }
    for name, dvec in [("internal", d_internal), ("final", d_final)]:
        print(f"  direction: {name}")
        report["sweep"][name] = sweep(torch.tensor(dvec, device=dev, dtype=torch.float16), name)

    rand_reports = []
    for i, dvec in enumerate(tqdm(randoms, desc="random directions")):
        rand_reports.append(sweep(torch.tensor(dvec, device=dev, dtype=torch.float16), f"rnd{i:02d}"))
    report["sweep"]["random_all"] = rand_reports

    # --- null distribution for the alignment gap --------------------------
    # Steer "toward relevant" (most negative alpha for the internal direction's
    # convention is the ABLATION arm); we summarize the gap at both extremes and
    # at the ablation end specifically, then ask where internal sits vs random.
    ai = {a: i for i, a in enumerate(alphas)}
    for a in (min(alphas), max(alphas)):
        g = [r["alignment_gap"][ai[a]] for r in rand_reports]
        report[f"gap_at_alpha{a:+.0f}"] = {
            "internal": report["sweep"]["internal"]["alignment_gap"][ai[a]],
            "final": report["sweep"]["final"]["alignment_gap"][ai[a]],
            "random_mean": round(float(np.mean(g)), 4),
            "random_std": round(float(np.std(g)), 4),
            "random_min": round(float(np.min(g)), 4),
            "random_max": round(float(np.max(g)), 4),
            "internal_percentile_vs_random": round(
                float((np.array(g) < report["sweep"]["internal"]["alignment_gap"][ai[a]]).mean()), 4),
        }
        a_auc = [r["auc"][ai[a]] for r in rand_reports]
        report[f"auc_at_alpha{a:+.0f}"] = {
            "internal": report["sweep"]["internal"]["auc"][ai[a]],
            "final": report["sweep"]["final"]["auc"][ai[a]],
            "random_mean": round(float(np.mean(a_auc)), 3),
            "random_min": round(float(np.min(a_auc)), 3),
            "random_max": round(float(np.max(a_auc)), 3),
        }

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(report, open(out, "w"), indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
