#!/usr/bin/env python3
"""
E-3: does the output pathway RECOVER the mid-layer signal when given the REAL
mid-layer activation? (distinguishes "attenuation" from "misalignment")

Why this experiment exists (TODO.md E-3 / R-3)
----------------------------------------------
The paper's title and abstract say the relevance signal is **attenuated** on its
way to the output, which implies the signal *would* reach the output if it were
not damped. But the logit-lens control shows the model's own readout is near
chance at EVERY depth (3B peak 0.560, 8B 0.577, zero training). If a signal were
merely damped, the model's own unembedding should recover at least part of it
somewhere. That evidence points to **misalignment** -- decodable in the residual
stream, but not aligned with the output readout direction.

The existing causal experiment (causal_steer.py) only injects a **diff-of-means
direction**, i.e. a single averaged vector, not the per-example activation. So it
answers "can this direction move the verdict?", not "can the actual mid-layer
representation be read out?".

Design
------
For each test example, take the REAL answer-token residual at the val-selected
internal layer L*, and write it into the residual stream at the FINAL layer
position (same hidden size, so this is a dimension-matched transplant):

    h_final[-1] <- alpha * h_L*[-1] + (1 - alpha) * h_final[-1]

alpha = 0 reproduces the natural forward pass; alpha = 1 fully replaces the
output-layer residual with the internal one. We then read the judge score
P(yes) at the answer token, exactly as llm_judge.py does.

Readers compared (all on the same test rows):
  judge(natural)   : alpha = 0 baseline
  patch_internal   : L* activation, the real thing
  patch_final      : the final layer's own activation (identity; sanity check --
                     must reproduce the judge at all alphas)
  patch_shuffled   : L* activation of a DIFFERENT test example (label-shuffled
                     control -- should NOT improve the verdict)

Diagnostics
-----------
  * AUC of P(yes) vs the TRUE label at each alpha.
  * alignment gap = mean P(yes) | relevant  -  mean P(yes) | irrelevant.
  * query-clustered paired bootstrap for (patch_internal - judge) at each alpha.

Reading it
----------
If transplanting the real internal activation raises AUC toward the probe's
decodability (best_resid ~0.61), the output pathway CAN express the signal and
"attenuation" is defensible. If AUC stays at the judge level while the gap
widens, the internal representation carries information the output readout
cannot use as-is -> misalignment, and the paper's wording must change.

Usage:
  python scripts/patch_internal.py --model meta-llama/Llama-3.2-3B \
      --cache results/cache/llama32_3b_q1500_instruct \
      --judge-cache results/cache/llama32_3b_q1500_judge \
      --tag llama32_3b --n-queries 1500 \
      --out results/patch_internal_llama32_3b_q1500.json
"""
import os, sys, argparse, json
import numpy as np, torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cache_paths import resolve_cache  # noqa: E402
from src.data import load_ms_marco, split_by_query


def build_prompt(query, passage):
    """Byte-identical to extract_instruct.py / llm_judge.py (selfcheck check 6)."""
    return (f"Document: {passage}\nQuestion: {query}\n"
            f"Is the document relevant to the question? Answer:")


def yes_no_ids(tok):
    def first(words):
        s = set()
        for w in words:
            e = tok.encode(w, add_special_tokens=False)
            if e: s.add(e[0])
        return sorted(s)
    return (first(["yes", "Yes", "YES", " yes", " YES"]),
            first(["no", "No", "NO", " no", " NO"]))


def boot_indices(qids, n_boot, rng):
    idx_by_q = {q: np.where(qids == q)[0] for q in np.unique(qids)}
    uq = np.unique(qids)
    return [np.concatenate([idx_by_q[q] for q in rng.choice(uq, size=len(uq), replace=True)])
            for _ in range(n_boot)]


def boot_auc(y, s, sets):
    return np.array([roc_auc_score(y[i], s[i]) if len(np.unique(y[i])) > 1 else 0.5
                     for i in sets])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--judge-cache", default=None)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--n-queries", type=int, default=1500)
    ap.add_argument("--max-passages", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--layer", type=int, default=None,
                    help="internal layer to transplant; default = val-selected from cache")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="results/patch_internal.json")
    args = ap.parse_args()
    args.cache = resolve_cache(args.cache)
    if args.judge_cache:
        args.judge_cache = resolve_cache(args.judge_cache)
    alphas = [float(a) for a in args.alphas.split(",")]

    # ---- which internal layer: VAL-selected, exactly like significance.py ----
    vd = torch.load(os.path.join(args.cache, "resid_val.pt"))
    Rva = vd["acts"].to(torch.float32).numpy(); yva = vd["labels"].numpy()
    td = torch.load(os.path.join(args.cache, "resid_test.pt"))
    Rte = td["acts"].to(torch.float32).numpy(); yte = td["labels"].numpy()
    qte = td["query_ids"].numpy() if "query_ids" in td else None
    trd = torch.load(os.path.join(args.cache, "resid_train.pt"))
    Rtr = trd["acts"].to(torch.float32).numpy(); ytr = trd["labels"].numpy()

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    def layer_val_auc(li):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000,
                          class_weight="balanced", random_state=42))
        m.fit(Rtr[:, li, :], ytr)
        return roc_auc_score(yva, m.predict_proba(Rva[:, li, :])[:, 1])

    if args.layer is None:
        from joblib import Parallel, delayed
        n_layers = Rtr.shape[1]
        vaucs = Parallel(n_jobs=max(1, min(32, (os.cpu_count() or 4) - 2)))(
            delayed(layer_val_auc)(li) for li in range(n_layers - 1))
        L_star = int(np.argmax(vaucs))
    else:
        L_star = args.layer
    tr_auc = layer_val_auc(L_star)
    print(f"[{args.tag}] transplant layer L*={L_star} (val AUC {tr_auc:.3f})")

    # probe's own test AUC at L* -- the ceiling the patch could in principle reach
    m_star = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000,
                        class_weight="balanced", random_state=42)).fit(Rtr[:, L_star, :], ytr)
    probe_test_auc = roc_auc_score(yte, m_star.predict_proba(Rte[:, L_star, :])[:, 1])

    # ---- data + model -----------------------------------------------------
    samples = load_ms_marco(n_queries=args.n_queries,
                            max_passages_per_query=args.max_passages, seed=args.seed)
    _, _, test_s = split_by_query(samples, seed=args.seed)
    labels = np.array([x["label"] for x in test_s])

    # the activation cache and the freshly loaded samples must be the same rows
    assert len(test_s) == len(yte), \
        f"row count mismatch: samples={len(test_s)} cache={len(yte)}"
    print(f"test rows: {len(test_s)} (cache {len(yte)}), pos={int(labels.sum())}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {args.model} ...")
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
    state = {"mode": "off", "alpha": 0.0, "internal": None,
             "layer_idx": L_star, "final_idx": len(layers) - 1}

    def hook(_m, _inp, out):
        if state["mode"] == "off" or state["internal"] is None:
            return out
        h = out[0] if isinstance(out, tuple) else out       # [B,S,H]
        a = state["alpha"]
        h[:, -1, :] = a * state["internal"] + (1.0 - a) * h[:, -1, :]
        return out

    last = layers[-1]
    last.register_forward_hook(hook)

    def mid_hidden(batch_prompts, layer_idx):
        """Capture the real answer-token residual at `layer_idx` for a batch."""
        captured = {}

        def cap(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured["h"] = h[:, -1, :].detach()
            return out
        handle = layers[layer_idx].register_forward_hook(cap)
        try:
            enc = tok(batch_prompts, return_tensors="pt", padding=True).to(dev)
            with torch.no_grad():
                model(**enc)
            del enc
        finally:
            handle.remove()
        torch.cuda.empty_cache()
        return captured["h"]

    @torch.no_grad()
    def judge_scores(mode, alpha, internal=None):
        """mode='off' -> natural forward pass; otherwise transplant `internal`."""
        out = []
        state["internal"] = internal
        for i in range(0, len(test_s), args.batch_size):
            batch = test_s[i:i + args.batch_size]
            prompts = [build_prompt(x["query"], x["passage"]) for x in batch]
            # per-batch alpha: index 0 of the batch is never patched (see below)
            h = None if internal is None else internal[i:i + args.batch_size]
            if h is None:
                state["mode"], state["alpha"] = "off", 0.0
            else:
                state["mode"], state["alpha"] = "on", alpha
                state["internal"] = h
            enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
            logits = model(**enc).logits
            for r in range(len(batch)):
                lg = logits[r, -1, :].float()
                yes = torch.logsumexp(lg[yes_ids], 0)
                no = torch.logsumexp(lg[no_ids], 0)
                out.append(torch.softmax(torch.stack([yes, no]), 0)[0].item())
            del enc, logits
        state["mode"], state["internal"] = "off", None
        torch.cuda.empty_cache()
        return np.array(out)

    # ---- 1. natural forward pass (= judge) --------------------------------
    s_judge = judge_scores("off", 0.0)
    if args.judge_cache:
        jd = torch.load(os.path.join(args.judge_cache, "judge_test.pt"))
        assert np.array_equal(jd["labels"].numpy(), yte), "judge/resid label mismatch"
        if "query_ids" in jd and qte is not None:
            assert np.array_equal(jd["query_ids"].numpy(), qte), \
                "judge/resid QUERY_ID mismatch -- caches describe different rows"
        ja = roc_auc_score(jd["labels"].numpy(), jd["scores"].numpy())
        print(f"sanity: our re-run judge AUC {roc_auc_score(yte, s_judge):.4f} "
              f"vs cached judge {ja:.4f} (|diff|={abs(roc_auc_score(yte, s_judge)-ja):.4f})")

    # ---- 2. capture the real mid-layer activations for every test row -----
    H_int = []
    for i in range(0, len(test_s), args.batch_size):
        batch = test_s[i:i + args.batch_size]
        prompts = [build_prompt(x["query"], x["passage"]) for x in batch]
        H_int.append(mid_hidden(prompts, L_star).to(torch.float16))
    H_int = torch.cat(H_int, dim=0)                     # [N, H]
    print(f"captured internal activations: {tuple(H_int.shape)}")

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(test_s))
    H_shuf = H_int[perm]                                # label-shuffled control

    # ---- 3. sweeps --------------------------------------------------------
    def summarize(name, scores):
        auc = float(roc_auc_score(labels, scores))
        gap = float(scores[labels == 1].mean() - scores[labels == 0].mean())
        return {"auc": round(auc, 4),
                "mean_pyes": round(float(scores.mean()), 4),
                "mean_pyes_relevant": round(float(scores[labels == 1].mean()), 4),
                "mean_pyes_irrelevant": round(float(scores[labels == 0].mean()), 4),
                "alignment_gap": round(gap, 4)}

    report = {
        "tag": args.tag, "model": args.model, "layer": int(L_star),
        "layer_selection": "val", "alphas": alphas,
        "n_test": int(len(test_s)), "probe_test_auc_at_layer": round(float(probe_test_auc), 4),
        "sweep": {},
    }

    curves = {}
    for a in alphas:
        row = {}
        # identity control: transplanting the final layer's OWN activation into
        # the final layer is a no-op, so this must equal the natural judge at
        # EVERY alpha (it re-runs the forward pass to prove the hook is inert
        # when the payload is the layer's own residual -- i.e. `off`).
        for name, h in (("patch_internal", H_int), ("patch_final", None),
                        ("patch_shuffled", H_shuf)):
            if name == "patch_final":
                s = judge_scores("off", 0.0)         # row-identical to judge
            else:
                s = judge_scores(name, a, h.to(dev))
            row[name] = summarize(name, s)
            row[name + "_scores"] = s
            print(f"  alpha={a:4.2f} {name:15s} AUC={row[name]['auc']:.4f} "
                  f"gap={row[name]['alignment_gap']:+.4f}", flush=True)
        curves[str(a)] = row
        report["sweep"][str(a)] = {k: v for k, v in row.items() if not k.endswith("_scores")}

    # ---- 4. paired bootstrap: patch_internal - judge at each alpha --------
    if qte is not None:
        idx_sets = boot_indices(qte, args.boot, rng)
        bj = boot_auc(yte, s_judge, idx_sets)
        report["edges"] = {}
        for a in alphas:
            bi = boot_auc(yte, curves[str(a)]["patch_internal_scores"], idx_sets)
            d = bi - bj
            report["edges"][f"patch_internal-judge@a={a}"] = {
                "mean": round(float(d.mean()), 4),
                "ci95": [round(float(np.percentile(d, 2.5)), 4),
                         round(float(np.percentile(d, 97.5)), 4)],
                "p_le0": round(float((d <= 0).mean()), 4),
            }
    report["judge_natural"] = summarize("judge", s_judge)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "sweep"}, indent=2)[:2000])
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
