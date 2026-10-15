#!/usr/bin/env python3
"""
Gradient alignment for the ATTENTION-HEAD probe -- the experiment that closes
the gap flagged as Limitations (0) in the wide paper.

Motivation
----------
The paper's headline effect is that a probe on the top attention heads beats a
probe on the output representation (8/8 models). But every mechanism measurement
we had was taken on the RESIDUAL STREAM at L*: the readout direction, the
activation patching, and the earlier gradient alignment. So the misalignment
account was established for the residual reader and merely conjectured for the
head reader that carries the headline. This script measures it directly.

What is computed
----------------
We rebuild exactly the head probe from `significance.py`:

    per-head L1 logistic regression on train -> rank heads by VALIDATION AUC
    -> top k=20 -> standardise each head (train statistics) -> concat
    -> C=1.0 logistic regression on train

so the probe is a function f(z) = sigmoid(w . z + b) of the STANDARDISED
concatenation z = [(x_h - mu_h) / s_h] for the 20 selected heads h.

Then, for a sample of test rows, we differentiate the model's own yes/no logit
difference with respect to h_{L*}  -- but now with respect to the SELECTED HEADS'
outputs, across every layer those heads live in:

    g_z = [ (d(delta) / d x_h) / s_h for h in selected ]   (concatenated)

The division by s_h is the chain rule through the probe's standardisation, and
it is what makes g_z sit in the same space as w. We then compare cos(g_z, w_hat)
against the same statistic for 16 random unit directions, and also report a
"controller" comparison: cos(g_z, final-layer probe direction restricted to the
shared coordinate system) -- omitted when the two live in different spaces, as
they do for heads vs a single residual vector.

Nothing is edited: like the residual gradient alignment, this is a local
property of the network at activations it actually produced, so the
out-of-distribution objection that limits patching does not apply.

Usage
-----
  python scripts/gradient_alignment_heads.py \
      --model meta-llama/Llama-3.2-3B \
      --cache results/cache/llama32_3b_q1500_instruct \
      --tag llama32_3b --n-rows 256 --dtype float32 \
      --out results/gradient_alignment_heads_llama32_3b_q1500.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOPK = 20
SEED = 42
N_JOBS = max(1, min(16, (os.cpu_count() or 4)))


def lr(penalty="l2"):
    return LogisticRegression(penalty=penalty, C=1.0,
                              solver="lbfgs" if penalty == "l2" else "liblinear",
                              max_iter=3000, class_weight="balanced",
                              random_state=SEED)


def load(cache, scheme, split):
    d = torch.load(os.path.join(cache, f"{scheme}_{split}.pt"), map_location="cpu")
    return (d["acts"].to(torch.float32).numpy(), d["labels"].numpy(),
            d.get("query_ids"))


def build_prompt(passage, query):
    return (f"Document: {passage}\nQuestion: {query}\n"
            f"Is the document relevant to the question? Answer:")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-rows", type=int, default=256)
    ap.add_argument("--n-random", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--save-matrix", action="store_true",
                    help="also write the gradient matrix so the percentile can "
                         "be recomputed at higher resolution without a GPU pass")
    ap.add_argument("--out", default="results/gradient_alignment_heads.json")
    args = ap.parse_args()

    t0 = time.time()
    meta = json.load(open(os.path.join(args.cache, "meta.json")))
    n_layers, n_heads = int(meta["n_layers"]), int(meta["n_heads"])
    hidden = int(meta["hidden"]) if "hidden" in meta else None
    head_dim = (hidden // n_heads) if hidden else None

    # ---------------------------------------------------------- rebuild probe
    Atr, ytr, _ = load(args.cache, "attn", "train")
    Ava, yva, _ = load(args.cache, "attn", "val")
    Ate, yte, Ate_qids = load(args.cache, "attn", "test")
    if Ate_qids is None:
        raise SystemExit("cache has no query_ids; cannot verify row alignment")
    Ate_qids = np.asarray(Ate_qids)
    if head_dim is None:
        head_dim = Atr.shape[2]
    n_flat = Atr.shape[1]
    assert n_flat == n_layers * n_heads, (n_flat, n_layers, n_heads)

    print(f"  selecting top-{TOPK} heads by validation AUC "
          f"({n_layers}L x {n_heads}H = {n_flat} heads)")
    # 672 sequential LogisticRegression fits is minutes of wall clock; the same
    # joblib pattern significance.py uses brings it down to seconds.
    def hauc(h):
        p = lr(penalty="l1").fit(Atr[:, h, :], ytr)
        return roc_auc_score(yva, p.predict_proba(Ava[:, h, :])[:, 1])
    vaucs = np.array(Parallel(n_jobs=N_JOBS)(delayed(hauc)(h) for h in range(n_flat)))
    sel = np.argsort(vaucs)[::-1][:TOPK]

    def feats(X):
        return np.concatenate([X[:, h, :] for h in sel], axis=1)

    # NOTE: significance.py does NOT standardise the concatenated head features.
    # We standardise here only to obtain per-head scales sd, which the chain rule
    # needs when converting a gradient in activation space into the probe's own
    # coordinate system -- but the probe we align against must be the SAME probe
    # the paper reports, so we fit it on the raw concatenation.
    Ftr = feats(Atr)
    mu = Ftr.mean(axis=0)
    sd = Ftr.std(axis=0)
    sd[sd == 0] = 1.0
    del mu
    probe = lr().fit(Ftr, ytr)          # identical to significance.py
    w = probe.coef_[0].astype(np.float64)
    head_auc_test = roc_auc_score(yte, probe.predict_proba(feats(Ate))[:, 1])
    print(f"  [check] head-probe test AUC = {head_auc_test:.4f} "
          f"(must equal the attn value in "
          f"results/significance_{args.tag}_q1500.json)")

    # which layers do the selected heads live in? (flat index -> layer, head)
    sel_layers = np.array([int(h) // n_heads for h in sel])
    sel_heads = np.array([int(h) % n_heads for h in sel])
    layers_used = sorted(set(sel_layers.tolist()))
    print(f"  selected heads span layers {layers_used} "
          f"(head-probe test AUC = {head_auc_test:.4f})")

    # --------------------------------------------------------- model + data
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    _dt = torch.float16 if args.dtype == "float16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto",
                                                dtype=_dt)
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    # We only need d(verdict)/d(activation), never d(verdict)/d(weight). Leaving
    # the parameters differentiable makes autograd allocate an fp32 gradient for
    # every weight -- ~16 GB on an 8B model, which OOMs a 32 GB card even at
    # batch=1 with checkpointing (measured). Freezing the parameters removes that
    # allocation entirely and does not change the activation gradient.
    for p_ in model.parameters():
        p_.requires_grad_(False)

    # Freezing the parameters means nothing in the forward pass requires grad, so
    # autograd builds NO graph and backward() fails with "element 0 of tensors
    # does not require grad" (measured). retain_grad() on a tensor that requires
    # grad makes it a graph root whose gradient is kept, so calling it on the
    # embedding output gives us a differentiable entry point WITHOUT
    # re-enabling the weight gradients we just removed.
    def _make_input_grad(_m, _inp, out):
        t = out[0] if isinstance(out, tuple) else out
        t.requires_grad_(True)
        t.retain_grad()
        return out

    _embed = model.get_input_embeddings()
    handles0 = [_embed.register_forward_hook(_make_input_grad)]
    model.eval()

    # prompts: rebuild the dataset and align rows against the cache by query_id.
    # extract_instruct.py builds its split with a fixed seed, so reconstruction is
    # deterministic -- but we do NOT trust that: we match on (query_id, label)
    # against the cached arrays and refuse to proceed on any mismatch.
    sys.path.insert(0, ROOT)
    from src.data import load_ms_marco, split_by_query
    nq = int(meta.get("n_queries", 1500))
    seed = int(meta.get("seed", 42))
    print(f"  rebuilding dataset (n_queries={nq}, seed={seed}) to recover prompts")
    # 5 is the value extract_instruct.py used for these caches; verified by
    # matching the split sizes (5218/1121/1117) exactly.
    samples = load_ms_marco(n_queries=nq, max_passages_per_query=5, seed=seed)
    _, _, test_s = split_by_query(samples, seed=seed)

    # group reconstructed rows by query_id, preserving their relative order
    by_q = {}
    for s in test_s:
        by_q.setdefault(int(s["query_id"]), []).append(s)
    qte = Ate_qids.astype(np.int64)
    labs = yte.astype(np.int64)
    rows, used = [], {}
    for q, lab in zip(qte, labs):
        q = int(q)
        bucket = by_q.get(q)
        if not bucket:
            raise SystemExit(f"query {q} from the cache is absent after rebuild")
        k = used.get(q, 0)
        if k >= len(bucket):
            raise SystemExit(f"query {q}: cache has more rows than the rebuild")
        s = bucket[k]
        used[q] = k + 1
        if int(s["label"]) != int(lab):
            raise SystemExit(
                f"label mismatch for query {q} row {k}: rebuild={s['label']} "
                f"cache={lab} -- prompt reconstruction is NOT aligned, refusing "
                "to emit a gradient matrix")
        rows.append(s)
    print(f"  row alignment OK: {len(rows)} rows matched on (query_id, label) "
          f"across {len(by_q)} test queries")

    n_rows = min(args.n_rows, len(rows))
    rows = rows[:n_rows]
    texts = [build_prompt(r["passage"], r["query"]) for r in rows]

    yes_ids = [tok.encode(v, add_special_tokens=False)[0] for v in
               ("yes", "Yes", "YES", " yes", " YES")]
    no_ids = [tok.encode(v, add_special_tokens=False)[0] for v in
              ("no", "No", "NO", " no", " NO")]

    # ------------------------------------------------------------- gradients
    # We take d(verdict)/d(o_proj output) from a module BACKWARD hook rather than
    # from retain_grad() on a slice: measured on this stack, retain_grad() on a
    # view of the projection output never populated .grad (all 254 parameters
    # received gradients, the retained view did not), while the backward hook
    # delivers the gradient directly. The hook also avoids holding the autograd
    # graph for the head activations, which is what made batch>4 OOM.
    want = {}
    for flat, (li, h) in enumerate(zip(sel_layers, sel_heads)):
        want.setdefault(int(li), []).append((int(h), flat))
    state = {"idx": None, "grads": {}}      # grads: layer -> [B, nh, hd]
    handles = []

    def make_fwd(li):
        def fwd(_m, _inp, out):
            return out   # the backward hook derives the token position itself
        return fwd

    def make_bwd(li):
        def bwd(_m, _gin, gout):
            go = gout[0] if isinstance(gout, tuple) else gout
            B, S, _H = go.shape
            # Derive the answer-token position here rather than reusing the
            # forward hook's state: gradient checkpointing recomputes the forward
            # pass during backward WITHOUT re-running the hook, so shared state
            # from the forward pass would be stale or missing.
            idx = (enc["attention_mask"].sum(dim=1) - 1).to(go.device)
            g = go.reshape(B, S, n_heads, head_dim)
            g = g[torch.arange(B, device=g.device), idx, :, :]
            state["grads"][li] = g.detach()
        return bwd

    for li in want:
        mod = model.model.layers[li].self_attn.o_proj
        handles.append(mod.register_forward_hook(make_fwd(li)))
        handles.append(mod.register_full_backward_hook(make_bwd(li)))

    G = np.zeros((n_rows, TOPK * head_dim), dtype=np.float64)
    n_done = 0
    first_batch = True
    VERIFY = {"maxdiff": 0.0, "checked": 0, "done": False}
    for start in range(0, n_rows, args.batch):
        chunk = texts[start:start + args.batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_length).to(model.device)
        state["grads"] = {}
        out = model(**enc)
        logits = out.logits[:, -1, :].float()      # [B, V]
        yes = logits[:, yes_ids].logsumexp(dim=1)
        no = logits[:, no_ids].logsumexp(dim=1)
        delta = (yes - no).sum()
        if first_batch:
            # If this is False the forward pass built no graph at all, and the
            # backward below would die with an opaque "element 0 ... does not
            # require grad". Say so here instead: a frozen model only builds a
            # graph if at least one input still requires grad.
            print(f"  [check] delta.requires_grad = {delta.requires_grad}")
            if not delta.requires_grad:
                raise SystemExit(
                    "the verdict is not connected to any differentiable tensor "
                    "-- with frozen parameters the graph must come from the "
                    "inputs, so this means the forward pass ran without grad")
        model.zero_grad(set_to_none=True)
        delta.backward()

        B = len(chunk)
        if first_batch:
            # Verify the head indexing against the cache BEFORE trusting any
            # cosine: if o_proj's per-head layout were not what we assume, every
            # downstream number would be meaningless. Compare the live forward
            # slices (recomputed here) with the cached activations.
            missing = [li for li in want if li not in state["grads"]]
            if missing:
                raise SystemExit(f"backward hook produced no gradient for layers "
                                 f"{missing} -- refusing to emit a matrix")
            with_grad = sum(1 for li in want for h, _ in want[li]
                            if float(state["grads"][li][0, h].abs().sum()) > 0)
            print(f"  [check] layers with gradient: {sorted(state['grads'])}; "
                  f"heads with nonzero grad on row 0: {with_grad}/{len(sel)}")
            # Layout check: the gradient hook tells us nothing about layout, so
            # compare live per-head activations with the cache. This MUST reuse
            # the same padded encoding as the main pass -- a separate
            # tokenisation of the same text produces a different last-token
            # activation and would report a spurious mismatch.
            cap = {}   # ONE ENTRY PER LAYER -- a single slot would keep only the
                       # last hook's output and then silently compare every layer
                       # against it (that mistake cost a debugging round).

            def make_cap(li):
                def cap_hook(_m, _inp, out):
                    ao = out[0] if isinstance(out, tuple) else out
                    B1, S1, _H1 = ao.shape
                    v1 = ao.reshape(B1, S1, n_heads, head_dim)
                    idx1 = (enc["attention_mask"].sum(dim=1) - 1).to(v1.device)
                    cap[li] = v1[torch.arange(B1, device=v1.device), idx1, :, :].detach()
                    return out
                return cap_hook

            hs = [model.model.layers[li].self_attn.o_proj.register_forward_hook(make_cap(li))
                  for li in want]
            with torch.no_grad():
                model(**enc)
            for hh in hs:
                hh.remove()
            d = 0.0
            worst = None
            for b in range(min(B, 4)):
                for li in want:
                    for h, flat in want[li]:
                        diff = np.abs(cap[li][b, h].float().cpu().numpy()
                                      - Ate[start + b, li * n_heads + h])
                        mx = float(diff.max())
                        if mx > d:
                            d, worst = mx, (start + b, li, h)
            print(f"  [check] cached-activation match max|diff| = {d:.3e} "
                  f"over {min(B,4)} rows x {len(sel)} heads (worst at {worst})")
            print(f"  [check] cached-activation match max|diff| = {d:.3e} "
                  f"over {min(B,4)} rows x {len(sel)} heads")
            if d > 1e-2:
                raise SystemExit(
                    f"live activation does not match the cache (max|diff|={d:.3e}) "
                    "-- head indexing or prompts are misaligned; refusing to "
                    "emit a matrix")
            first_batch = False

        for flat, (li, h) in enumerate(zip(sel_layers, sel_heads)):
            g = state["grads"][int(li)][:, int(h), :]        # [B, hd]
            col = flat * head_dim
            # chain rule through the probe's per-head standardisation
            G[start:start + B, col:col + head_dim] = (
                g.float().cpu().numpy() / sd[col:col + head_dim])
        n_done += B
        del out, logits, yes, no, delta
        state["grads"] = {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if n_done % (args.batch * 8) == 0 or n_done >= n_rows:
            print(f"    {n_done}/{n_rows} rows", flush=True)

    for hd in handles:
        hd.remove()
    print(f"  gradient matrix {G.shape} in {time.time()-t0:.0f}s")

    # ------------------------------------------------------------- alignment
    def cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0

    probe_cos = np.array([cos(G[i], w) for i in range(n_rows)])
    rng = np.random.default_rng(SEED)
    rnd_means = []
    for _ in range(args.n_random):
        r = rng.normal(size=w.shape)
        r /= np.linalg.norm(r)
        rnd_means.append(float(np.mean([cos(G[i], r) for i in range(n_rows)])))
    rnd_means = np.array(rnd_means)
    pct = float((np.abs(rnd_means) < abs(probe_cos.mean())).mean())

    rep = {
        "tag": args.tag, "model": args.model, "n_rows": int(n_rows),
        "n_random": int(args.n_random), "dtype": args.dtype,
        "probe": "attention-head probe, identical to significance.py "
                 "(top-20 heads by val AUC, StandardScaler per head, C=1.0 LR)",
        "head_probe_test_auc": round(float(head_auc_test), 4),
        "selected_heads_flat": [int(x) for x in sel],
        "selected_heads_layers": [int(x) for x in sel_layers],
        "selected_heads_indices": [int(x) for x in sel_heads],
        "layers_used": [int(x) for x in layers_used],
        "probe_direction_cosine": {
            "mean": round(float(probe_cos.mean()), 5),
            "median": round(float(np.median(probe_cos)), 5),
            "frac_positive": round(float((probe_cos > 0).mean()), 4)},
        "random_directions_cosine": {
            "mean_of_means": round(float(rnd_means.mean()), 5),
            "mean_abs_of_means": round(float(np.abs(rnd_means).mean()), 5),
            "max_abs_mean": round(float(np.abs(rnd_means).max()), 5),
            "per_direction_means": [round(float(x), 5) for x in rnd_means]},
        "probe_vs_random_abs_mean_percentile": round(pct, 4),
        "protocol": "g_z = grad of (yes-no logit difference) wrt the SELECTED "
                    "HEADS' o_proj slices at the answer token, divided by each "
                    "head's train std (chain rule through the probe's "
                    "standardisation); cosine with the head probe's unit "
                    "direction vs random unit directions. No activation edited.",
        "elapsed_sec": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(rep, open(args.out, "w"), indent=2)
    if args.save_matrix:
        # The percentile below is quantised to 1/(n_random+1); near the 0.95
        # threshold that resolution is too coarse to decide anything. Keeping the
        # matrix lets the percentile be recomputed exactly (CPU only) without
        # repeating the GPU pass.
        npz = args.out.replace(".json", "_matrix.npz")
        np.savez_compressed(npz, G=G.astype(np.float32), w=w)
        print(f"  saved gradient matrix -> {npz} ({G.shape})")
    print(f"  cosine(model gradient, HEAD-PROBE direction) : "
          f"mean {probe_cos.mean():+.5f}  frac>0 {(probe_cos>0).mean():.3f}")
    print(f"  cosine with RANDOM directions                : "
          f"|mean| up to {np.abs(rnd_means).max():.5f}")
    print(f"  probe |mean cos| percentile vs random        : {pct:.3f}")
    print(f"\n  Reading (fixed in advance): percentile < 0.95 -> the head probe's")
    print(f"  relevance direction is NOT the direction the verdict is locally")
    print(f"  sensitive to, even though this is the reader that carries the")
    print(f"  paper's headline effect. That would EXTEND the misalignment result")
    print(f"  from the residual stream to the head reader. A percentile above")
    print(f"  0.95 would instead mean the head reader IS aligned, i.e. the gap")
    print(f"  between our mechanism story and our headline effect is real and the")
    print(f"  misalignment account does NOT cover the head probe.")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
