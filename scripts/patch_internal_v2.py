#!/usr/bin/env python3
"""
CAUSAL mechanism test, v2 -- removes the SCALE CONFOUND from patch_internal.py.

Why v2 exists (2026-10-14)
--------------------------
`patch_internal.py` (v1) transplanted the REAL L* activation into the FINAL layer
residual as  h_final <- a*h_L* + (1-a)*h_final.  Measured per-layer residual norms
show that h_L* is 4x-16x SMALLER than h_final (3B L9 7.3 vs L27 70.1 = 9.6x;
8B L8 5.3 vs L31 86.4 = 16.3x; 3B-Inst 4.0x; Qwen 3.6x), and the v1 AUC collapse
tracked that ratio almost monotonically.  So v1 cannot separate

    (i)  FORMAT: the L* representation is not in a form the output readout can use
    (ii) SCALE : we injected a vector an order of magnitude too small

and its conclusion ("misalignment") was therefore under-determined.  v2 fixes that
and also asks the question v1 never actually asked.

The residual cache is `layers[i].output[0, -1, :]`, i.e. the residual-stream
coordinate AFTER decoder layer i -- and the hook point for "the final layer" is
`layers[-1].output`, the SAME kind of coordinate.  Therefore replacing the
answer-token activation at layer L* (or L*-1) with another sample's activation is
legal: same stream, same coordinate system, same distribution. No reinterpretation.

Experiments (all on the test split, one model load)
---------------------------------------------------
A. SELF patch at L*   : h <- h_self.  The payload is the sample's OWN activation,
   so this is the identity up to fp16/kernels -> MUST reproduce the natural judge
   verdict. It is the ceiling AND the sanity check.

B. CROSS patch at L*  : h_i <- h_j (random partner j, preferentially a DIFFERENT
   label).  This is the causal analogue of a probe.  The probe can decode relevance
   from h at L* with AUC ~0.6; the question here is whether the MODEL's OWN
   downstream computation uses that content.  Metrics:
     - auc_vs_source : AUC(model verdict, source label) -- does the verdict FOLLOW
       the transplanted activation? compared against auc_vs_target (natural labels)
     - auc_vs_target : how much of the target's own signal survives
   Both are reported, plus a per-label breakdown so `follow` is legible.
   Interpolation sweep  h <- a*h_j + (1-a)*h_i  traces the crossover.

C. CROSS patch at the FINAL layer, NORM-MATCHED (the key control):
   h_final_i <- a * (||h_final_i|| / ||h_L*_j||) * h_L*_j + (1-a) * h_final_i
   i.e. the same transplant as v1 but with the injected vector rescaled to the
   final layer's own norm. If discriminability is still destroyed -> format, not
   scale. Also run WITHOUT the rescale (v1 behaviour) in the same script so the
   two are directly comparable on identical batches.

D. Controls: identity, label-shuffled payloads, and a random-direction injection.

Judgement (pre-registered, do not adjust after seeing numbers)
-------------------------------------------------------------
  * If B shows auc_vs_source clearly > 0.5 and rising with alpha -> the internal
    state IS causally used, the model is not "attenuating" a signal it never had;
    the story is about WHERE it is readable.
  * If B is flat at 0.5 while the probe gets 0.6 at the same layer -> the decodable
    content is NOT what the model computes with (epiphenomenal to the verdict).
  * If C still collapses AUC after norm matching -> v1's conclusion survives its
    confound and "misalignment" is supported.
  * If C recovers -> v1's conclusion was a scale artefact and must be retracted.

Usage:
  python scripts/patch_internal_v2.py --model meta-llama/Llama-3.2-3B \
      --cache results/cache/llama32_3b_q1500_instruct --tag llama32_3b \
      --out results/patch_v2_llama32_3b_q1500.json
"""
import os, sys, argparse, json, time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cache_paths import resolve_cache  # noqa: E402


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


def boot_indices(qids, n_boot, rng):
    idx_by_q = {q: np.where(qids == q)[0] for q in np.unique(qids)}
    uq = np.unique(qids)
    return [np.concatenate([idx_by_q[q] for q in rng.choice(uq, size=len(uq), replace=True)])
            for _ in range(n_boot)]


def boot_auc(y, s, sets):
    out = []
    for i in sets:
        yy, ss = y[i], s[i]
        out.append(roc_auc_score(yy, ss) if len(np.unique(yy)) > 1 else 0.5)
    return np.array(out)


def edge(y, s_a, s_b, sets):
    """paired bootstrap of AUC(a) - AUC(b) on shared resamples."""
    d = boot_auc(y, s_a, sets) - boot_auc(y, s_b, sets)
    return {"mean": round(float(d.mean()), 4),
            "ci95": [round(float(np.percentile(d, 2.5)), 4),
                     round(float(np.percentile(d, 97.5)), 4)],
            "p_le0": round(float((d <= 0).mean()), 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--judge-cache", default=None)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--n-queries", type=int, default=1500)
    ap.add_argument("--max-passages", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cross-alphas", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--final-alphas", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="results/patch_v2.json")
    args = ap.parse_args()
    args.cache = resolve_cache(args.cache)
    cross_alphas = [float(a) for a in args.cross_alphas.split(",")]
    final_alphas = [float(a) for a in args.final_alphas.split(",")]

    # ---- residuals: labels, query ids, and the val-selected internal layer ----
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

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from joblib import Parallel, delayed

    def layer_val_auc(li):
        m = make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=42))
        m.fit(Rtr[:, li, :], ytr)
        return roc_auc_score(yva, m.predict_proba(Rva[:, li, :])[:, 1])

    if args.layer is None:
        vaucs = Parallel(n_jobs=max(1, min(32, (os.cpu_count() or 4) - 2)))(
            delayed(layer_val_auc)(li) for li in range(n_layers - 1))
        L_star = int(np.argmax(vaucs))
    else:
        L_star = args.layer

    # probe ceiling at L* (the number the causal test is compared against)
    m_star = make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=3000, class_weight="balanced", random_state=42)).fit(Rtr[:, L_star, :], ytr)
    probe_test_auc = float(roc_auc_score(yte, m_star.predict_proba(Rte[:, L_star, :])[:, 1]))
    print(f"[{args.tag}] L*={L_star} (val AUC {layer_val_auc(L_star):.3f}), "
          f"probe test AUC at L* = {probe_test_auc:.4f}, test rows={len(yte)}")

    # ---- rebuild the exact test prompts (same order as the cache) ----------
    from src.data import load_ms_marco, split_by_query
    samples = load_ms_marco(n_queries=args.n_queries,
                            max_passages_per_query=args.max_passages, seed=args.seed)
    _, _, test_s = split_by_query(samples, seed=args.seed)
    assert len(test_s) == len(yte), f"row mismatch samples={len(test_s)} cache={len(yte)}"
    labels = np.array([x["label"] for x in test_s])
    assert np.array_equal(labels, yte), "cache/sample LABEL mismatch"
    prompts = [build_prompt(x["query"], x["passage"]) for x in test_s]
    print(f"test rows: {len(test_s)} pos={int(labels.sum())}")

    # ---- random cross partners, preferring a DIFFERENT label --------------
    rng = np.random.default_rng(args.seed)
    partner = np.arange(len(yte))
    for i in range(len(yte)):
        pool = np.where(yte != yte[i])[0]
        partner[i] = int(rng.choice(pool)) if len(pool) else int(rng.choice(len(yte)))
    n_diff = int((yte[partner] != yte).sum())
    print(f"cross partners with a DIFFERENT label: {n_diff}/{len(yte)}")

    # ---- model ------------------------------------------------------------
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
    assert n_layers == len(layers), f"cache has {n_layers} layers, model has {len(layers)}"
    H = model.config.hidden_size

    # state: inject[site] with per-row source vectors
    state = {"site": None, "alpha": 1.0, "src": None, "norm_ref": None}
    sites = {"Lstar": L_star, "final": len(layers) - 1}
    handles = {}
    for name, li in sites.items():
        def make_hook(nm):
            def hook(_m, _inp, out):
                if state["site"] != nm or state["src"] is None:
                    return out
                h = out[0] if isinstance(out, tuple) else out
                a = state["alpha"]
                src = state["src"]
                if state["norm_ref"] is not None:
                    # rescale the payload to the TARGET row's own norm
                    tgt_norm = h[:, -1, :].norm(dim=-1, keepdim=True)
                    src_norm = src.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    src = src * (tgt_norm / src_norm)
                h[:, -1, :] = a * src + (1.0 - a) * h[:, -1, :]
                return out
            return hook
        handles[name] = layers[li].register_forward_hook(make_hook(name))

    @torch.no_grad()
    def run(prompt_list, capture=None, site=None, alpha=1.0, src=None, norm_ref=None):
        """One pass. capture='Lstar' collects the L* answer-token hidden states.

        Returns (scores, pos_mean, neg_mean, captured[h]).
        """
        state["site"], state["alpha"], state["src"], state["norm_ref"] = site, alpha, src, norm_ref
        cap = {}

        def cap_hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            cap["h"] = h[:, -1, :].detach().clone()
            return out

        # prepend=True so we always read the layer's natural output, never the
        # value written by the injection hook of the same module
        handle = (layers[L_star].register_forward_hook(cap_hook, prepend=True)
                  if capture else None)
        scores = []
        try:
            for i in range(0, len(prompt_list), args.batch_size):
                chunk = prompt_list[i:i + args.batch_size]
                enc = tok(chunk, return_tensors="pt", padding=True).to(dev)
                logits = model(**enc).logits
                for r in range(len(chunk)):
                    lg = logits[r, -1, :].float()
                    yes = torch.logsumexp(lg[yes_ids], 0)
                    no = torch.logsumexp(lg[no_ids], 0)
                    scores.append(torch.softmax(torch.stack([yes, no]), 0)[0].item())
                del enc, logits
        finally:
            if handle is not None:
                handle.remove()
            state["site"], state["src"], state["norm_ref"] = None, None, None
            torch.cuda.empty_cache()
        s = np.array(scores)
        return s, cap.get("h")

    t0 = time.time()
    # ---- 1. natural pass + capture real L* activations --------------------
    print("  [1/4] natural forward pass (+ capture L*) ...")
    nat, H_true = run(prompts, capture="Lstar")
    assert H_true is not None and H_true.shape == (len(prompts), H)
    judge_auc = float(roc_auc_score(yte, nat))
    print(f"        judge AUC = {judge_auc:.4f}  ({time.time()-t0:.0f}s)")

    # ROW-ALIGNMENT CHECK: the freshly captured L* activations must match the
    # cached ones row by row. Comparing labels alone would NOT catch a
    # "same label, different text" misalignment (TODO P1-15); this does.
    Ht = H_true.float().cpu().numpy()
    cache_mae = float(np.abs(Ht - Rte[:, L_star, :]).mean())
    cache_scale = float(np.abs(Rte[:, L_star, :]).mean())
    cache_rel = cache_mae / max(cache_scale, 1e-9)
    print(f"        row-alignment: mean|h_live - h_cache| = {cache_mae:.5f} "
          f"(cache mean|h| = {cache_scale:.3f}, rel {cache_rel:.5f})")
    assert cache_rel < 0.02, (
        f"captured activations do not match the cache (rel MAE {cache_rel:.4f}) -- "
        "the prompt list and the residual cache are not the same rows")

    # cross payload: the PARTNER's real activation, cast to the model dtype
    H_cross = H_true[torch.tensor(partner, device=H_true.device)].contiguous()
    # shuffled control: permutation of the real activations (same distribution)
    perm = torch.tensor(rng.permutation(len(prompts)), device=H_true.device)
    H_shuf = H_true[perm].contiguous()
    # random-direction control matched in norm to the true activations
    R = torch.randn_like(H_true)
    R = R / R.norm(dim=-1, keepdim=True) * H_true.norm(dim=-1, keepdim=True)

    report = {
        "tag": args.tag, "model": args.model, "layer": L_star, "layer_selection": "val",
        "n_test": int(len(yte)), "n_test_relevant": int(yte.sum()),
        "n_layers": int(n_layers), "hidden": int(H),
        "probe_test_auc_at_Lstar": round(probe_test_auc, 4),
        "judge_auc_natural": round(judge_auc, 4),
        "n_cross_label_differ": n_diff,
        "row_alignment_rel_mae": round(cache_rel, 6),
        "alphas_cross": cross_alphas, "alphas_final": final_alphas,
        "protocol": {
            "site": "hook on decoder-layer output (residual stream coordinate)",
            "norm_match": "payload rescaled to the TARGET row's own L2 norm",
            "partner": "random test row, preferentially a different label",
            "note": "residual cache == layers[i].output[0,-1,:]; same coordinate as hook",
        },
        "sweep": {}, "edges": {},
    }

    print("  [2/4] cross-sample patch at L* (does the verdict follow the payload?) ...")
    y_src = yte[partner]
    cross_scores = {0.0: nat}
    for a in cross_alphas:
        if a == 0.0:
            s = nat
        else:
            s, _ = run(prompts, site="Lstar", alpha=a, src=H_cross)
            cross_scores[a] = s
        m1 = float(s[y_src == 1].mean()) if (y_src == 1).any() else float("nan")
        m0 = float(s[y_src == 0].mean()) if (y_src == 0).any() else float("nan")
        report["sweep"][f"cross_Lstar@{a}"] = {
            "auc_vs_target": round(float(roc_auc_score(yte, s)), 4),
            "auc_vs_source": round(float(roc_auc_score(y_src, s)), 4),
            "follow_gap_vs_source": round(m1 - m0, 4),
            "mean_pyes": round(float(s.mean()), 4),
            "mean_pyes_label1": round(float(s[yte == 1].mean()), 4),
            "mean_pyes_label0": round(float(s[yte == 0].mean()), 4),
            "mean_pyes_srclabel1": round(m1, 4),
            "mean_pyes_srclabel0": round(m0, 4),
        }
        print(f"        a={a:4.2f} auc_target={report['sweep'][f'cross_Lstar@{a}']['auc_vs_target']:.4f} "
              f"auc_source={report['sweep'][f'cross_Lstar@{a}']['auc_vs_source']:.4f} "
              f"follow_gap={report['sweep'][f'cross_Lstar@{a}']['follow_gap_vs_source']:+.4f}")

    # identity / controls at L*
    print("  [3/4] controls at L* ...")
    self_patch, _ = run(prompts, site="Lstar", alpha=1.0, src=H_true)
    shuf_patch, _ = run(prompts, site="Lstar", alpha=1.0, src=H_shuf)
    rand_patch, _ = run(prompts, site="Lstar", alpha=1.0, src=R)
    report["controls_Lstar"] = {
        "self_patch_auc": round(float(roc_auc_score(yte, self_patch)), 4),
        "self_patch_max_absdiff_vs_natural": round(float(np.abs(self_patch - nat).max()), 6),
        "shuffled_payload_auc": round(float(roc_auc_score(yte, shuf_patch)), 4),
        "random_payload_auc": round(float(roc_auc_score(yte, rand_patch)), 4),
    }
    print(f"        self   AUC={report['controls_Lstar']['self_patch_auc']:.4f} "
          f"(max|Δ| vs natural {report['controls_Lstar']['self_patch_max_absdiff_vs_natural']:.2e})")
    print(f"        shuf   AUC={report['controls_Lstar']['shuffled_payload_auc']:.4f}")
    print(f"        random AUC={report['controls_Lstar']['random_payload_auc']:.4f}")

    print("  [4/4] cross patch at the FINAL layer, with and without norm matching ...")
    for a in final_alphas:
        if a == 0.0:
            report["sweep"][f"final_normmatched@{a}"] = {"auc_vs_target": round(judge_auc, 4),
                                                         "auc_vs_source": round(float(roc_auc_score(y_src, nat)), 4)}
            report["sweep"][f"final_raw@{a}"] = dict(report["sweep"][f"final_normmatched@{a}"])
            continue
        s_nm, _ = run(prompts, site="final", alpha=a, src=H_cross, norm_ref="target")
        s_raw, _ = run(prompts, site="final", alpha=a, src=H_cross)
        report["sweep"][f"final_normmatched@{a}"] = {
            "auc_vs_target": round(float(roc_auc_score(yte, s_nm)), 4),
            "auc_vs_source": round(float(roc_auc_score(y_src, s_nm)), 4),
            "mean_pyes": round(float(s_nm.mean()), 4),
        }
        report["sweep"][f"final_raw@{a}"] = {
            "auc_vs_target": round(float(roc_auc_score(yte, s_raw)), 4),
            "auc_vs_source": round(float(roc_auc_score(y_src, s_raw)), 4),
            "mean_pyes": round(float(s_raw.mean()), 4),
        }
        print(f"        a={a:4.2f} norm-matched target={report['sweep'][f'final_normmatched@{a}']['auc_vs_target']:.4f} "
              f"| raw (v1) target={report['sweep'][f'final_raw@{a}']['auc_vs_target']:.4f}")

    # ---- paired bootstrap edges (query-clustered, shared resamples) --------
    print("  bootstrap edges ...")
    rng_b = np.random.default_rng(args.seed)
    sets = boot_indices(qte, args.boot, rng_b)
    base_nat = boot_auc(yte, nat, sets)
    base_src = boot_auc(y_src, nat, sets)
    for a in cross_alphas:
        if a == 0.0:
            continue
        key = f"cross_Lstar@{a}"
        s = cross_scores[a]
        d_t = boot_auc(yte, s, sets) - base_nat
        d_s = boot_auc(y_src, s, sets) - base_src
        report["edges"][key] = {
            "auc_source_minus_natural_on_srclabels": {
                "mean": round(float(d_s.mean()), 4),
                "ci95": [round(float(np.percentile(d_s, 2.5)), 4),
                         round(float(np.percentile(d_s, 97.5)), 4)],
                "p_le0": round(float((d_s <= 0).mean()), 5)},
            "auc_target_minus_judge": {
                "mean": round(float(d_t.mean()), 4),
                "ci95": [round(float(np.percentile(d_t, 2.5)), 4),
                         round(float(np.percentile(d_t, 97.5)), 4)],
                "p_le0": round(float((d_t <= 0).mean()), 5)},
        }
    report["env"] = {"torch": torch.__version__, "numpy": np.__version__,
                     "transformers": __import__("transformers").__version__}
    report["elapsed_sec"] = round(time.time() - t0, 1)

    def _finite(o):
        if isinstance(o, float):
            return o if np.isfinite(o) else None
        if isinstance(o, dict):
            return {k: _finite(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_finite(v) for v in o]
        return o

    for h in handles.values():
        h.remove()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(_finite(report), f, indent=2, allow_nan=False)
    print(f"wrote {args.out}  ({report['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
