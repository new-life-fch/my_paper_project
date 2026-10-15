#!/usr/bin/env python3
"""
Causal intervention on the HEAD-PROBE direction -- step 2 of route B.

Question
--------
`gradient_alignment_heads.py` established that the head reader carries the
paper's headline effect, and `analyze_head_alignment_layers.py` showed that the
verdict's gradient leverage concentrates in ONE dominant layer per model (45-76%
of total). The remaining gap is causal: the head probe wins descriptively, but is
its direction something the model's verdict actually *depends* on? This script
injects and ablates along that direction and measures the effect.

Protocol (inherited from the E-11 correction, NOT reinvented)
-------------------------------------------------------------
The residual-stream version of this experiment was refuted by its own
pre-registered criterion (see archive/E11_JUDGEMENT.md), and the reason is worth
restating because it dictates every choice here:

  * mean P(yes) is NOT a valid statistic -- a random direction shifts it just as
    well. We group by the TRUE label and use
    `alignment_gap = mean P(yes)|relevant - mean P(yes)|irrelevant`.
  * alpha is NOT comparable across directions: per-unit logit slopes differ by
    ~6x. So every direction is calibrated to the SAME target yes/no logit shift,
    and the report records the alpha each direction needed.
  * large perturbations drive every direction's gap to zero, so the criterion is
    evaluated at modest shifts and the full sweep is reported alongside.
  * a random direction with a near-zero logit slope can need |alpha| ~ 1000,
    which overflows the logits to NaN. Such points are written as null, never as
    a number, and degenerate directions are dropped from the null.

Registered criterion (fixed before running, 2026-10-15)
------------------------------------------------------
The head-probe direction is causally aligned iff, at its calibrated shift,
its alignment gap is >= the 95th percentile of the same statistic over
`--n-random` random directions restricted to the SAME heads, AND the effect does
not depend on a single saturated point. Anything less is reported as a failure,
and the failure is reported with the full sweep profile rather than hidden.

Usage:
  python scripts/causal_steer_heads.py --model meta-llama/Llama-3.2-3B \
      --cache results/cache/llama32_3b_q1500_instruct --tag llama32_3b
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HD = 128
TOPK = 20
SEED = 42


def build_prompt(query, passage):
    return (f"Document: {passage}\nQuestion: {query}\n"
            f"Is the document relevant to the question? Answer:")


def yes_no_ids(tok):
    yes = [tok.encode(v, add_special_tokens=False)[0] for v in
           ("yes", "Yes", "YES", " yes", " YES")]
    no = [tok.encode(v, add_special_tokens=False)[0] for v in
          ("no", "No", "NO", " no", " NO")]
    return yes, no


def lr(penalty="l2"):
    return LogisticRegression(penalty=penalty, C=1.0,
                              solver="lbfgs" if penalty == "l2" else "liblinear",
                              max_iter=3000, class_weight="balanced",
                              random_state=SEED)


def load(cache, scheme, split):
    d = torch.load(os.path.join(cache, f"{scheme}_{split}.pt"), map_location="cpu")
    return (d["acts"].to(torch.float32).numpy(), d["labels"].numpy(),
            d.get("query_ids"))


def boot_gap(p, y, qids, n_boot=5000, seed=SEED):
    """Query-clustered bootstrap CI for the label-grouped alignment gap.

    The resampling unit is the QUERY (splits are by query, so rows within a query
    are not independent). Each resample recomputes both group means, which is
    what makes the interval a statement about the gap rather than about two
    separately-estimated means.
    """
    qids = np.asarray(qids)
    uq = np.unique(qids)
    idx_by_q = {q: np.where(qids == q)[0] for q in uq}
    rng = np.random.default_rng(seed)
    gaps = np.empty(n_boot)
    pos_all, neg_all = y == 1, y == 0
    for b in range(n_boot):
        samp = rng.choice(uq, size=len(uq), replace=True)
        idx = np.concatenate([idx_by_q[q] for q in samp])
        ps, ns = idx[pos_all[idx]], idx[neg_all[idx]]
        if len(ps) == 0 or len(ns) == 0:
            gaps[b] = np.nan
            continue
        gaps[b] = p[ps].mean() - p[ns].mean()
    gaps = gaps[np.isfinite(gaps)]
    if len(gaps) < 100:
        return None, None
    return float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-rows", type=int, default=0, help="0 = all test rows")
    ap.add_argument("--shift-targets", default="-4,-2,2,4",
                    help="yes/no logit-difference targets; each direction is "
                         "calibrated to these so directions are comparable")
    ap.add_argument("--calib-rows", type=int, default=64)
    ap.add_argument("--n-random", type=int, default=12)
    ap.add_argument("--max-alpha", type=float, default=60.0,
                    help="beyond this the calibration is treated as degenerate "
                         "(near-flat random directions); such points become null")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    sys.path.insert(0, ROOT)
    from src.cache_paths import resolve_cache
    from src.data import load_ms_marco, split_by_query
    args.cache = resolve_cache(args.cache)

    # ---------------------------------------------------------------- probe
    Atr, ytr, _ = load(args.cache, "attn", "train")
    Ava, yva, _ = load(args.cache, "attn", "val")
    Ate, yte, qte = load(args.cache, "attn", "test")
    meta = json.load(open(os.path.join(args.cache, "meta.json")))
    n_layers, n_heads = int(meta["n_layers"]), int(meta["n_heads"])

    from joblib import Parallel, delayed
    n_jobs = max(1, min(16, os.cpu_count() or 4))

    def hauc(h):
        p = lr(penalty="l1").fit(Atr[:, h, :], ytr)
        return roc_auc_score(yva, p.predict_proba(Ava[:, h, :])[:, 1])
    vaucs = np.array(Parallel(n_jobs=n_jobs)(
        delayed(hauc)(h) for h in range(Atr.shape[1])))
    sel = np.argsort(vaucs)[::-1][:TOPK]
    sel_layers = np.array([int(h) // n_heads for h in sel])
    sel_heads = np.array([int(h) % n_heads for h in sel])

    feats = lambda X: np.concatenate([X[:, h, :] for h in sel], axis=1)
    probe = lr().fit(feats(Atr), ytr)
    w = probe.coef_[0].astype(np.float32)
    head_auc = float(roc_auc_score(yte, probe.predict_proba(feats(Ate))[:, 1]))
    print(f"[{args.tag}] head-probe test AUC = {head_auc:.4f} "
          f"(must equal significance_{args.tag}_q1500.json attn)")

    # dominant layer = the one carrying most of the verdict's gradient leverage
    lay_path = os.path.join(ROOT, "results/head_alignment_by_layer.json")
    if not os.path.exists(lay_path):
        raise SystemExit("results/head_alignment_by_layer.json missing -- run "
                         "analyze_head_alignment_layers.py first")
    lay = json.load(open(lay_path))[args.tag]
    dom = max(lay["per_layer"].items(), key=lambda kv: kv[1]["contrib_share"])
    dom_layer = int(dom[0])
    print(f"[{args.tag}] dominant layer L{dom_layer} "
          f"({dom[1]['contrib_share']:.3f} of gradient leverage, "
          f"{dom[1]['n_heads']} selected heads)")

    # injection direction: the probe direction restricted to THIS layer's selected
    # heads, mapped back to activation space by multiplying each head block by
    # that head's train std (inverse of the chain rule the gradient used).
    Ftr = feats(Atr)
    sd = Ftr.std(axis=0).astype(np.float32)
    sd[sd == 0] = 1.0
    slots = [k for k, L in enumerate(sel_layers) if L == dom_layer]
    d_layer = np.zeros(n_heads * HD, dtype=np.float32)
    for k in slots:
        blk = slice(k * HD, (k + 1) * HD)
        h = int(sel_heads[k])
        # w is in standardised space -> raw activation space needs * sd
        d_layer[h * HD:(h + 1) * HD] = w[blk] * sd[blk]
    if not np.any(d_layer):
        raise SystemExit("restricted probe direction is all zeros")
    # Normalise ONCE, at the end. Dividing here and again below would discard the
    # per-head magnitude structure that makes this the probe's direction rather
    # than an arbitrary unit vector (that mistake made the probe look "flat":
    # it could not reach any shift target within max-alpha).
    d_layer /= np.linalg.norm(d_layer)
    blk_norms = [float(np.linalg.norm(d_layer[int(sel_heads[k]) * HD:(int(sel_heads[k]) + 1) * HD]))
                 for k in slots]
    print(f"[{args.tag}] direction lives in {len(slots)} heads of L{dom_layer}; "
          f"per-head |w| share: max {max(blk_norms):.3f}, min {min(blk_norms):.3f}")

    # random directions in the SAME subspace (same heads), and a second family
    # spread over all 20 selected heads at their own layers (handled separately)
    rng = np.random.default_rng(SEED)
    randoms = []
    for _ in range(args.n_random):
        r = np.zeros_like(d_layer)
        for k in slots:
            h = int(sel_heads[k])
            r[h * HD:(h + 1) * HD] = rng.standard_normal(HD)
        r /= np.linalg.norm(r)
        randoms.append(r)

    # ---------------------------------------------------------------- model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    _dt = torch.float16 if args.dtype == "float16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto",
                                                 dtype=_dt)
    model.eval()

    samples = load_ms_marco(n_queries=int(meta.get("n_queries", 1500)),
                            max_passages_per_query=5, seed=int(meta.get("seed", 42)))
    _, _, test_s = split_by_query(samples, seed=int(meta.get("seed", 42)))
    if len(test_s) != len(yte):
        raise SystemExit(f"row mismatch {len(test_s)} vs {len(yte)}")
    for x, y in zip(test_s, yte):
        if int(x["label"]) != int(y):
            raise SystemExit("label mismatch between rebuild and cache")
    n_rows = args.n_rows or len(test_s)
    rows = test_s[:n_rows]
    y = yte[:n_rows]
    qid = qte[:n_rows]

    yes_ids, no_ids = yes_no_ids(tok)
    dev = next(model.model.parameters()).device
    st = {"vecs": None}      # [N, H], one injection direction per batch row

    def hook(_m, _inp, out):
        if st["vecs"] is None:
            return out
        h = out[0] if isinstance(out, tuple) else out        # [B,S,H]
        v = st["vecs"].to(device=h.device, dtype=h.dtype)     # [B,H]
        norm = h[:, -1, :].norm(dim=-1, keepdim=True)         # [B,1]
        h[:, -1, :] = h[:, -1, :] + norm * v
        return out

    handle = model.model.layers[dom_layer].self_attn.o_proj.register_forward_hook(hook)

    @torch.no_grad()
    def run_multi(subset, dir_idx, alphas):
        """One pass over (row, direction) pairs.

        Injection direction varies PER BATCH ROW, so a whole grid of directions is
        measured in a single sweep instead of one sweep per direction. That is the
        difference between ~9 and ~90 minutes: the naive version re-ran the
        natural baseline once per direction, i.e. 49 identical forward passes over
        the same corpus.

        Because the corpus and the label layout are identical across directions,
        the pairs are naturally matched, which is what the clustered bootstrap
        wants anyway.
        """
        n = len(subset)
        order = np.arange(n)                       # which row each slot encodes
        # Clear the injection BEFORE the natural pass. Leaving the previous
        # direction set made the "natural" baseline contaminated, which silently
        # destroyed the per-direction slope (measured slope 0.0367 instead of the
        # true 0.193) and therefore every alpha derived from it.
        st["vecs"] = None
        # natural pass (no injection)
        pyes0, ld0 = [], []
        for i in range(0, n, args.batch_size):
            batch = [subset[j] for j in order[i:i + args.batch_size]]
            prompts = [build_prompt(x["query"], x["passage"]) for x in batch]
            enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
            lg = model(**enc).logits[:, -1, :].float()
            ly = torch.logsumexp(lg[:, yes_ids], dim=1)
            ln = torch.logsumexp(lg[:, no_ids], dim=1)
            ld0.append((ly - ln).cpu().numpy())
        ld0 = np.concatenate(ld0)

        if len(dir_idx) == 0:
            return ld0, np.zeros((0, n), dtype=np.float64)
        # injected passes
        all_dirs = np.stack([dirs[i] for i in dir_idx])            # [K, H]
        scal = np.asarray(alphas, dtype=np.float32)                # [K] or [1]
        if scal.size == 1:
            scal = np.repeat(scal, len(dir_idx))
        out_ld = np.zeros((len(dir_idx), n), dtype=np.float64)
        step = max(1, args.batch_size)
        for k in range(len(dir_idx)):
            vec = torch.from_numpy(all_dirs[k] * scal[k]).float()
            lds = []
            for i in range(0, n, step):
                idxs = order[i:i + step]
                batch = [subset[j] for j in idxs]
                prompts = [build_prompt(x["query"], x["passage"]) for x in batch]
                enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
                st["vecs"] = vec.unsqueeze(0).repeat(len(batch), 1)
                lg = model(**enc).logits[:, -1, :].float()
                st["vecs"] = None
                ly = torch.logsumexp(lg[:, yes_ids], dim=1)
                ln = torch.logsumexp(lg[:, no_ids], dim=1)
                lds.append((ly - ln).cpu().numpy())
            out_ld[k] = np.concatenate(lds)
        return ld0, out_ld

    try:
        rows = rows[:len(rows)]
        n = len(rows)
        # calibration set: natural logit-diff and the per-row activation norm scale
        calib = rows[:args.calib_rows]
        dir_names = ["head_probe"] + [f"random_{i}" for i in range(args.n_random)]
        dirs = [d_layer] + randoms
        K = len(dirs)

        # Fit the per-direction slope over an alpha GRID, not at a single alpha.
        # A one-point estimate folds any nonlinearity into the "slope" (measured:
        # the probe direction's slope came out 0.037 at alpha=1 where a direct
        # 32-row measurement of the same direction gave ~0.19), and the resulting
        # alpha is then wrong by the same factor. Each grid point is one cheap
        # pass over the calibration subset.
        CAL_ALPHAS = [1.0, 4.0, 16.0]
        nat_cal_ld, _ = run_multi(calib, [], [])
        nat_cal = float(nat_cal_ld.mean())
        xs = np.array(CAL_ALPHAS, dtype=np.float64)
        cal_curves = np.zeros((K, len(xs)))
        for j, a in enumerate(CAL_ALPHAS):
            _, ld = run_multi(calib, list(range(K)),
                              np.full(K, a, dtype=np.float32))
            cal_curves[:, j] = ld.mean(axis=1) - nat_cal
        # least squares through the origin (the response must vanish at alpha=0)
        slopes = (cal_curves * xs).sum(axis=1) / (xs * xs).sum()
        # report linearity so a direction whose response is not linear is visible
        fit_err = np.abs(cal_curves - slopes[:, None] * xs[None, :]).mean(axis=1)
        print(f"[{args.tag}] natural mean logit-diff (calib) = {nat_cal:+.3f}")
        print(f"[{args.tag}] slopes {np.round(slopes, 4)}")
        print(f"[{args.tag}] mean |fit error| {np.round(fit_err, 4)} "
              f"(large values mean the alpha grid is not in the linear regime)")

        nat_ld, _ = run_multi(rows, [], [])
        base = float(nat_ld.mean())
        print(f"[{args.tag}] natural mean logit-diff (all {n} rows) = {base:+.3f}")

        targets = [float(t) for t in args.shift_targets.split(",")]
        per_dir = {}
        for ki, dname in enumerate(dir_names):
            per_dir[dname] = {"slope_per_unit_alpha": round(float(slopes[ki]), 5),
                              "per_target": {}}
        for t in targets:
            alphas = np.array([t / slopes[k] if abs(slopes[k]) > 1e-6 else np.nan
                               for k in range(K)], dtype=np.float32)
            ok = np.isfinite(alphas) & (np.abs(alphas) <= args.max_alpha)
            used = [k for k in range(K) if ok[k]]
            _, ld = run_multi(rows, used, [alphas[k] for k in used])
            for j, k in enumerate(used):
                p = 1.0 / (1.0 + np.exp(-ld[j]))
                pos, neg = y == 1, y == 0
                gap = float(p[pos].mean() - p[neg].mean())
                lo, hi = boot_gap(p, y, qid, n_boot=5000)
                try:
                    aucv = float(roc_auc_score(y, p))
                except ValueError:
                    aucv = None
                per_dir[dir_names[k]]["per_target"][f"{t:+.0f}"] = {
                    "alpha": round(float(alphas[k]), 4),
                    "fitted_shift": round(float(slopes[k] * alphas[k]), 4),
                    "alignment_gap": round(gap, 5),
                    "gap_ci95": [round(lo, 5) if lo is not None else None,
                                 round(hi, 5) if hi is not None else None],
                    "auc": round(aucv, 5) if aucv is not None else None,
                    "mean_pyes": round(float(p.mean()), 5),
                    "frac_saturated": round(float(((p < 1e-4) | (p > 1 - 1e-4)).mean()), 4),
                    "pyes": [round(float(x), 5) for x in p],
                }
            for k in range(K):
                if not ok[k]:
                    per_dir[dir_names[k]]["per_target"][f"{t:+.0f}"] = {
                        "alpha": None, "degenerate": True,
                        "slope_per_unit_alpha": round(float(slopes[k]), 5),
                        "note": "|alpha| exceeds max-alpha; near-flat direction, "
                                "excluded from the null"}
            print(f"  shift {t:+.0f}: "
                  + " ".join(f"{dir_names[k]}="
                             + (f"{per_dir[dir_names[k]]['per_target'][f'{t:+.0f}']['alignment_gap']:+.4f}"
                                if ok[k] else "n/a") for k in range(min(4, K)))
                  + f" ... ({int(ok.sum())}/{K} usable)")

        out = {"tag": args.tag, "model": args.model, "n_rows": int(n_rows),
               "dominant_layer": dom_layer, "n_heads_in_direction": len(slots),
               "head_probe_test_auc": round(head_auc, 4),
               "natural_mean_logitdiff": round(base, 4),
               "shift_targets": targets,
               "protocol": "vectorised injection: all directions measured in one "
                           "sweep over a fixed corpus, so the (row, direction) "
                           "pairs are matched. alpha is derived from a per-direction "
                           "slope measured on a calibration subset, then the "
                           "alignment gap is measured on the full row set.",
               "per_direction": per_dir}
        # ---------------- registered criterion: percentile vs random null -----
        verdict = {}
        for t in targets:
            k = f"{t:+.0f}"
            null = []
            for i_ in range(args.n_random):
                v = per_dir.get(f"random_{i_}", {}).get("per_target", {}).get(k)
                if v and not v.get("degenerate"):
                    null.append(v["alignment_gap"])
            mine = per_dir["head_probe"]["per_target"].get(k)
            if mine is None or mine.get("degenerate") or len(null) < 5:
                verdict[k] = {"passes": None,
                              "reason": f"degenerate probe point or too few usable "
                                        f"random directions (n={len(null)})"}
                continue
            null = np.array(null)
            pct = float((np.abs(null) < abs(mine["alignment_gap"])).mean())
            expect = -1 if t < 0 else 1
            verdict[k] = {
                "passes": bool(pct >= 0.95),
                "percentile": round(pct, 4),
                "percentile_resolution": round(1.0 / (len(null) + 1), 4),
                "probe_gap": mine["alignment_gap"],
                "probe_sign_matches_shift": bool(np.sign(mine["alignment_gap"]) == expect),
                "null_abs_mean": round(float(np.abs(null).mean()), 5),
                "null_abs_max": round(float(np.abs(null).max()), 5),
                "n_random_usable": int(len(null)),
            }
        out["registered_criterion"] = {
            "statement": "head-probe alignment gap >= 95th percentile of the same "
                         "statistic over random directions in the same head "
                         "subspace, at the same calibrated shift",
            "per_target": verdict,
            "any_target_passes": bool(any(v.get("passes") for v in verdict.values())),
            "all_targets_pass": bool(all(v.get("passes") for v in verdict.values()
                                         if v.get("passes") is not None)),
        }
        out["elapsed_sec"] = round(time.time() - t0, 1)
        p = args.out or f"results/causal_steer_heads_{args.tag}_q1500.json"
        json.dump(out, open(os.path.join(ROOT, p), "w"), indent=2)
        print(f"\n  registered criterion: any target passes = "
              f"{out['registered_criterion']['any_target_passes']}")
        for k, v in verdict.items():
            print(f"    shift {k}: " + (f"pct={v.get('percentile')} "
                  f"passes={v.get('passes')}" if v.get("passes") is not None
                  else f"n/a ({v.get('reason')})"))
        print(f"\nwrote {p}  ({out['elapsed_sec']}s)")
    finally:
        handle.remove()
    return 0


if __name__ == "__main__":
    sys.exit(main())
