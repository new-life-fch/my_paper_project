#!/usr/bin/env python3
"""
Single source of truth for every number that goes into the paper.

Why this exists
---------------
The repo's worst recurring failure has been documentation drifting away from the
artifacts: the old Table 2 reproduced a bootstrap generation that no longer
existed anywhere, Table 1 mixed two sample sizes, and a figure annotated a
test-argmax peak while the text quoted a val-selected one. Each time, the cause
was the same -- numbers were copied by hand from a log into LaTeX, and nothing
connected the two.

This script walks results/*.json and emits ONE consolidated, provenance-carrying
table. Rule (TODO.md「证据纪律」): if a number is not in this file, it does not
go into the paper.

Inputs (all optional; missing pieces are reported, not invented):
  results/internals_vs_output_{tag}_q{N}.json     -- four-reader AUCs (val-selected)
  results/significance_{tag}_q{N}.json            -- per-model edges (paired bootstrap)
  results/logit_lens_{tag}_q{N}.json              -- zero-training depth control
  results/capacity_control_{tag}_q{N}.json        -- readout-flexibility control
  results/patch_internal_{tag}_q{N}.json          -- attenuation vs misalignment
  results/causal_steer_{tag}_q{N}.json            -- steering + alignment gap
  results/figures/{tag}_layer_signal.json         -- depth curves / per-head grid

Output:
  results/PAPER_NUMBERS.json   -- every value + which file it came from
  plus a human-readable table on stdout

Usage:
  python scripts/collect_paper_numbers.py --n 1500
"""
import argparse
import glob
import json
import os
import sys

MODELS = [
    ("llama32_3b", "LLaMA-3.2-3B"),
    ("llama32_3b_instruct", "LLaMA-3.2-3B-Instruct"),
    ("llama31_8b", "LLaMA-3.1-8B"),
    ("llama31_8b_instruct", "LLaMA-3.1-8B-Instruct"),
    ("mistral_7b", "Mistral-7B-v0.3"),
    ("mistral_7b_instruct", "Mistral-7B-Instruct-v0.3"),
    ("qwen25_7b", "Qwen2.5-7B"),
    ("qwen25_7b_instruct", "Qwen2.5-7B-Instruct"),
]

# Second in-domain dataset (TODO E-12). The tag already encodes the negative
# construction ("bm25" = BM25 hard negatives), so it must NOT be looked up
# with the plain model tag.
EXTRA_DATASETS = [
    ("llama32_3b_fiqa_bm25",
     "LLaMA-3.2-3B / FiQA (BM25 hard negatives)"),
]


def load(path):
    if not os.path.exists(path):
        return None
    try:
        return json.load(open(path))
    except Exception as e:
        print(f"  WARN unreadable {path}: {e}", file=sys.stderr)
        return None


def first(d):
    return d[0] if isinstance(d, list) and d else d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500, help="query count suffix")
    ap.add_argument("--out", default="results/PAPER_NUMBERS.json")
    args = ap.parse_args()
    N = args.n

    rep = {"n_queries": N, "models": {}, "missing": [], "provenance": {}}

    for tag, name in MODELS:
        row = {"display": name}
        src = {}
        # ---- four readers (val-selected) ----
        p = f"results/internals_vs_output_{tag}_q{N}.json"
        d = load(p)
        if d:
            src["readers"] = p
            bl = d.get("resid_best_layer", {})
            row["readers"] = {
                "judge": d.get("llm_judge_zeroshot"),
                "final": d.get("resid_final_layer(output_repr)"),
                "best_resid": bl.get("auc"),
                "best_resid_layer": bl.get("layer"),
                "best_resid_selection": bl.get("selection"),
                "best_resid_val_auc": bl.get("val_auc_at_selected"),
                "attn_topk": d.get("attn_head_probe(ours)"),
            }
            # the old, biased field is kept only to quantify the bias
            bad = d.get("resid_best_layer_test_argmax_DO_NOT_REPORT")
            if bad:
                row["readers"]["best_resid_test_argmax_DO_NOT_REPORT"] = bad
        else:
            rep["missing"].append(p)

        # ---- significance edges ----
        p = f"results/significance_{tag}_q{N}.json"
        d = load(p)
        if d:
            r = first(d)
            src["significance"] = p
            if all(k in r for k in ("n_test", "n_boot", "n_layers")):
                row["significance"] = {
                    "n_test": r["n_test"], "n_boot": r["n_boot"],
                    "best_layer_val": r.get("best_layer_val"),
                    "auc": {k: v for k, v in r.get("auc", {}).items()},
                    "edges": r.get("edges", {}),
                }
            else:
                rep["missing"].append(p + " (legacy format, ignored)")
        else:
            rep["missing"].append(p)

        # ---- logit lens (zero training) ----
        p = f"results/logit_lens_{tag}_q{N}.json"
        d = load(p)
        if d and d.get("selection") == "val":
            src["logit_lens"] = p
            row["logit_lens"] = {
                "peak_layer": d["logit_lens_best_layer"]["layer"],
                "peak_auc": d["logit_lens_best_layer"]["auc"],
                "peak_val_auc": d["logit_lens_best_layer"].get("val_auc_at_selected"),
                "final_layer_auc": d.get("logit_lens_final_layer_auc"),
                "judge_auc": d.get("llm_judge_auc"),
                "sanity_absdiff": d.get("sanity_lastlayer_vs_judge_absdiff"),
            }
        elif d:
            rep["missing"].append(p + " (legacy: selection-on-test)")
        else:
            rep["missing"].append(p)

        # ---- capacity control (optional) ----
        p = f"results/capacity_control_{tag}_q{N}.json"
        d = load(p)
        if d:
            src["capacity_control"] = p
            row["capacity_control"] = {
                "feature_counts": d.get("feature_counts"),
                "readers": {k: v["auc"] for k, v in d.get("readers", {}).items()},
                "edges": {k: v for k, v in d.get("edges", {}).items()},
                "chunk_perm_summary": d.get("final_select20_chunks_perm_summary"),
                "attn_topk_sweep_auc": d.get("attn_topk_sweep_auc"),
            }

        # ---- patch experiment (optional) ----
        p = f"results/patch_internal_{tag}_q{N}.json"
        d = load(p)
        if d:
            src["patch_internal"] = p
            row["patch_internal"] = {
                "layer": d.get("layer"),
                "probe_test_auc_at_layer": d.get("probe_test_auc_at_layer"),
                "judge_natural": d.get("judge_natural"),
                "sweep": d.get("sweep"),
                "edges": d.get("edges"),
            }

        # ---- causal steering (optional) ----
        p = f"results/causal_steer_{tag}_q{N}.json"
        d = load(p)
        if d and ("random_all" in d.get("sweep", {}) or "gap_at_alpha-8" in d):
            src["causal_steer"] = p
            row["causal_steer"] = {
                "layer": d.get("best_internal_layer"),
                "layer_selection": d.get("best_layer_selection"),
                "proj_auc_internal": d.get("proj_auc_internal"),
                "proj_auc_final": d.get("proj_auc_final"),
                "n_random": d.get("n_random"),
                "protocol": d.get("protocol"),
                "gap_summary": {k: v for k, v in d.items() if k.startswith("gap_at_alpha")},
                "auc_summary": {k: v for k, v in d.items() if k.startswith("auc_at_alpha")},
            }
        elif d:
            rep["missing"].append(p + " (legacy: no label-grouped gap)")

        # ---- depth figure data ----
        p = f"results/figures/{tag}_layer_signal.json"
        d = load(p)
        if d and d.get("resid_peak", {}).get("selection") == "val":
            src["layer_signal"] = p
            row["layer_signal"] = {
                "judge": d.get("judge"),
                "resid_peak": d.get("resid_peak"),
                "resid_final": d.get("resid_final"),
                "topk_head_layers": d.get("topk_head_layers"),
            }

        rep["models"][tag] = row
        rep["provenance"][tag] = src

    # ---- second in-domain dataset (FiQA, BM25 hard negatives) ----------------
    rep["datasets"] = {}
    for tag, name in EXTRA_DATASETS:
        row, src = {"display": name}, {}
        p = f"results/internals_vs_output_{tag}_q{N}.json"
        d = first(load(p))
        if d:
            src["readers"] = p
            bl = d.get("resid_best_layer", {})
            row["readers"] = {
                "judge": d.get("llm_judge_zeroshot"),
                "final": d.get("resid_final_layer(output_repr)"),
                "best_resid": bl.get("auc"),
                "best_resid_layer": bl.get("layer"),
                "best_resid_selection": bl.get("selection"),
                "attn_topk": d.get("attn_head_probe(ours)"),
            }
        else:
            rep["missing"].append(p)
        q = f"results/significance_{tag}_q{N}.json"
        s = first(load(q))
        if s:
            src["significance"] = q
            row["significance"] = {
                "auc": {k: v for k, v in (s.get("auc") or {}).items()},
                "edges": s.get("edges"),
                "best_layer_val": s.get("best_layer_val"),
                "n_test": s.get("n_test"),
                "n_boot": s.get("n_boot"),
            }
        else:
            rep["missing"].append(q)
        rep["datasets"][tag] = row
        rep["provenance"][tag] = src

    # ---- completeness report -------------------------------------------------
    have = [t for t, r in rep["models"].items() if "significance" in r and "readers" in r]
    rep["complete_models"] = have
    rep["n_complete"] = len(have)
    have_ds = [t for t, r in rep["datasets"].items()
               if "significance" in r and "readers" in r]
    rep["complete_datasets"] = have_ds
    rep["n_complete_datasets"] = len(have_ds)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(rep, open(args.out, "w"), indent=2)

    print(f"=== PAPER NUMBERS (n={N}) ===")
    print(f"{'model':26s} {'judge':>6} {'final':>6} {'resid':>6} {'attn':>6}  "
          f"{'attn-final':>10} {'p':>8}  {'peak L':>6}")
    print("-" * 92)
    for tag, name in MODELS:
        r = rep["models"][tag]
        rd = r.get("readers") or {}
        sg = r.get("significance") or {}
        e = (sg.get("edges") or {}).get("attn-final") or {}
        pk = (r.get("layer_signal") or {}).get("resid_peak") or {}
        print(f"{name:26s} {str(rd.get('judge')):>6} {str(rd.get('final')):>6} "
              f"{str(rd.get('best_resid')):>6} {str(rd.get('attn_topk')):>6}  "
              f"{e.get('mean', '-'):>10} {str(e.get('p_le0', '-')):>8}  "
              f"{str(pk.get('layer', '-')):>6}")
    print(f"\ncomplete: {rep['n_complete']}/8")
    if rep["missing"]:
        print(f"missing ({len(rep['missing'])}):")
        for m in rep["missing"]:
            print(f"  - {m}")
    print(f"\nSaved {args.out}  (every number carries its source file in .provenance)")


if __name__ == "__main__":
    main()
