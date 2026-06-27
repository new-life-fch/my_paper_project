#!/usr/bin/env python3
"""
Aggregate internals_vs_output_<tag>.json across models into one comparison
table — the paper's multi-model generalization result.

For each model reports the relevance-reader hierarchy:
  judge (output, 0-train)  <  final-layer probe (output repr)  <  best resid (internal)  <=  attn-head probe
and the key gaps:
  internal_edge = best_internal - final_layer_probe   (reading internals beyond training)
  train_edge    = final_layer_probe - judge            (training beyond just asking)

Usage: python scripts/aggregate_models.py
"""
import os, glob, json

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")

NAMES = {
    "llama32_3b": "LLaMA-3.2-3B",
    "llama31_8b": "LLaMA-3.1-8B",
    "mistral_7b": "Mistral-7B-v0.3",
    "qwen25_7b": "Qwen2.5-7B",
}


def load_tag(tag):
    # the main control json
    f = os.path.join(RES, f"internals_vs_output_{tag}.json")
    if tag == "llama32_3b" and not os.path.exists(f):
        f = os.path.join(RES, "internals_vs_output.json")  # the original 3B run
    if not os.path.exists(f):
        return None
    return json.load(open(f))


def main():
    rows = []
    for tag, name in NAMES.items():
        d = load_tag(tag)
        if d is None:
            continue
        judge = d["llm_judge_zeroshot"]
        final = d["resid_final_layer(output_repr)"]
        best = d["resid_best_layer"]["auc"]
        best_layer = d["resid_best_layer"]["layer"]
        attn = d["attn_head_probe(ours)"]
        n_layers = len(d["resid_per_layer_test_auc"])
        internal_best = max(best, attn)
        rows.append({
            "tag": tag, "model": name, "n_layers": n_layers,
            "judge": judge, "final": final,
            "best_resid": best, "best_layer": best_layer,
            "attn": attn,
            "peak_frac": round(best_layer / max(1, n_layers - 1), 2),
            "internal_edge": round(internal_best - final, 3),
            "train_edge": round(final - judge, 3),
            "holds": internal_best > final and internal_best > judge,
        })

    if not rows:
        print("no results yet")
        return

    hdr = (f"{'model':<18}{'L':>4}{'judge':>8}{'final':>8}{'bestL':>7}"
           f"{'best':>7}{'attn':>7}{'pkfrac':>8}{'intEdge':>9}{'trEdge':>8}{'holds':>7}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['model']:<18}{r['n_layers']:>4}{r['judge']:>8.3f}{r['final']:>8.3f}"
              f"{('L'+str(r['best_layer'])):>7}{r['best_resid']:>7.3f}{r['attn']:>7.3f}"
              f"{r['peak_frac']:>8.2f}{r['internal_edge']:>+9.3f}{r['train_edge']:>+8.3f}"
              f"{('YES' if r['holds'] else 'NO'):>7}")

    out = os.path.join(RES, "multimodel_summary.json")
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")
    n_hold = sum(r["holds"] for r in rows)
    print(f"internals>output holds on {n_hold}/{len(rows)} models.")
    print("intEdge = best_internal - final-layer probe (the depth/internal effect)")
    print("trEdge  = final-layer probe - judge (the training effect)")


if __name__ == "__main__":
    main()
