# results/figures/ — provenance and recovery notes

## Why this file exists

The figures in this directory were **nearly lost**. `.gitignore` used to contain
`results/` and `*.png`, so a figure only survived a clone if it was explicitly
`git add -f`'d. Two paper figures had never been force-added, so they vanished
from every copy of the project — while the newer `qwen25_7b_layer_signal.png`
(force-added later) survived. `paper/main.tex` still referenced the two missing
files, so the paper could no longer compile.

They were recovered from `paper/main.pdf`, which had been compiled while the
files still existed: LaTeX embeds each `\includegraphics` as an image XObject
(RGB + grayscale `/SMask` alpha), so the original pixels were still inside the
PDF. See `scripts/extract_figures_from_pdf.py`.

**Fixed 2026-10-12**: `.gitignore` was narrowed to `results/cache/` +
`results/**/*.pt` + `results/logs/`, so figures and result JSONs are tracked by
default now. This cannot happen again by the same route.

## What is here

| File | Referenced by | Regenerable? |
|---|---|---|
| `llama32_3b_layer_signal.png` (3133×887) | `main.tex` Fig. 1 (`fig:depth`) | ❌ needs `.pt` caches |
| `causal_steer_3b.png` (1634×609) | `main.tex` Fig. 2 (`fig:causal`) | ✅ from `results/causal_steer_llama32_3b_q1500.json` |
| `qwen25_7b_layer_signal.png` (3133×887) | supplement (not in `main.tex`) | ❌ needs `.pt` caches |

## How to regenerate / re-recover

```bash
# 1. Recover from the compiled PDF (no GPU, no caches needed)
python scripts/extract_figures_from_pdf.py --pdf paper/main.pdf \
    --outdir results/figures            # add --dry-run to just list XObjects

# 2. Fig. 2 is fully reproducible from its result JSON
python scripts/plot_causal.py \
    --json results/causal_steer_llama32_3b_q1500.json \
    --out  results/figures/causal_steer_3b.png
```

`plot_layer_signal.py` (Fig. 1) **cannot** be re-run from this checkout: it needs
`results/cache/<tag>_instruct/{attn,resid}_{train,val,test}.pt` plus `meta.json`
(1–2 GB per model, served from a symlink on the GPU host, never committed).
The per-layer residual AUC array and the per-head heatmap underlying Fig. 1 exist
**only** in this PNG — `internals_vs_output_llama32_3b_q1500.json` did not survive.
Re-running it requires the GPU host and the activation caches.

## Also missing (not referenced by `main.tex`)

Figures mentioned in `REPORT.md` / `CLAUDE.md` but absent here:
`llama31_8b_layer_signal.png`, `llama31_8b_instruct_layer_signal.png`,
`llama32_3b_instruct_layer_signal.png`, `mistral_7b_layer_signal.png`,
`mistral_7b_instruct_layer_signal.png`, `causal_steer_8b_base.png`,
`causal_steer_8b_inst.png`. The paper still compiles because it only includes
Fig. 1 and Fig. 2.

## Guard against recurrence

Done on 2026-10-12 — `.gitignore` now ignores only `results/cache/`,
`results/**/*.pt` and `results/logs/`, so everything in this directory is
tracked without needing `git add -f`. See `docs/CACHE_NAMING.md` for how the
cache directories underneath are named, and `TODO.md` P2 for what is still
missing from the GPU host.
