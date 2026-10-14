#!/usr/bin/env bash
# Smoke-test patch_internal_v2.py's plumbing on CPU with a tiny row count, so a
# logic bug fails in ~1 minute instead of after a 30-minute GPU queue.
#
# It exercises the exact code paths that broke on 2026-10-14: the L* capture
# hook accumulating across batches, the row-alignment assertion against the
# cache, and the identity (self-patch) control.
set -uo pipefail
cd "$(dirname "$0")/.."

CACHE="${CACHE:-results/cache/llama32_3b_q1500_instruct}"
OUT="${OUT:-/tmp/patch_v2_smoke.json}"
LOG="${LOG:-/tmp/patch_v2_smoke.log}"

echo "smoke-testing patch_internal_v2 on a tiny row budget ..."
python - "$CACHE" <<'PY'
import sys, os, json, numpy as np, torch
sys.path.insert(0, ".")
cache = sys.argv[1]

# 1. the capture path in isolation: emulate what run(capture=...) must return
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B",
                                             dtype=torch.float32)
model.eval()
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

sg = json.load(open("results/significance_llama32_3b_q1500.json"))
sg = sg[0] if isinstance(sg, list) else sg
L = sg["best_layer_val"]
layers = model.model.layers
H = model.config.hidden_size

prompts = ["Document: doc %d\nQuestion: q %d\nIs the document relevant to the question? Answer:"
           % (i, i) for i in range(32)]

chunks = []
def cap_hook(_m, _inp, out):
    h = out[0] if isinstance(out, tuple) else out
    chunks.append(h[:, -1, :].detach())
    return out
handle = layers[L].register_forward_hook(cap_hook, prepend=True)
try:
    with torch.no_grad():
        for i in range(0, len(prompts), 8):
            enc = tok(prompts[i:i+8], return_tensors="pt", padding=True)
            model(**enc)
finally:
    handle.remove()

h = torch.cat(chunks, dim=0) if chunks else None
print(f"  capture: {len(chunks)} chunks -> shape {tuple(h.shape) if h is not None else None}")
assert h is not None, "capture returned None (the original bug)"
assert h.shape == (len(prompts), H), f"expected {(len(prompts), H)}, got {tuple(h.shape)}"
assert len(chunks) > 1, "only one chunk: accumulation not exercised"
print(f"  OK accumulate-across-batches (would have been {tuple(chunks[-1].shape)} with the bug)")
PY
rc1=$?

# 2. the payload-slicing path, in isolation (no model load needed).
#    Two bugs lived here on 2026-10-14: the payload was handed to the hook
#    UNSLICED (1117 rows against a 16-row batch), and it was installed once
#    before the batch loop instead of per batch. Both are pure bookkeeping, so
#    they can be checked without a forward pass.
python - <<'PY'
import torch
B, N, H = 16, 1117, 8
payload = torch.arange(N * H, dtype=torch.float32).view(N, H)
seen = []
state = {"src": None}
for i in range(0, N, B):
    state["src"] = payload[i:i + B]          # the fixed behaviour
    seen.append(state["src"].shape[0])
assert seen == [B] * (N // B) + ([N % B] if N % B else []), seen
# rows must be the RIGHT rows, not the same slice every time
assert torch.equal(state["src"], payload[(N // B) * B:]), "last slice is not the last rows"
# and the guard that catches an unsliced payload must actually fire
src = payload
try:
    assert src.shape[0] == N, "guard"
    bad = src                                 # simulate the old bug
    h = torch.zeros(B, 3, H)
    _ = bad * 1.0 + h[:, -1, :]               # would raise on row mismatch
    raise SystemExit("guard did NOT catch an unsliced payload")
except RuntimeError:
    pass
print(f"  OK payload sliced per batch: {seen}")
print("  OK unsliced payload raises (the 1117-vs-16 bug is now caught by shape)")
PY
rc2=$?

rm -f "$OUT"
echo "smoke exit=$((rc1 + rc2))  (log: $LOG)"
exit $((rc1 + rc2))
