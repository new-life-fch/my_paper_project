"""
Activation extraction module using nnsight.

Provides two extraction schemes:
  Scheme A (last_token):  Take the last token's per-head activation from o_proj output.
  Scheme B (pooling):     Mean-pool per-head activations over the passage token range.

For LLaMA-3.2-3B:
  - 28 layers, 24 heads/layer, head_dim=128
  - Total heads: 672
  - Per-head activation dim: 128
"""

import torch
from tqdm import tqdm


def _build_prompt(query: str, passage: str) -> str:
    """Build the (query, passage) prompt for forward pass."""
    return f"Q: {query}\nP: {passage}"


def extract_activations_last_token(model, samples, n_layers, n_heads, head_dim,
                                    max_length=512):
    """
    Scheme A: Extract per-head activations at the last token position.

    For each sample, runs a forward pass and extracts:
      o_proj.output[0].view(B, S, n_heads, head_dim)[0, -1, :, :]

    Args:
        model: nnsight LanguageModel instance.
        samples: List of {query, passage, label} dicts.
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads per layer.
        head_dim: Dimension of each attention head.
        max_length: Max token length for truncation.

    Returns:
        Tensor of shape [N, n_layers, n_heads, head_dim] on CPU.
    """
    n_total_heads = n_layers * n_heads
    all_activations = []

    for sample in tqdm(samples, desc="Extracting (last_token)"):
        prompt = _build_prompt(sample["query"], sample["passage"])

        with model.trace(prompt):
            head_acts_list = []
            for layer_idx in range(n_layers):
                # Access layers in forward-pass order (critical for nnsight)
                attn_out = model.model.layers[layer_idx].self_attn.o_proj.output[0]
                B, S, H = attn_out.shape

                # Reshape to [B, S, n_heads, head_dim] and extract last token
                # [0, -1] -> [n_heads, head_dim] for batch=0, last_position
                per_head_last = attn_out.view(B, S, n_heads, head_dim)[0, -1, :, :]
                head_acts_list.append(per_head_last.detach().cpu().save())

        # head_acts_list[i] shape: [n_heads, head_dim]
        # Stack all layers: [n_layers, n_heads, head_dim]
        acts = torch.stack(head_acts_list, dim=0)
        all_activations.append(acts)

    # [N, n_layers, n_heads, head_dim]
    return torch.stack(all_activations, dim=0).cpu()


def extract_activations_pooling(model, samples, n_layers, n_heads, head_dim,
                                 max_length=512):
    """
    Scheme B: Extract per-head activations via mean pooling over passage tokens.

    For each sample:
      1. Tokenize prompt to locate passage token boundaries.
      2. Run forward pass.
      3. Mean-pool per-head activations over [passage_start, passage_end].

    Args:
        model: nnsight LanguageModel instance.
        samples: List of {query, passage, label} dicts.
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads per layer.
        head_dim: Dimension of each attention head.
        max_length: Max token length for truncation.

    Returns:
        Tensor of shape [N, n_layers, n_heads, head_dim] on CPU.
    """
    tokenizer = model.tokenizer
    all_activations = []

    for sample in tqdm(samples, desc="Extracting (pooling)"):
        query = sample["query"]
        passage = sample["passage"]
        prompt = _build_prompt(query, passage)

        # Locate passage token boundaries
        # Tokenize the query part to find where passage starts
        prefix = f"Q: {query}\nP: "
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)
        passage_start = len(prefix_ids)

        full_ids = tokenizer.encode(prompt, add_special_tokens=True,
                                     truncation=True, max_length=max_length)
        passage_end = len(full_ids)

        # Safety: if passage tokens not found, fall back to last token
        if passage_start >= passage_end:
            passage_start = max(1, passage_end - 1)

        with model.trace(prompt):
            head_acts_list = []
            for layer_idx in range(n_layers):
                attn_out = model.model.layers[layer_idx].self_attn.o_proj.output[0]
                B, S, H = attn_out.shape

                # Reshape and mean-pool over passage token range
                per_head = attn_out.view(B, S, n_heads, head_dim)
                # [0, start:end, :, :] -> [passage_len, n_heads, head_dim]
                passage_acts = per_head[0, passage_start:passage_end, :, :]
                pooled = passage_acts.mean(dim=0)  # [n_heads, head_dim]
                head_acts_list.append(pooled.detach().cpu().save())

        acts = torch.stack(head_acts_list, dim=0)
        all_activations.append(acts)

    return torch.stack(all_activations, dim=0).cpu()


def reshape_for_probes(activations):
    """
    Reshape activations from [N, n_layers, n_heads, head_dim]
    to [N, n_total_heads, head_dim] for probe training.

    Args:
        activations: Tensor of shape [N, n_layers, n_heads, head_dim].

    Returns:
        Tensor of shape [N, n_total_heads, head_dim].
    """
    N, n_layers, n_heads, head_dim = activations.shape
    return activations.reshape(N, n_layers * n_heads, head_dim)
