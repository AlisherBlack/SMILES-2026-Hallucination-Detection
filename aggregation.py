"""
aggregation.py — H4 chunk-layer grid aggregation.

For each sample we emit a flat vector of size (len(LAYERS) * 4 * hidden_dim):

    layers = (13, 15, 17, 24)
    pools  = (response_head, response_mid, response_tail, last_seq)

Response token positions are computed in solution.py via the helpers below
(same logic as experiments/h4_chunk_layer_pca.py): offsets_mapping when the
tokenizer is fast, tokenizer.encode(prompt) length otherwise, fallback to the
last token.
"""

from __future__ import annotations

import numpy as np
import torch


LAYERS = (13, 15, 17, 24)


def response_positions_from_offsets(
    prompt: str,
    input_ids_row: torch.Tensor,
    attention_mask_row: torch.Tensor,
    offsets_row: torch.Tensor,
    eos_id: int,
) -> list[int]:
    n_real = int(attention_mask_row.sum().item())
    response_start_char = len(str(prompt))
    ids = input_ids_row[:n_real]
    offsets = offsets_row[:n_real].tolist()

    positions: list[int] = []
    for pos, ((_, end), token_id) in enumerate(zip(offsets, ids.tolist())):
        if token_id == eos_id:
            continue
        if end > response_start_char:
            positions.append(pos)
    return positions


def response_positions_from_prompt_len(
    prompt: str,
    input_ids_row: torch.Tensor,
    attention_mask_row: torch.Tensor,
    tokenizer,
    eos_id: int,
) -> list[int]:
    n_real = int(attention_mask_row.sum().item())
    prompt_len = len(tokenizer.encode(str(prompt), add_special_tokens=False))
    start = min(prompt_len, max(0, n_real - 1))

    real_ids = input_ids_row[:n_real]
    eos_positions = (real_ids == eos_id).nonzero(as_tuple=True)[0]
    stop = int(eos_positions[-1].item()) if eos_positions.numel() > 0 else n_real
    return list(range(start, max(start, stop)))


def fallback_response_positions(
    attention_mask_row: torch.Tensor,
    input_ids_row: torch.Tensor,
    eos_id: int,
) -> list[int]:
    n_real = int(attention_mask_row.sum().item())
    if n_real <= 1:
        return [0]
    last_pos = n_real - 1
    if int(input_ids_row[last_pos].item()) == eos_id and last_pos > 0:
        return [last_pos - 1]
    return [last_pos]


def _split_response_chunks(
    positions: list[int],
) -> tuple[list[int], list[int], list[int]]:
    arr = np.array(positions, dtype=int)
    n = len(arr)
    head_end = max(1, int(np.ceil(n / 3)))
    mid_start = min(n - 1, n // 3)
    mid_end = min(n, max(mid_start + 1, int(np.ceil(2 * n / 3))))
    tail_start = max(0, n - head_end)
    return arr[:head_end].tolist(), arr[mid_start:mid_end].tolist(), arr[tail_start:].tolist()


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    response_positions: list[int],
) -> torch.Tensor:
    real_positions = attention_mask.nonzero(as_tuple=False).squeeze(-1)
    last_pos = int(real_positions[-1].item())

    head, mid, tail = _split_response_chunks(response_positions)
    pools = (head, mid, tail, [last_pos])

    parts: list[torch.Tensor] = []
    for layer_idx in LAYERS:
        layer_hidden = hidden_states[layer_idx]
        for pool in pools:
            indices = torch.tensor(pool, dtype=torch.long, device=layer_hidden.device)
            parts.append(layer_hidden.index_select(0, indices).mean(dim=0))

    return torch.cat(parts, dim=0).float().cpu()


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    return torch.zeros(0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    response_positions: list[int],
    use_geometric: bool = False,
) -> torch.Tensor:
    agg = aggregate(hidden_states, attention_mask, response_positions)
    if use_geometric:
        geo = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg, geo], dim=0)
    return agg
