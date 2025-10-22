import math
import re
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


ANSWER_PATTERN = re.compile(r"<answer>\s*([1-5]\.\d{2})\s*</answer>", re.IGNORECASE | re.DOTALL)


def build_score_candidates(tokenizer, step: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if step <= 0:
        raise ValueError("`step` must be positive.")

    values = []
    current = 1.00
    while current <= 5.0001:  # include 5.00
        values.append(round(current, 2))
        current = round(current + step, 4)

    cand_tokens = []
    max_len = 0
    for value in values:
        text = f"{value:.2f}"
        encoded = tokenizer(text, add_special_tokens=False)
        ids = encoded["input_ids"]
        cand_tokens.append(ids)
        max_len = max(max_len, len(ids))

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    cand_ids = torch.full((len(values), max_len), pad_id, dtype=torch.long)
    cand_mask = torch.zeros((len(values), max_len), dtype=torch.bool)
    for i, ids in enumerate(cand_tokens):
        cand_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        cand_mask[i, : len(ids)] = True

    grid_vals = torch.tensor(values, dtype=torch.float32)
    return grid_vals, cand_ids, cand_mask


def find_answer_start_position(rendered_text: str, tokenizer) -> int:
    match = ANSWER_PATTERN.search(rendered_text)
    if match is None:
        return -1

    num_start = match.start(1)
    prefix = rendered_text[:num_start]
    encoded = tokenizer(prefix, add_special_tokens=False)
    ans_pos = len(encoded["input_ids"])
    return ans_pos


def seq_logprob_at(
    logits: torch.Tensor,
    ans_pos: int,
    cand_ids: torch.Tensor,
    cand_mask: torch.Tensor,
) -> torch.Tensor:
    if ans_pos < 0:
        raise ValueError("ans_pos must be non-negative.")

    L = cand_ids.size(1)
    vocab_size = logits.size(-1)
    if ans_pos >= logits.size(0):
        seg = logits.new_full((L, vocab_size), -1e9)
    else:
        seg = logits[ans_pos : ans_pos + L]
        if seg.size(0) < L:
            pad_rows = L - seg.size(0)
            pad_tensor = logits.new_full((pad_rows, vocab_size), -1e9)
            seg = torch.cat([seg, pad_tensor], dim=0)

    log_probs = F.log_softmax(seg.float(), dim=-1)

    cand_ids = cand_ids.to(log_probs.device)
    cand_mask = cand_mask.to(log_probs.device)

    expanded = log_probs.unsqueeze(0).expand(cand_ids.size(0), -1, -1)
    gathered = torch.gather(expanded, dim=2, index=cand_ids.unsqueeze(-1)).squeeze(-1)
    gathered = gathered.masked_fill(~cand_mask, 0.0)
    logp_per_cand = gathered.sum(dim=1)
    return logp_per_cand


def huber_loss(err: torch.Tensor, delta: float) -> torch.Tensor:
    abs_err = err.abs()
    quadratic = 0.5 * (abs_err**2)
    linear = delta * (abs_err - 0.5 * delta)
    return torch.where(abs_err <= delta, quadratic, linear)


def sample_delta_pairs(
    gt_scores: torch.Tensor,
    valid_mask: torch.Tensor,
    num_pairs: int,
    near_ratio: float,
    near_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if num_pairs <= 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
    n_valid = valid_indices.numel()
    if n_valid < 2:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    scores = gt_scores[valid_indices]
    diff_matrix = torch.abs(scores.unsqueeze(0) - scores.unsqueeze(1))
    iu, ju = torch.triu_indices(n_valid, n_valid, offset=1, device=gt_scores.device)
    all_diffs = diff_matrix[iu, ju]

    near_mask = all_diffs <= near_threshold
    near_indices = torch.nonzero(near_mask, as_tuple=False).squeeze(-1)
    far_indices = torch.nonzero(~near_mask, as_tuple=False).squeeze(-1)

    num_near_target = int(math.ceil(num_pairs * near_ratio))
    num_far_target = num_pairs - num_near_target

    selected_near = []
    if near_indices.numel() > 0 and num_near_target > 0:
        perm = torch.randperm(near_indices.numel(), device=near_indices.device)
        count = min(num_near_target, near_indices.numel())
        selected_near = near_indices[perm[:count]]

    selected_far = []
    remaining = num_pairs - len(selected_near)
    if far_indices.numel() > 0 and remaining > 0:
        perm = torch.randperm(far_indices.numel(), device=far_indices.device)
        count = min(remaining, far_indices.numel())
        selected_far = far_indices[perm[:count]]

    selected = []
    if len(selected_near) > 0:
        selected.append(selected_near)
    if len(selected_far) > 0:
        selected.append(selected_far)

    if not selected:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    selected = torch.cat(selected, dim=0)
    i_idx = valid_indices[iu[selected]]
    j_idx = valid_indices[ju[selected]]
    return i_idx, j_idx
