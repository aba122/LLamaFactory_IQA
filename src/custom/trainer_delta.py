from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from transformers import Trainer

from .utils_iqa import (
    build_score_candidates,
    huber_loss,
    sample_delta_pairs,
    seq_logprob_at,
)


@dataclass
class DeltaArgs:
    enable_delta_loss: bool = False
    delta_weight: float = 0.5
    delta_pairs_per_batch: int = 32
    delta_huber_delta: float = 0.1
    delta_near_ratio: float = 0.8
    delta_near_threshold: float = 0.3
    score_grid_step: float = 0.05


class IQATrainerDelta(Trainer):
    def __init__(
        self,
        *args,
        delta_args: Optional[DeltaArgs] = None,
        tokenizer=None,
        **kwargs,
    ) -> None:
        super().__init__(*args, tokenizer=tokenizer, **kwargs)

        self.delta_args = delta_args or DeltaArgs()
        if tokenizer is None:
            raise ValueError("Tokenizer is required for IQATrainerDelta.")

        (
            self._grid_vals_cpu,
            self._cand_ids_cpu,
            self._cand_mask_cpu,
        ) = build_score_candidates(tokenizer, self.delta_args.score_grid_step)

    def _scores_from_logits(self, logits: torch.Tensor, ans_pos: torch.Tensor) -> torch.Tensor:
        device = logits.device
        cand_ids = self._cand_ids_cpu.to(device)
        cand_mask = self._cand_mask_cpu.to(device)
        grid_vals = self._grid_vals_cpu.to(device)

        scores = []
        for b in range(logits.size(0)):
            pos = int(ans_pos[b].item())
            logp = seq_logprob_at(logits[b], pos, cand_ids, cand_mask)
            probs = torch.softmax(logp, dim=0)
            probs = probs.clamp_min(1e-12)
            probs = probs / probs.sum()
            score = torch.sum(probs * grid_vals)
            scores.append(score)

        return torch.stack(scores, dim=0)

    def compute_loss(self, model, inputs: Dict[str, Any], return_outputs: bool = False):
        forward_inputs = {k: v for k, v in inputs.items() if k not in {"gt_scores", "ans_pos", "ans_valid"}}
        outputs = model(**forward_inputs)
        logits = outputs.logits

        if outputs.loss is not None:
            loss_ce = outputs.loss
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()
            loss_ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        total_loss = loss_ce
        loss_delta = torch.tensor(0.0, device=logits.device)

        if self.delta_args.enable_delta_loss:
            gt_scores = inputs["gt_scores"].to(logits.device)
            ans_pos = inputs["ans_pos"].to(logits.device)
            ans_valid = inputs["ans_valid"].to(logits.device)

            valid_mask = ans_valid & torch.isfinite(gt_scores)
            if valid_mask.sum() >= 2:
                scores_hat = self._scores_from_logits(logits, ans_pos)
                i_idx, j_idx = sample_delta_pairs(
                    gt_scores,
                    valid_mask,
                    self.delta_args.delta_pairs_per_batch,
                    self.delta_args.delta_near_ratio,
                    self.delta_args.delta_near_threshold,
                )

                if i_idx.numel() > 0:
                    d_hat = torch.abs(scores_hat[i_idx] - scores_hat[j_idx])
                    d_gt = torch.abs(gt_scores[i_idx] - gt_scores[j_idx])
                    err = d_hat - d_gt
                    loss_delta = huber_loss(err, self.delta_args.delta_huber_delta).mean()
                    total_loss = total_loss + self.delta_args.delta_weight * loss_delta

        try:
            self.log(
                {
                    "loss_ce": loss_ce.detach().float().item(),
                    "loss_delta": loss_delta.detach().float().item() if loss_delta is not None else 0.0,
                }
            )
        except Exception:
            pass

        return (total_loss, outputs) if return_outputs else total_loss
