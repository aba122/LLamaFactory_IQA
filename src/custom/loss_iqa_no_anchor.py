from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


class IQAScoreLoss:
    """
    Build candidate score strings on a grid, compute their sequence log-probs at the numeric span (ans_pos),
    derive pi(s), and add:
      - KL(pi || Gaussian(MOS, sigma))  (implemented as KL(q||pi))
      - Huber/EU on the expected score (mean of pi)
    """

    def __init__(self, tokenizer, step: float = 0.05, sigma: float = 0.15, delta: float = 0.1) -> None:
        self.tokenizer = tokenizer
        self.sigma = sigma
        self.delta = delta
        self.vals, self.cand_ids, self.cand_mask = self._build_grid(step)

    def switch_to_fine(self, step: float = 0.01) -> None:
        self.vals, self.cand_ids, self.cand_mask = self._build_grid(step)

    def _build_grid(self, step: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vals = torch.round(torch.arange(1.00, 5.00 + 1e-9, step), decimals=2)
        str_vals = [f"{value:.2f}" for value in vals.tolist()]
        tokenized = [self.tokenizer(value, add_special_tokens=False)["input_ids"] for value in str_vals]
        max_len = max(len(toks) for toks in tokenized)
        pad = self.tokenizer.pad_token_id or 0
        cand_ids = torch.full((len(str_vals), max_len), pad, dtype=torch.long)
        cand_mask = torch.zeros((len(str_vals), max_len), dtype=torch.bool)
        for idx, toks in enumerate(tokenized):
            cand_ids[idx, : len(toks)] = torch.tensor(toks, dtype=torch.long)
            cand_mask[idx, : len(toks)] = True

        return vals, cand_ids, cand_mask

    def _seq_logp(self, logits: torch.Tensor, ans_pos: torch.Tensor) -> torch.Tensor:
        """
        logits: (B,T,V), ans_pos: (B,)
        Returns: (B,C) log-prob of each candidate sequence starting at ans_pos
        """
        batch, time, vocab = logits.shape
        num_cands, max_len = self.cand_ids.shape
        device = logits.device

        offsets = torch.arange(max_len, device=device).unsqueeze(0)
        indices = ans_pos.unsqueeze(1) + offsets
        indices = torch.clamp(indices, max=time - 1)
        segment = logits.gather(1, indices.unsqueeze(-1).expand(-1, -1, vocab))
        log_probs = F.log_softmax(segment.float(), dim=-1)
        gathered = log_probs.unsqueeze(1).expand(-1, num_cands, -1, -1).gather(
            -1, self.cand_ids.to(device).unsqueeze(0).unsqueeze(-1)
        )
        gathered = gathered.squeeze(-1)
        gathered = gathered.masked_fill(~self.cand_mask.to(device).unsqueeze(0), 0.0)
        return gathered.sum(-1)

    def _kl_gauss(self, pi: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        values = self.vals.to(pi.device).unsqueeze(0)
        q = torch.exp(-0.5 * ((values - target.unsqueeze(1)) / self.sigma) ** 2)
        q = q / q.sum(1, keepdim=True).clamp_min(1e-12)
        return (q * (q.clamp_min(1e-12).log() - pi.clamp_min(1e-12).log())).sum(1).mean()

    def _huber_on_mean(self, pi: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        values = self.vals.to(pi.device).unsqueeze(0)
        s_hat = (pi * values).sum(1)
        return F.huber_loss(s_hat, target.to(pi.device), delta=self.delta), s_hat

    def extra_losses(
        self, logits: torch.Tensor, ans_pos: torch.Tensor, mos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_logp = self._seq_logp(logits, ans_pos)
        pi = torch.softmax(seq_logp, dim=1)
        loss_kl = self._kl_gauss(pi, mos)
        loss_reg, s_hat = self._huber_on_mean(pi, mos)
        return loss_kl, loss_reg, s_hat
