from typing import Any, Dict

import torch
from transformers import Trainer

from src.custom.anspos_no_anchor import find_ans_pos_via_prefix


class IQATrainerNoAnchor(Trainer):
    """
    Custom Trainer that:
      - keeps CE from HF Trainer/model forward
      - adds KL(Gaussian soft label) + Huber(mean) on the numeric answer
    Assumes collator provided:
      - raw_texts: list[str], the exact raw text used to build input_ids/labels
      - mos: FloatTensor (B,)
    """

    def __init__(self, score_loss, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.score_loss = score_loss

    def compute_loss(
        self, model: torch.nn.Module, inputs: Dict[str, Any], return_outputs: bool = False
    ):
        forward_inputs = {key: value for key, value in inputs.items() if key not in {"raw_texts", "mos", "images"}}
        outputs = model(**forward_inputs)
        logits = outputs.logits

        loss_ce = getattr(outputs, "loss", None)
        if loss_ce is None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()
            loss_ce = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        raw_texts = inputs["raw_texts"]
        ans_pos = []
        for text in raw_texts:
            pos, _ = find_ans_pos_via_prefix(text, self.tokenizer)
            ans_pos.append(pos)
        ans_pos_tensor = torch.tensor(ans_pos, dtype=torch.long, device=logits.device)

        loss_kl, loss_reg, s_hat = self.score_loss.extra_losses(
            logits, ans_pos_tensor, inputs["mos"].to(logits.device)
        )
        loss = loss_ce + loss_kl + loss_reg

        try:
            self.log(
                {
                    "loss_ce": float(loss_ce.detach().cpu()),
                    "loss_kl": float(loss_kl.detach().cpu()),
                    "loss_reg": float(loss_reg.detach().cpu()),
                    "s_hat_mean": float(s_hat.detach().mean().cpu()),
                }
            )
        except Exception:
            pass

        return (loss, outputs) if return_outputs else loss
