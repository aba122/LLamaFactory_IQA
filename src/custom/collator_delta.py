from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence

from .utils_iqa import find_answer_start_position


@dataclass
class IQACollatorDelta:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        gt_scores = []
        ans_positions = []
        ans_valid_mask = []

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

        for feature in features:
            messages = feature["messages"]
            rendered = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            ans_pos = find_answer_start_position(rendered, self.tokenizer)
            ans_positions.append(ans_pos if ans_pos >= 0 else -1)
            ans_valid_mask.append(ans_pos >= 0)

            encoded = self.tokenizer(rendered, add_special_tokens=False)
            input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
            attention_mask = torch.tensor(encoded["attention_mask"], dtype=torch.long)

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)

            labels = input_ids.clone()
            labels_list.append(labels)

            score = feature.get("gt_score", feature.get("mos"))
            gt_scores.append(float(score) if score is not None else float("nan"))

        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
        attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "gt_scores": torch.tensor(gt_scores, dtype=torch.float32),
            "ans_pos": torch.tensor(ans_positions, dtype=torch.long),
            "ans_valid": torch.tensor(ans_valid_mask, dtype=torch.bool),
        }
        return batch
