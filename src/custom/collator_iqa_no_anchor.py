from dataclasses import dataclass
from typing import Any, Dict, List

import torch


@dataclass
class IQACollatorNoAnchor:
    tokenizer: Any

    def _concat_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        IMPORTANT: This must match the exact way you build the raw text for tokenizer -> input_ids/labels
        (same chat template). If your project uses a custom chat template, call it here instead of a raw join.
        """
        return "".join([message["content"] for message in messages])

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        pad_id = self.tokenizer.pad_token_id or 0

        def pad_stack(key: str, pad_value: int) -> torch.Tensor:
            return torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(feature[key], dtype=torch.long) for feature in features],
                batch_first=True,
                padding_value=pad_value,
            )

        batch: Dict[str, Any] = {
            "input_ids": pad_stack("input_ids", pad_id),
            "labels": pad_stack("labels", -100),
            "attention_mask": pad_stack("attention_mask", 0),
            "raw_texts": [self._concat_messages(feature["messages"]) for feature in features],
            "mos": torch.tensor(
                [float(feature.get("mos", feature.get("gt_score"))) for feature in features],
                dtype=torch.float32,
            ),
        }

        if "images" in features[0]:
            batch["images"] = [feature["images"] for feature in features]

        return batch
