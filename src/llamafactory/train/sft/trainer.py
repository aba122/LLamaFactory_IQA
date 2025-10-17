# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # Configure FP8 environment if enabled
        if model_args is not None and model_args.fp8:
            configure_fp8_environment(model_args)
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        self._iqa_score_loss = None

        # Verify FP8 status after trainer initialization (accelerator should be available)
        if model_args is not None and model_args.fp8 and hasattr(self, "accelerator"):
            verify_fp8_status(self.accelerator, model_args)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    def _get_iqa_tokenizer(self):
        tokenizer = getattr(self, "processing_class", None)
        if tokenizer is None:
            tokenizer = getattr(self, "tokenizer", None)

        if tokenizer is None:
            raise RuntimeError("Tokenizer is required to compute IQA losses but was not initialized.")

        return tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer

    def _maybe_init_iqa_loss(self) -> None:
        if self._iqa_score_loss is not None:
            return

        try:
            from custom.loss_iqa_no_anchor import IQAScoreLoss
        except ImportError as err:
            raise ImportError(
                "Cannot import IQAScoreLoss from `custom.loss_iqa_no_anchor`. "
                "Ensure `src/custom/` is on PYTHONPATH and files are present."
            ) from err

        tokenizer = self._get_iqa_tokenizer()
        self._iqa_score_loss = IQAScoreLoss(
            tokenizer,
            step=self.finetuning_args.iqa_loss_step,
            sigma=self.finetuning_args.iqa_loss_sigma,
            delta=self.finetuning_args.iqa_loss_delta,
        )

    def _compute_iqa_losses(
        self, logits: torch.Tensor, input_ids: torch.Tensor, mos: torch.Tensor
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if logits.numel() == 0:
            return None

        self._maybe_init_iqa_loss()

        try:
            from custom.anspos_no_anchor import find_ans_pos_via_prefix
        except ImportError as err:
            raise ImportError(
                "Cannot import `find_ans_pos_via_prefix` from `custom.anspos_no_anchor`. "
                "Ensure custom modules are available."
            ) from err

        tokenizer = self._get_iqa_tokenizer()
        input_texts = tokenizer.batch_decode(input_ids.detach().cpu().tolist(), skip_special_tokens=False)

        ans_positions: list[int] = []
        valid_indices: list[int] = []
        for idx, text in enumerate(input_texts):
            try:
                ans_pos, _ = find_ans_pos_via_prefix(text, tokenizer)
            except ValueError:
                continue

            ans_positions.append(ans_pos)
            valid_indices.append(idx)

        if len(valid_indices) == 0:
            return None

        device = logits.device
        ans_pos_tensor = torch.tensor(ans_positions, dtype=torch.long, device=device)
        logits_subset = logits[valid_indices]
        mos_subset = mos[valid_indices]

        loss_kl, loss_reg, s_hat = self._iqa_score_loss.extra_losses(logits_subset, ans_pos_tensor, mos_subset)
        return loss_kl, loss_reg, s_hat

    @override
    def compute_loss(
        self, model, inputs, return_outputs: bool = False, *args, **kwargs
    ):  # type: ignore[override]
        forward_inputs = {k: v for k, v in inputs.items() if k not in {"mos", "gt_score"}}
        loss, outputs = super().compute_loss(model, forward_inputs, return_outputs=True, *args, **kwargs)
        total_loss = loss

        if getattr(self.finetuning_args, "use_iqa_no_anchor_loss", False):
            mos_tensor = inputs.get("mos") or inputs.get("gt_score")
            logits = getattr(outputs, "logits", None)
            if mos_tensor is not None and logits is not None:
                mos_tensor = mos_tensor.to(logits.device).float()
                iqa_losses = self._compute_iqa_losses(logits, inputs["input_ids"], mos_tensor)
                if iqa_losses is not None:
                    loss_kl, loss_reg, s_hat = iqa_losses
                    total_loss = loss + (
                        self.finetuning_args.iqa_loss_kl_weight * loss_kl
                        + self.finetuning_args.iqa_loss_reg_weight * loss_reg
                    )
                    if hasattr(outputs, "loss"):
                        outputs.loss = total_loss

                    try:
                        self.log(
                            {
                                "loss_ce": float(loss.detach().cpu()),
                                "loss_kl": float(loss_kl.detach().cpu()),
                                "loss_reg": float(loss_reg.detach().cpu()),
                                "s_hat_mean": float(s_hat.detach().mean().cpu()),
                            }
                        )
                    except Exception:
                        pass

        return (total_loss, outputs) if return_outputs else total_loss

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
