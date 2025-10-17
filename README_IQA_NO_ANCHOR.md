# IQA SFT (No-Anchor) for LLaMA-Factory / Qwen2.5-VL

This patch adds **non-intrusive** modules to train **continuous image quality scores** with:
1) `L_CE` (original autoregressive CE)
2) `L_KL` (KL divergence to a **Gaussian soft label** centered at MOS)
3) `L_Reg` (Huber/MSE on the expected score \( \hat{s} = \sum s\,\pi(s) \))

No model architecture change. **No `<ANS>` token** is inserted. We locate the numeric answer inside `<answer>…</answer>` by **regex + prefix re-tokenization**, then compute the candidate score distribution \( \pi(s) \) at that token span.

## Files added
- `src/custom/collator_iqa_no_anchor.py` — collator that provides `raw_texts` and `mos`
- `src/custom/anspos_no_anchor.py` — locate `ans_pos` (token index of the number) via regex + prefix re-tokenization
- `src/custom/loss_iqa_no_anchor.py` — candidate grid, sequence-probabilities, KL(Gaussian), Huber(mean)
- `src/custom/trainer_iqa_no_anchor.py` — custom Trainer that adds `L_KL` and `L_Reg` on top of CE
- `src/custom/__init__.py` — makes the package importable
- `scripts/train_iqa_no_anchor_example.py` — minimal example to run training

## Data contract
- Keep your original messages:
  ```
  <think>…</think>
  <answer>4.52</answer>
  ```
  Only **one** number `x.xx` in `<answer>`; range [1.00, 5.00].
- Provide MOS as `mos` (or `gt_score` — we fallback to it).

## How it works
1) Regex finds the number & its **character start** inside `<answer>…</answer>`.
2) Prefix re-tokenization turns the char-start into the **first token index** `ans_pos`.
3) From `logits[:, ans_pos : ans_pos + L]` we build **sequence log-probs** for each candidate number string (grid step 0.05 → 0.01).
4) Normalize to \( \pi(s) \). Add
   - **KL to Gaussian soft label** (center = MOS, σ ≈ 0.12–0.20)
   - **Huber on mean** (δ ≈ 0.1)

## Suggested hyperparams
- Grid: start `0.05`, switch to `0.01` for last 1–2 epochs (`IQAScoreLoss.switch_to_fine(0.01)`).
- Gaussian σ: `0.12–0.20` (default 0.15); Huber δ: `0.1`.
- (Optional) Upweight CE on numeric tokens (1.2–1.5) in your CE implementation.

## Example run
```bash
python scripts/train_iqa_no_anchor_example.py \
  --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
  --train_json your_train.jsonl \
  --eval_json your_eval.jsonl
```

## CLI integration (`llamafactory-cli train`)
- Register the MOS (or GT) field in `data/dataset_info.json`, e.g. `"score": "gt_score"` (or `"score": "mos"` if that is the column name).
- Enable `use_iqa_no_anchor_loss: true` inside your YAML (under the `### method` block) and optionally adjust `iqa_loss_step`, `iqa_loss_sigma`, `iqa_loss_delta`, `iqa_loss_kl_weight`, `iqa_loss_reg_weight`.
- Ready-made config: `examples/train_full/qwen2_5vl_full_sft_new_loss.yaml`.

## Notes
- The built-in collator forwards the MOS tensor automatically as long as the dataset entry exposes it via `dataset_info.json`.
- Sequence-probability math uses **FP32** (`log_softmax`/`logsumexp`) to avoid underflow.
- The CLI pipeline now reuses the default `CustomSeq2SeqTrainer`; the standalone trainer/script remain available for bespoke experiments.
