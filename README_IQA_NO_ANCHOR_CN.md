# IQA SFT（无锚点）适配 LLaMA-Factory / Qwen2.5-VL

本文介绍如何在 **不改模型结构、且不插入 `<ANS>`** 的前提下，让模型在 `<answer>x.xx</answer>` 中学习连续评分，对应三件套损失：
1. `L_CE`：保持原有自回归交叉熵；
2. `L_KL`：预测分布对齐以 MOS 为均值的 **高斯软标签**；
3. `L_Reg`：对候选分布期望值 \( \hat{s} = \sum s \,\pi(s) \) 施加 Huber（或 MSE）。

通过“正则 + 前缀重分词”定位 `<answer>` 内数字的 token 起点，无需引入额外锚点词。

## 新增文件
- `src/custom/collator_iqa_no_anchor.py`：示例 collator，提供 `raw_texts` 与 `mos`；
- `src/custom/anspos_no_anchor.py`：结合正则与前缀重分词，返回数字起始 token；
- `src/custom/loss_iqa_no_anchor.py`：构造候选网格、序列概率、Gaussian KL 与 Huber；
- `src/custom/trainer_iqa_no_anchor.py`：示例 Trainer，将 `L_KL` 和 `L_Reg` 叠加到 CE 上；
- `src/custom/__init__.py`：便于导入；
- `scripts/train_iqa_no_anchor_example.py`：最小可运行脚本；
- `README_IQA_NO_ANCHOR.md`：英文说明。

## 数据约定
- 保持原消息格式，例如：
  ```
  <think>…</think>
  <answer>4.52</answer>
  ```
  `<answer>` 内仅包含一个 `x.xx` 数值（范围 [1.00, 5.00]）。
- 数据集中需提供 MOS 字段 `mos`（或 `gt_score`，框架会自动回退）。

## 工作流程
1. 正则定位 `<answer>…</answer>`，提取分数及其在原文本中的字符起点；
2. 前缀重分词将字符起点映射为 token 起点 `ans_pos`；
3. 依据 `logits[:, ans_pos : ans_pos + L]` 计算候选数字串的序列对数概率（步长 0.05→0.01）；
4. 归一化得到 \( \pi(s) \)，并叠加：
   - **KL(Gaussian)**：均值 = MOS，建议 σ≈0.12–0.20；
   - **Huber**：作用于期望值 \( \hat{s} \)，建议 δ≈0.1。

## 推荐超参
- 候选网格：前期使用 0.05，后期可切换到 0.01（`IQAScoreLoss.switch_to_fine(0.01)`）；
- 高斯 σ：0.12–0.20（默认 0.15）；Huber δ：0.1；
- （可选）在 CE 中对数字 token 加权（1.2–1.5）以缓解离散偏差。

## 示例命令
```bash
python scripts/train_iqa_no_anchor_example.py \
  --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
  --train_json your_train.jsonl \
  --eval_json your_eval.jsonl
```

## CLI 集成（`llamafactory-cli train`）
- 在 `data/dataset_info.json` 中为数据集注册评分字段，例如 `"score": "gt_score"` 或 `"score": "mos"`。
- 在 YAML 配置的 `### method` 区域开启 `use_iqa_no_anchor_loss: true`，并按需调整 `iqa_loss_step`、`iqa_loss_sigma`、`iqa_loss_delta`、`iqa_loss_kl_weight`、`iqa_loss_reg_weight`。
- 参考配置：`examples/train_full/qwen2_5vl_full_sft_new_loss.yaml`。

## 注意事项
- 只要在 `dataset_info.json` 映射了 MOS 字段，默认 collator 会自动传递该值，无需额外拼接 `raw_texts`。
- 序列概率计算统一使用 **FP32**（`log_softmax` / `logsumexp`）避免数值下溢。
- CLI 训练路径复用了 `CustomSeq2SeqTrainer`；独立的脚本/Trainer 仍适合做自定义实验。
