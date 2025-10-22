import argparse

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

from src.custom.collator_delta import IQACollatorDelta
from src.custom.trainer_delta import DeltaArgs, IQATrainerDelta


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--eval_json", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accum", type=int, default=8)

    parser.add_argument("--enable_delta_loss", action="store_true")
    parser.add_argument("--delta_weight", type=float, default=0.5)
    parser.add_argument("--delta_pairs_per_batch", type=int, default=32)
    parser.add_argument("--delta_huber_delta", type=float, default=0.1)
    parser.add_argument("--delta_near_ratio", type=float, default=0.8)
    parser.add_argument("--delta_near_threshold", type=float, default=0.3)
    parser.add_argument("--score_grid_step", type=float, default=0.05)

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=600)

    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )

    train_dataset = load_dataset("json", data_files=args.train_json, split="train")
    eval_dataset = load_dataset("json", data_files=args.eval_json, split="train")

    data_collator = IQACollatorDelta(tokenizer)
    delta_args = DeltaArgs(
        enable_delta_loss=args.enable_delta_loss,
        delta_weight=args.delta_weight,
        delta_pairs_per_batch=args.delta_pairs_per_batch,
        delta_huber_delta=args.delta_huber_delta,
        delta_near_ratio=args.delta_near_ratio,
        delta_near_threshold=args.delta_near_threshold,
        score_grid_step=args.score_grid_step,
    )

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        remove_unused_columns=False,
        bf16=True,
        report_to="none",
    )

    trainer = IQATrainerDelta(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        delta_args=delta_args,
    )

    trainer.train()


if __name__ == "__main__":
    main()
