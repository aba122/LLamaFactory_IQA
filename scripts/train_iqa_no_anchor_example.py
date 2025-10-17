import argparse

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from src.custom.collator_iqa_no_anchor import IQACollatorNoAnchor
from src.custom.loss_iqa_no_anchor import IQAScoreLoss
from src.custom.trainer_iqa_no_anchor import IQATrainerNoAnchor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--eval_json", type=str, required=True)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--out_dir", type=str, default="outputs/iqa_no_anchor")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=False, trust_remote_code=True
    )

    train_ds = load_dataset("json", data_files=args.train_json, split="train")
    eval_ds = load_dataset("json", data_files=args.eval_json, split="train")

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    data_collator = IQACollatorNoAnchor(tokenizer)
    score_loss = IQAScoreLoss(tokenizer, step=0.05, sigma=0.15, delta=0.1)

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        logging_steps=10,
        evaluation_strategy="no",
        save_strategy="epoch",
        bf16=True,
        save_total_limit=1,
        remove_unused_columns=False,
    )

    trainer = IQATrainerNoAnchor(
        score_loss=score_loss,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()


if __name__ == "__main__":
    main()
