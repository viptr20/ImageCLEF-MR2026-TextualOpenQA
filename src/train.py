import argparse
import yaml
import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
)
from .model_utils import load_model, load_tokenizer
from .data_utils import prepare_dataset
from .metrics_wrapper import compute_metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))

    model_name = cfg["model_name"]
    output_dir = cfg["output_dir"]

    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name)
    device = torch.device("cpu")
    model.to(device)

    compute_metrics.tokenizer = tokenizer  # type: ignore

    dataset_cfg = {
        "dataset_name": cfg["dataset_name"],
        "text_field_question": cfg["text_field_question"],
        "text_field_answer": cfg["text_field_answer"],
        "id_field": cfg["id_field"],
        "language_field": cfg["language_field"],
        "max_source_length": cfg["max_source_length"],
        "max_target_length": cfg["max_target_length"],
    }
    tokenized = prepare_dataset(tokenizer, dataset_cfg)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=float(cfg["num_train_epochs"]),
        learning_rate=float(cfg["learning_rate"]),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(cfg["per_device_eval_batch_size"]),
        weight_decay=float(cfg["weight_decay"]),
        save_total_limit=int(cfg["save_total_limit"]),
        predict_with_generate=True,
        fp16=False,  # force no mixed precision
        logging_steps=int(cfg["logging_steps"]),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized.get("train"),
        eval_dataset=tokenized.get("validation"),
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()