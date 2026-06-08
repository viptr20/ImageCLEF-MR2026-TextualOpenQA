import datasets
from datasets import load_dataset
from typing import Dict

LANG_TAG_PREFIX = "lang:"

def load_openqa_dataset(dataset_name: str):
    ds = load_dataset(dataset_name)
    return ds

def preprocess_example(example, tokenizer, cfg):
    lang = example.get(cfg["language_field"], "")
    question = example.get(cfg["text_field_question"], "").strip()
    answer = example.get(cfg["text_field_answer"], "").strip() if cfg.get("is_train", True) else ""

    prefix = f"{LANG_TAG_PREFIX} {lang} question: "
    source_text = prefix + question

    model_inputs = tokenizer(
        source_text,
        max_length=cfg["max_source_length"],
        truncation=True,
    )

    if cfg.get("with_labels", True):
        labels = tokenizer(
            answer,
            max_length=cfg["max_target_length"],
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]

    model_inputs["id"] = example.get(cfg["id_field"], None)
    model_inputs["language"] = lang
    return model_inputs

def prepare_dataset(tokenizer, cfg: Dict):
    ds = load_openqa_dataset(cfg["dataset_name"])

    train_cfg = dict(cfg)
    train_cfg["is_train"] = True
    train_cfg["with_labels"] = True

    def _preprocess_train(ex):
        return preprocess_example(ex, tokenizer, train_cfg)

    def _preprocess_eval(ex):
        return preprocess_example(ex, tokenizer, train_cfg)

    tokenized = {}
    if "train" in ds:
        tokenized["train"] = ds["train"].map(
            _preprocess_train,
            remove_columns=ds["train"].column_names,
            batched=False,
        )
    if "dev" in ds or "validation" in ds:
        split_name = "dev" if "dev" in ds else "validation"
        tokenized["validation"] = ds[split_name].map(
            _preprocess_eval,
            remove_columns=ds[split_name].column_names,
            batched=False,
        )
    if "test" in ds:
        test_cfg = dict(cfg)
        test_cfg["is_train"] = False
        test_cfg["with_labels"] = False

        def _preprocess_test(ex):
            return preprocess_example(ex, tokenizer, test_cfg)

        tokenized["test"] = ds["test"].map(
            _preprocess_test,
            remove_columns=ds["test"].column_names,
            batched=False,
        )

    return datasets.DatasetDict(tokenized)