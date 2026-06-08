import datasets
from datasets import load_dataset
from typing import Dict
from .ocr_utils import ocr_image_to_text  # new import

LANG_TAG_PREFIX = "lang:"

def load_openqa_dataset(dataset_name: str):
    ds = load_dataset(dataset_name)
    return ds

def build_source_text(example, cfg, tokenizer):
    lang = example.get(cfg["language_field"], "")
    subject = example.get("subject", "")
    # OCR step: image -> question_text
    image = example.get("image", None)
    question_text = ""
    if image is not None:
        try:
            question_text = ocr_image_to_text(image)
        except Exception:
            question_text = ""

    # Build prompt
    parts = [f"{LANG_TAG_PREFIX} {lang}"]
    if subject:
        parts.append(f"subject: {subject}")
    if question_text:
        parts.append(f"question: {question_text}")
    source_text = " ".join(parts)
    return lang, source_text

def preprocess_example(example, tokenizer, cfg):
    lang, source_text = build_source_text(example, cfg, tokenizer)
    answer = example.get(cfg["text_field_answer"], "").strip() if cfg.get("is_train", True) else ""

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