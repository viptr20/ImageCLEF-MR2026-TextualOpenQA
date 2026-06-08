import argparse
import json

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import torch

from .data_utils import LANG_TAG_PREFIX
from .ocr_utils import ocr_image_to_text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to fine-tuned model directory.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="HF dataset name, e.g. SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Textual",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split to run prediction on: train/dev/test",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output JSON file for predictions.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Use sampling instead of pure beam search.",
    )
    return parser.parse_args()


def build_prompt(ex):
    lang = ex["language"]
    subject = ex.get("subject", "")
    image = ex.get("image", None)

    question_text = ""
    if image is not None:
        try:
            question_text = ocr_image_to_text(image)
        except Exception:
            question_text = ""

    parts = [f"{LANG_TAG_PREFIX} {lang}"]
    if subject:
        parts.append(f"subject: {subject}")
    if question_text:
        parts.append(f"question: {question_text}")
    parts.append("answer:")

    return " ".join(parts)


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    model.eval()

    ds = load_dataset(args.dataset_name)
    split = args.split
    if split not in ds:
        if split == "dev" and "validation" in ds:
            split = "validation"
        elif split == "validation" and "dev" in ds:
            split = "dev"
    data = ds[split]

    predictions = []

    for ex in data:
        qid = ex.get("question_id", ex.get("id"))
        input_text = build_prompt(ex)

        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )

        with torch.no_grad():
            generated = model.generate(
                **{k: v.to(model.device) for k, v in inputs.items()},
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                do_sample=args.do_sample,
            )

        answer_text = tokenizer.decode(
            generated[0],
            skip_special_tokens=True,
        ).strip()

        predictions.append(
            {
                "question_id": qid,
                "answer": answer_text,
            }
        )

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()