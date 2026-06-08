# ImageCLEF 2026 – Multimodal Reasoning – Textual OpenQA (mT5-Base Baseline)

This is a Tiny-category baseline solution for the **Textual OpenQA** track of ImageCLEF 2026 MultimodalReasoning using `google/mt5-base`.[page:1][web:20]

## 1. Setup

```bash
cd 2026/textual-openqa
pip install -r requirements.txt

# make helper scripts executable (once per clone)
chmod +x scripts/*.sh
```

Make sure you have access to:

- `SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Textual` on Hugging Face (accept terms in browser first).[web:20]

## 2. Training

```bash
bash scripts/train_mt5_base.sh
```

This will:

- Load the OpenQA dataset from Hugging Face.
- Fine-tune `google/mt5-base` on the train split.
- Evaluate on dev/validation with METEOR.
- Save the model and tokenizer to `outputs/mt5_base`.

You can modify hyperparameters in `configs/mt5_base.yaml`.

## 3. Prediction

To generate predictions for the test split:

```bash
bash scripts/predict_mt5_base.sh
```

This produces `outputs/mt5_base_predictions_test.json` of the form:

```json
[
  {
    "question_id": "5e9sf6b9-3338-4e97-ba6b-762e24a07e69",
    "answer": "your generated free-text answer"
  }
]
```

This format aligns with MCQ-style `question_id` identifiers, but uses free-form `answer` instead of `answer_key`.[web:20]

## 4. Competition Use

For leaderboard submissions:

1. Run the prediction script on the official test split (or the HF test split if instructed).
2. Upload `mt5_base_predictions_test.json` to the Textual OpenQA leaderboard on AI4Media-Bench.[web:3][web:28]

For final evaluation:

- Provide this repository URL.
- Provide or host the fine-tuned model under `outputs/mt5_base` (or adjust `inference.sh`).
- The organizers will run:

```bash
bash inference.sh
```

on their A40 GPU VM.[page:1]