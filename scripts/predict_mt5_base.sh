#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_DIR="outputs/mt5_base"
DATASET_NAME="SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Textual"
SPLIT="test"
OUTPUT_PATH="outputs/mt5_base_predictions_test.json"

python3 -m src.predict \
  --model_dir "${MODEL_DIR}" \
  --dataset_name "${DATASET_NAME}" \
  --split "${SPLIT}" \
  --output_path "${OUTPUT_PATH}" \
  --max_new_tokens 64 \
  --num_beams 4