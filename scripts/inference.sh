#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

pip install -r requirements.txt

python3 -m src.predict \
  --model_dir "outputs/mt5_base" \
  --dataset_name "SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Textual" \
  --split "test" \
  --output_path "outputs/mt5_base_predictions_test.json"