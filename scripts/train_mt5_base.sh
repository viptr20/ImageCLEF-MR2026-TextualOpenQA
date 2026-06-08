#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG_PATH="configs/mt5_base.yaml"

python3 -m src.train --config "${CONFIG_PATH}"