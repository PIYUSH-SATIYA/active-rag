#!/bin/bash
# Run the FLARE pipeline with the local Qwen model.
# Usage: bash qwen.sh [--eval_mode no_retrieval|single_retrieval|flare]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate the venv
source .venv/bin/activate

# Make sure src/ is importable
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

echo "=== Starting FLARE run ==="
python scripts/run_flare.py "$@"
