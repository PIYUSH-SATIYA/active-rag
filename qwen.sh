#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting FLARE execution using Qwen local model..."

# Activate virtual environment if using venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Set python path
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Run the integration script
python scripts/run_flare.py

echo "Execution completed."
