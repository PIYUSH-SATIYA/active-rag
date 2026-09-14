#!/bin/bash
# Run the FLARE pipeline with the local Qwen model.
# Usage: bash qwen.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate the venv
source .venv/bin/activate

# Make sure src/ is importable
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

echo "=== Checking Elasticsearch ==="
if ! curl -sf http://localhost:9200 > /dev/null; then
    echo "ERROR: Elasticsearch is not reachable at http://localhost:9200"
    echo "Start it with: docker start flare-elasticsearch"
    exit 1
fi
echo "Elasticsearch OK."

echo ""
echo "=== Starting FLARE run ==="
python scripts/run_flare.py
