#!/usr/bin/env bash
# Create the project venv and install dependencies.
# GB10 (Grace-Blackwell, aarch64, CUDA 13): the default PyPI torch aarch64 wheel
# bundles CUDA SBSA libs. If torch.cuda is unavailable after install, rerun with
# the cu130 index: pip install torch --index-url https://download.pytorch.org/whl/cu130
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt
./venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda_available:', torch.cuda.is_available())"
./venv/bin/python -m ipykernel install --user --name dpa_kt --display-name "Python (dpa_kt)"
echo "venv ready: $ROOT/venv"
