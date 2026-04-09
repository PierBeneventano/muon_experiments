#!/bin/bash
# Setup NanoGPT with Muon optimizer support
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Clone nanoGPT if not present
if [ ! -d "nanoGPT" ]; then
    echo "Cloning nanoGPT..."
    git clone https://github.com/karpathy/nanoGPT.git
fi

# Prepare Shakespeare char-level data
echo "Preparing Shakespeare char-level data..."
cd nanoGPT/data/shakespeare_char
python prepare.py
cd ../../..

# Apply Muon patch
echo "Applying Muon optimizer patch to train.py..."
python experiments/nanogpt/patch_muon.py

echo "Setup complete. NanoGPT is ready with Muon support."
