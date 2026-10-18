#!/bin/bash
# One-time setup on the ICE login node:
#   git clone https://github.com/moriowen/canary ~/scratch/canary
#   cd ~/scratch/canary && bash pace/setup.sh
source "$(dirname "$0")/env.sh"
[ -f "$VENV/bin/activate" ] || python3 -m venv "$VENV"
set +u; source "$VENV/bin/activate"; set -u
pip install -q --upgrade pip
pip install -q -e '.[train]'
python -m pytest -q
echo "setup done"
