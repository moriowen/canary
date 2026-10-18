# Sourced by setup.sh and every job. ICE home is small (30 GB), so the venv, the
# HF cache and the data live under ~/scratch. Override any of these by exporting first.
set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/scratch/canary}"
VENV="${VENV:-$HOME/scratch/canary-venv}"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/scratch/pip-cache}"
export CANARY_DATA="${CANARY_DATA:-$HOME/scratch/canary-data}"
export CANARY_RESULTS="${CANARY_RESULTS:-$PROJECT_DIR/results}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Lmod and venv activation reference unset variables, so strict mode (-u) comes after them.
if command -v module >/dev/null 2>&1; then
    module load python/3.12.5 >/dev/null 2>&1 || true
fi
if [ -f "$VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi
set -u

cd "$PROJECT_DIR"
mkdir -p logs "$CANARY_DATA" "$CANARY_RESULTS"
