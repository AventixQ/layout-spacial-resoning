#!/bin/bash
#SBATCH --job-name=layout-local-hf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_geforce_rtx_4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/home/s504297/jobs/logs/layout_local_hf_%j.out
#SBATCH --error=/home/s504297/jobs/logs/layout_local_hf_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/work/s504297/layout-spacial-resoning}"
cd "$PROJECT_DIR"

export PATH="/work/s504297/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/s504297/uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/work/s504297/uv-python}"
export HF_HOME="${HF_HOME:-/work/s504297/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/work/s504297/hf-cache}"
export TORCH_HOME="${TORCH_HOME:-/work/s504297/torch-cache}"
export TMPDIR="${TMPDIR:-/work/s504297/tmp}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/work/s504297/pycache}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$HF_HOME" "$TORCH_HOME" "$TMPDIR" "$PYTHONPYCACHEPREFIX" /home/s504297/jobs/logs

export FORMS_PATH="${FORMS_PATH:-data/splits/val.jsonl}"
export EXPERIMENT_LIMIT="${EXPERIMENT_LIMIT:-20}"
export DEFAULT_EMBEDDING_PROVIDER="${DEFAULT_EMBEDDING_PROVIDER:-local}"
export EXPERIMENT_LLM_PROVIDERS="local_hf"
export INCLUDE_FINE_TUNED_MODEL=false
export INCLUDE_LOCAL_LORA=false
export LOCAL_HF_LAYOUT_MODEL="${LOCAL_HF_LAYOUT_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
export LOCAL_HF_MAX_NEW_TOKENS="${LOCAL_HF_MAX_NEW_TOKENS:-4096}"
export LOCAL_HF_LOAD_IN_4BIT="${LOCAL_HF_LOAD_IN_4BIT:-true}"
export LOCAL_HF_INCLUDE_STRUCTURED_OUTPUT="${LOCAL_HF_INCLUDE_STRUCTURED_OUTPUT:-false}"

SAFE_MODEL="$(printf '%s' "$LOCAL_HF_LAYOUT_MODEL" | tr '/:.' '___' | tr -cd '[:alnum:]_-' | tr '[:upper:]' '[:lower:]')"
LIMIT_TAG="$EXPERIMENT_LIMIT"
if [ "$LIMIT_TAG" = "0" ]; then
  LIMIT_TAG="full"
fi
export EXPERIMENT_RUN_ID="${EXPERIMENT_RUN_ID:-local_hf_${SAFE_MODEL}_val_${LIMIT_TAG}}"

nvidia-smi
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
echo "Run id: $EXPERIMENT_RUN_ID"
echo "Model : $LOCAL_HF_LAYOUT_MODEL"
echo "Forms : $FORMS_PATH"
echo "Limit : $EXPERIMENT_LIMIT"

uv run python experiments/run_full_experiment.py

cat "outputs/experiments/${EXPERIMENT_RUN_ID}/manifest.json"
cat "outputs/experiments/${EXPERIMENT_RUN_ID}/metrics/method_summary.md"
