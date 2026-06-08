#!/bin/bash
#SBATCH --job-name=layout-local-matrix
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_geforce_rtx_4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/s504297/jobs/logs/layout_local_hf_matrix_%A_%a.out
#SBATCH --error=/home/s504297/jobs/logs/layout_local_hf_matrix_%A_%a.err

set -euo pipefail

MODELS=(
  "Qwen/Qwen3-4B-Instruct-2507"
  "Qwen/Qwen3-8B"
  "speakleash/Bielik-11B-v3.0-Instruct"
  "CYFRAGOVPL/Llama-PLLuM-8B-instruct-2512"
)

export LOCAL_HF_LAYOUT_MODEL="${LOCAL_HF_LAYOUT_MODEL:-${MODELS[$SLURM_ARRAY_TASK_ID]}}"
export PROJECT_DIR="${PROJECT_DIR:-/work/s504297/layout-spacial-resoning}"
export FORMS_PATH="${FORMS_PATH:-data/splits/val.jsonl}"
export EXPERIMENT_LIMIT="${EXPERIMENT_LIMIT:-20}"
export LOCAL_HF_MAX_NEW_TOKENS="${LOCAL_HF_MAX_NEW_TOKENS:-4096}"
export LOCAL_HF_LOAD_IN_4BIT="${LOCAL_HF_LOAD_IN_4BIT:-true}"
export LOCAL_HF_INCLUDE_STRUCTURED_OUTPUT="${LOCAL_HF_INCLUDE_STRUCTURED_OUTPUT:-false}"

exec "$PROJECT_DIR/cluster/layout_local_hf_val.sbatch"
