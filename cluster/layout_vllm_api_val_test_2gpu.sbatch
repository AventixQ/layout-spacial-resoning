#!/bin/bash
#SBATCH --job-name=layout-vllm-2g
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_geforce_rtx_4090:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/home/s504297/jobs/logs/layout_vllm_val_test_2gpu_%j.out
#SBATCH --error=/home/s504297/jobs/logs/layout_vllm_val_test_2gpu_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/work/s504297/layout-spacial-resoning}"
cd "$PROJECT_DIR"

export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"

exec bash cluster/layout_vllm_api_val_test.sbatch
