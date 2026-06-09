#!/bin/bash
#SBATCH --job-name=layout-vllm-vt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_geforce_rtx_4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=24:00:00
#SBATCH --output=/home/s504297/jobs/logs/layout_vllm_val_test_%j.out
#SBATCH --error=/home/s504297/jobs/logs/layout_vllm_val_test_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/work/s504297/layout-spacial-resoning}"
cd "$PROJECT_DIR"

export PATH="/work/s504297/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/s504297/uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/work/s504297/uv-python}"
export HF_HOME="${HF_HOME:-/work/s504297/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/work/s504297/hf-cache/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/work/s504297/hf-cache}"
export TORCH_HOME="${TORCH_HOME:-/work/s504297/torch-cache}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/work/s504297/pycache}"
export TOKENIZERS_PARALLELISM=false

NODE_TMP="${NODE_TMP:-/tmp/s504297-layout-vllm-${SLURM_JOB_ID:-$$}}"
export TMPDIR="$NODE_TMP"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$NODE_TMP/triton-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$NODE_TMP/xdg-cache}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$NODE_TMP/cuda-cache}"

mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$HF_HOME" "$HF_HUB_CACHE" "$TORCH_HOME" "$TMPDIR" "$PYTHONPYCACHEPREFIX" /home/s504297/jobs/logs
mkdir -p "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH"

MODEL="${VLLM_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
PORT="${VLLM_PORT:-8000}"
IMAGE="${VLLM_IMAGE:-/home/s504297/vllm901.sif}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-layout-local-model}"
SPLITS="${VLLM_SPLITS:-val,test}"
RUN_SUFFIX="${VLLM_RUN_SUFFIX:-}"

if command -v apptainer >/dev/null 2>&1; then
  CONTAINER_RUNTIME="apptainer"
elif command -v singularity >/dev/null 2>&1; then
  CONTAINER_RUNTIME="singularity"
else
  echo "Neither apptainer nor singularity is available." >&2
  exit 1
fi

if [ ! -f "$IMAGE" ]; then
  echo "Missing VLLM_IMAGE: $IMAGE" >&2
  exit 1
fi

SAFE_MODEL="$(printf '%s' "$MODEL" | tr '/:.' '___' | tr -cd '[:alnum:]_-' | tr '[:upper:]' '[:lower:]')"
LIMIT_TAG="${EXPERIMENT_LIMIT:-0}"
if [ "$LIMIT_TAG" = "0" ]; then
  LIMIT_TAG="full"
fi

export EXPERIMENT_LIMIT="${EXPERIMENT_LIMIT:-0}"
export DEFAULT_EMBEDDING_PROVIDER="${DEFAULT_EMBEDDING_PROVIDER:-local}"
export EXPERIMENT_LLM_PROVIDERS="openai"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
export OPENAI_LAYOUT_MODEL="$SERVED_MODEL_NAME"
export OPENAI_RESPONSE_FORMAT_MODE="${OPENAI_RESPONSE_FORMAT_MODE:-disabled}"
export INCLUDE_FINE_TUNED_MODEL=false
export INCLUDE_LOCAL_LORA=false
export EXPERIMENT_SINGLE_VARIANTS="${EXPERIMENT_SINGLE_VARIANTS:-zero_shot,few_shot,cot}"
export EXPERIMENT_METHODS="${EXPERIMENT_METHODS:-single_zero_shot,single_few_shot,single_cot,multi_agent,hybrid}"
if [[ ",$EXPERIMENT_SINGLE_VARIANTS," == *",structured_output,"* ]] && [[ ",$EXPERIMENT_METHODS," != *",single_structured_output,"* ]]; then
  export EXPERIMENT_METHODS="${EXPERIMENT_METHODS},single_structured_output"
fi

SERVER_RUN_ID="vllm_${SAFE_MODEL}_val_test_${LIMIT_TAG}${RUN_SUFFIX}"

nvidia-smi
echo "Node        : $(hostname)"
echo "Model       : $MODEL"
echo "Served name : $SERVED_MODEL_NAME"
echo "Image       : $IMAGE"
echo "Base URL    : $OPENAI_BASE_URL"
echo "Tensor par. : $TENSOR_PARALLEL_SIZE"
echo "Splits      : $SPLITS"
echo "Limit       : $EXPERIMENT_LIMIT"
echo "Run suffix  : ${RUN_SUFFIX:-<none>}"
echo "Methods     : $EXPERIMENT_METHODS"
echo "Single vars : $EXPERIMENT_SINGLE_VARIANTS"
echo "Resp format : $OPENAI_RESPONSE_FORMAT_MODE"

"$CONTAINER_RUNTIME" exec --nv \
  --bind /work:/work \
  --bind /home:/home \
  "$IMAGE" \
  python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --dtype auto \
    --max-model-len "$MAX_MODEL_LEN" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    > "/home/s504297/jobs/logs/${SERVER_RUN_ID}_vllm_server.log" 2>&1 &

SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; rm -rf "$NODE_TMP" 2>/dev/null || true' EXIT

echo "Waiting for vLLM server..."
for attempt in $(seq 1 120); do
  if curl -fsS "${OPENAI_BASE_URL}/models" >/dev/null 2>&1; then
    echo "vLLM is ready."
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "vLLM server exited early. Server log:" >&2
    tail -120 "/home/s504297/jobs/logs/${SERVER_RUN_ID}_vllm_server.log" >&2 || true
    exit 1
  fi
  sleep 5
done

curl -fsS "${OPENAI_BASE_URL}/models"

IFS=',' read -ra SPLIT_LIST <<< "$SPLITS"
for split in "${SPLIT_LIST[@]}"; do
  split="$(printf '%s' "$split" | xargs)"
  export FORMS_PATH="data/splits/${split}.jsonl"
  export EXPERIMENT_RUN_ID="vllm_${SAFE_MODEL}_${split}_${LIMIT_TAG}${RUN_SUFFIX}"

  echo "Running split=$split"
  echo "Run ID: $EXPERIMENT_RUN_ID"
  echo "Forms : $FORMS_PATH"

  uv run python experiments/run_full_experiment.py

  EXPERIMENT_RUN_DIR="outputs/experiments/${EXPERIMENT_RUN_ID}" \
    FORMS_PATH="$FORMS_PATH" \
    uv run python experiments/diagnose_experiment_results.py

  cat "outputs/experiments/${EXPERIMENT_RUN_ID}/manifest.json"
  cat "outputs/experiments/${EXPERIMENT_RUN_ID}/metrics/method_summary.md"
  cat "outputs/experiments/${EXPERIMENT_RUN_ID}/diagnostics/diagnostics.md"
done
