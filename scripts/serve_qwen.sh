#!/usr/bin/env bash
# Serve cached Qwen3 weights without changing the existing inference environment.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
INFERENCE_ENV="${INFERENCE_ENV:-$PROJECT_DIR/.venv-serve}"
MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR to your downloaded Qwen3-8B directory}"
MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-21014}"
if [[ ! -x "$INFERENCE_ENV/bin/python" ]]; then
  printf 'Missing server Python: %s/bin/python\nSet INFERENCE_ENV to your vLLM environment.\n' "$INFERENCE_ENV" >&2
  exit 1
fi
if [[ ! -d "$MODEL_DIR" ]]; then
  printf 'Missing model directory: %s\n' "$MODEL_DIR" >&2
  exit 1
fi
SITE_PACKAGES="$("$INFERENCE_ENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Enable the localhost-only cache reset endpoint used between benchmark arms.
export VLLM_SERVER_DEV_MODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cuda_runtime/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cusparse/lib:$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"

cd "$PROJECT_DIR"
exec "$INFERENCE_ENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name "$MODEL_NAME" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN:-16384}" \
  --max-num-seqs "${MAX_NUM_SEQS:-16}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.55}" \
  --enable-prefix-caching \
  --chat-template "$PROJECT_DIR/scripts/qwen3_non_thinking.jinja" \
  --generation-config vllm
