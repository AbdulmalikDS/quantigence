#!/usr/bin/env bash
# Launch llama-server for a Quantigence model.
# Usage: scripts/serve.sh <model.gguf> [port] [ctx] [gpu_layers]
#
# Runs on GPU 0 with an enforced context size so peak VRAM is measurable against
# the paper's 8GB budget. Tool calling needs --jinja; Qwen3.5 thinking is toggled
# per-request by the client (enable_thinking:false), not here.
set -euo pipefail

MODEL="${1:?path to .gguf required}"
PORT="${2:-8080}"
CTX="${3:-8192}"
NGL="${4:-99}"   # offload all layers; 99 = as many as fit

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/vendor/llama.cpp/build/bin/llama-server"
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec "$BIN" \
  --model "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --ctx-size "$CTX" \
  --n-gpu-layers "$NGL" \
  --jinja \
  --temp 0.0 \
  --no-warmup
