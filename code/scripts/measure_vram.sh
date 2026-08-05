#!/usr/bin/env bash
# Measure peak VRAM for each model alone on a clean GPU (default GPU 0), at the
# 16k context used in the experiments. Writes results/scale/vram.json.
# Usage: scripts/measure_vram.sh [gpu]
set -euo pipefail
GPU="${1:-0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p results/scale
declare -A MODELS=(
  [qwen3.5-4b]="models/Qwen3.5-4B-Q4_K_M.gguf"
  [qwen3.5-9b]="models/Qwen3.5-9B-Q4_K_M.gguf"
  [qwen3-14b]="models/Qwen3-14B-Q4_K_M.gguf"
)
echo "{" > results/scale/vram.json
first=1
for tag in qwen3.5-4b qwen3.5-9b qwen3-14b; do
  model="${MODELS[$tag]}"
  [ -f "$model" ] || { echo "skip $tag (missing)"; continue; }
  CUDA_VISIBLE_DEVICES="$GPU" bash scripts/serve.sh "$model" 8090 16384 99 > /tmp/vram_$tag.log 2>&1 &
  pid=$!
  for i in $(seq 1 90); do grep -q "server is listening" /tmp/vram_$tag.log 2>/dev/null && break; sleep 1; done
  sleep 3
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | head -1 | tr -d ' ')
  kill $pid 2>/dev/null || true
  sleep 3
  echo "$tag: ${used} MiB"
  [ $first -eq 0 ] && echo "," >> results/scale/vram.json
  printf '  "%s": %s' "$tag" "$used" >> results/scale/vram.json
  first=0
done
echo "" >> results/scale/vram.json
echo "}" >> results/scale/vram.json
echo "wrote results/scale/vram.json"
