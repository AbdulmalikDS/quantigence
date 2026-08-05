"""Download the GGUF models used in the experiments.

Disk-aware: refuses to download if free space would drop below a safety margin.
Usage: python scripts/download_models.py [primary|small|large|all]
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

MODELS = {
    "primary": ("unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-Q4_K_M.gguf", 5.7),
    "small":   ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf", 2.8),
    "large":   ("unsloth/Qwen3-14B-GGUF",  "Qwen3-14B-Q4_K_M.gguf",  9.1),
}
MODEL_DIR = Path("models")
SAFETY_GB = 3.0


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def fetch(key: str) -> Path:
    repo, fname, size_gb = MODELS[key]
    MODEL_DIR.mkdir(exist_ok=True)
    target = MODEL_DIR / fname
    if target.exists():
        print(f"[{key}] already present: {target}")
        return target
    have = free_gb(MODEL_DIR)
    if have - size_gb < SAFETY_GB:
        raise SystemExit(f"[{key}] refusing: need ~{size_gb}GB, only {have:.1f}GB free "
                         f"(margin {SAFETY_GB}GB). Free space or delete another model.")
    print(f"[{key}] downloading {repo}/{fname} (~{size_gb}GB); {have:.1f}GB free")
    path = hf_hub_download(repo_id=repo, filename=fname, local_dir=MODEL_DIR)
    print(f"[{key}] done: {path}")
    return Path(path)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "primary"
    keys = list(MODELS) if which == "all" else [which]
    for k in keys:
        fetch(k)
