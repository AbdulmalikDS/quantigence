"""Assemble the scale-study points on a consistent subset, with clean VRAM.

The 4B and 14B points come from their own runs (results/scale/<tag>). The 9B
point is computed from the main run's quantigence results restricted to the SAME
standards+risk subset, so all three sit on one curve. Peak VRAM per model is
read from results/scale/vram.json (measured separately on a clean GPU by
scripts/measure_vram.sh), since the harness can only sample one GPU.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]


def nine_b_point() -> dict:
    # 9B point over the full 40-query benchmark (reuses the main quantigence run).
    correct = total = 0
    for f in glob.glob(str(CODE / "results/main/qwen3.5-9b/quantigence__*.json")):
        d = json.loads(Path(f).read_text())
        correct += int(d["correct"]); total += 1
    return {"quantigence": {"n": total,
                            "accuracy": round(correct / total, 4) if total else 0}}


def main() -> None:
    # write the 9B subset point
    out9 = CODE / "results/scale/qwen3.5-9b"
    out9.mkdir(parents=True, exist_ok=True)
    (out9 / "summary.json").write_text(json.dumps(nine_b_point(), indent=2))

    # merge measured VRAM into each scale summary if available
    vram_path = CODE / "results/scale/vram.json"
    vram = json.loads(vram_path.read_text()) if vram_path.exists() else {}
    for tag in ("qwen3.5-4b", "qwen3.5-9b", "qwen3-14b"):
        sp = CODE / f"results/scale/{tag}/summary.json"
        if not sp.exists():
            continue
        s = json.loads(sp.read_text())
        if tag in vram and "quantigence" in s:
            s["quantigence"]["peak_vram_mb"] = vram[tag]
        sp.write_text(json.dumps(s, indent=2))
        acc = s.get("quantigence", {}).get("accuracy")
        v = s.get("quantigence", {}).get("peak_vram_mb")
        print(f"{tag}: acc={acc} vram={v}")


if __name__ == "__main__":
    main()
