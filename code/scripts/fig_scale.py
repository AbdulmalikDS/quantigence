"""Figure: model-scale ablation under the 8GB budget.

(a) Quantigence accuracy vs model size (4B / 9B / 14B).
(b) Peak VRAM vs model size, with the 8GB budget line — showing the 4B and 9B
    models fit while the 14B exceeds it.
Reads results/scale/<tag>/summary.json (quantigence condition).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402
import figstyle  # noqa: E402

# (tag, display, param_billions)
MODELS = [("qwen3.5-4b", "Qwen3.5-4B", 4.0),
          ("qwen3.5-9b", "Qwen3.5-9B", 9.0),
          ("qwen3-14b", "Qwen3-14B", 14.8)]
BUDGET_MB = 8192


def main(scale_dir: str = "results/scale") -> None:
    sd = Path(scale_dir)
    params, accs, vram, names = [], [], [], []
    for tag, disp, p in MODELS:
        f = sd / tag / "summary.json"
        if not f.exists():
            print(f"missing {f}; skipping")
            continue
        s = json.loads(f.read_text()).get("quantigence", {})
        params.append(p)
        accs.append(s.get("accuracy", 0))
        vram.append(s.get("peak_vram_mb") or 0)
        names.append(disp)
    if not params:
        print("no scale results yet")
        return

    figstyle.apply()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.9))
    x = np.arange(len(names))

    # 4B/9B are the same (Qwen3.5) generation; the 14B is an earlier generation,
    # shown in a different colour to flag that it is not a clean size comparison.
    acc_cols = [figstyle.COLORS["quantigence"] if "3.5" in nm else figstyle.COLORS["single_agent"]
                for nm in names]
    ax1.bar(x, accs, width=0.6, color=acc_cols)
    for xi, a in zip(x, accs):
        ax1.text(xi, a + 0.02, f"{a:.0%}", ha="center", fontsize=10)
    ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=9.5)
    ax1.set_ylabel("Accuracy (40 queries)")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("(a) Accuracy vs. model scale")

    cols = [figstyle.COLORS["quantigence"] if v <= BUDGET_MB else figstyle.COLORS["single_agent"]
            for v in vram]
    ax2.bar(x, [v / 1024 for v in vram], width=0.6, color=cols)
    ax2.axhline(BUDGET_MB / 1024, color=figstyle.MUTED, ls="--", lw=1.2)
    ax2.text(len(names) - 0.5, BUDGET_MB / 1024 + 0.15, "8 GB budget",
             ha="right", color=figstyle.MUTED, fontsize=9)
    for xi, v in zip(x, vram):
        ax2.text(xi, v / 1024 + 0.15, f"{v/1024:.1f}", ha="center", fontsize=9.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=9.5)
    ax2.set_ylabel("Peak VRAM (GB)")
    ax2.set_title("(b) Memory footprint")

    figstyle.save(fig, "fig_scale")


if __name__ == "__main__":
    main(*sys.argv[1:])
