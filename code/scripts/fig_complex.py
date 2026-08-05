"""Figure: complex-task evaluation — where multi-agent decomposition helps.

(a) Mean rubric coverage by condition (fraction of required sub-points addressed).
(b) Mean judge quality by condition (comprehensiveness + correctness, [0,1]).
Reads results/complex/summary.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402
import figstyle  # noqa: E402

CONDS = ["zeroshot", "single_agent", "quantigence"]
LABELS = ["Zero-shot", "Single\nagent", "Quantigence"]


def main(path: str = "results/complex/summary.json") -> None:
    f = Path(path)
    if not f.exists():
        print(f"missing {f}; run eval.complex_eval first")
        return
    s = json.loads(f.read_text())
    conds = [c for c in CONDS if c in s]
    labels = [LABELS[CONDS.index(c)] for c in conds]

    figstyle.apply()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.9))
    x = np.arange(len(conds))
    cols = [figstyle.COLORS[c] for c in conds]

    cov = [s[c]["mean_coverage"] for c in conds]
    ax1.bar(x, cov, width=0.6, color=cols)
    for xi, v in zip(x, cov):
        ax1.text(xi, v + 0.02, f"{v:.0%}", ha="center", fontsize=10)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9.5)
    ax1.set_ylabel("Mean rubric coverage")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("(a) Coverage of required sub-points")

    qual = [s[c]["mean_quality"] for c in conds]
    ax2.bar(x, qual, width=0.6, color=cols)
    for xi, v in zip(x, qual):
        ax2.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9.5)
    ax2.set_ylabel("Mean judge quality")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("(b) Judged answer quality")

    figstyle.save(fig, "fig_complex")


if __name__ == "__main__":
    main(*sys.argv[1:])
