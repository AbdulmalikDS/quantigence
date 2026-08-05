"""Figure: main ablation results.

(a) Overall benchmark accuracy by condition.
(b) Accuracy by query category, grouped by condition — shows where the
    multi-agent structure and tools help most.
Reads the per-query result JSONs written by eval.harness.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402
import figstyle  # noqa: E402

CONDITIONS = ["zeroshot", "single_agent", "quantigence"]
CATS = [("standards", "Standards"), ("algorithm", "Algorithm"),
        ("vulnerability", "Vulnerability"), ("risk", "Risk")]


def load_records(result_dir: Path) -> list[dict]:
    recs = []
    for f in result_dir.glob("*.json"):
        if f.name == "summary.json":
            continue
        recs.append(json.loads(f.read_text()))
    return recs


def main(result_dir: str = "results/main/qwen3.5-9b") -> None:
    rd = Path(result_dir)
    recs = load_records(rd)
    if not recs:
        print(f"no records in {rd}; run the harness first")
        return

    # overall + per-category accuracy
    overall = {c: [] for c in CONDITIONS}
    percat = {c: defaultdict(list) for c in CONDITIONS}
    for r in recs:
        c = r["condition"]
        if c not in overall:
            continue
        overall[c].append(int(r["correct"]))
        percat[c][r["category"]].append(int(r["correct"]))

    figstyle.apply()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9),
                                   gridspec_kw={"width_ratios": [1, 1.7]})

    # (a) overall
    xs = np.arange(len(CONDITIONS))
    vals = [np.mean(overall[c]) if overall[c] else 0 for c in CONDITIONS]
    bars = ax1.bar(xs, vals, width=0.62,
                   color=[figstyle.COLORS[c] for c in CONDITIONS])
    for x, v in zip(xs, vals):
        ax1.text(x, v + 0.02, f"{v:.0%}", ha="center", va="bottom", fontsize=10)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(["Zero-shot", "Single\nagent", "Quantigence"], fontsize=9.5)
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("(a) Overall accuracy")

    # (b) per category grouped
    width = 0.26
    x = np.arange(len(CATS))
    for i, c in enumerate(CONDITIONS):
        ys = [np.mean(percat[c][cat]) if percat[c][cat] else 0 for cat, _ in CATS]
        ax2.bar(x + (i - 1) * width, ys, width=width * 0.92,
                color=figstyle.COLORS[c], label=figstyle.LABELS[c])
    ax2.set_xticks(x)
    ax2.set_xticklabels([lbl for _, lbl in CATS])
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("(b) Accuracy by query category")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=9)

    figstyle.save(fig, "fig_results")


if __name__ == "__main__":
    main(*sys.argv[1:])
