"""Figure: poisoning robustness — retrieval vs generation.

(a) Retrieval condition: mean number of poison chunks in the top-k, without
    defenses vs with the source-hierarchy defense (deterministic).
(b) Generation condition: end-to-end attack success rate and utility, without
    defenses vs with all defenses (LLM-judged).
Reads results/robustness/summary.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402
import figstyle  # noqa: E402


def main(path: str = "results/robustness/summary.json") -> None:
    f = Path(path)
    if not f.exists():
        print(f"missing {f}; run eval.robustness first")
        return
    s = json.loads(f.read_text())
    retr, gen = s["retrieval"], s["generation"]

    figstyle.apply()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.9))

    # (a) retrieval: mean poison in top-k
    labels = ["No defense", "Source\nhierarchy"]
    vals = [retr["none"]["mean_poison_in_topk"], retr["source_hierarchy"]["mean_poison_in_topk"]]
    k = retr["none"]["k"]
    cols = [figstyle.COLORS["single_agent"], figstyle.COLORS["quantigence"]]
    x = np.arange(2)
    ax1.bar(x, vals, width=0.6, color=cols)
    for xi, v in zip(x, vals):
        ax1.text(xi, v + 0.05, f"{v:.2f}", ha="center", fontsize=10)
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_ylabel(f"Mean poison chunks in top-{k}")
    ax1.set_ylim(0, k)
    ax1.set_title("(a) Retrieval condition")

    # (b) generation: attack ASR and utility, none vs all
    configs = ["none", "all"]
    nice = {"none": "No defense", "all": "All defenses"}
    asr = [gen[c]["attack_asr"] for c in configs]
    util = [gen[c]["utility"] for c in configs]
    x = np.arange(2); w = 0.36
    ax2.bar(x - w / 2, asr, width=w, color=figstyle.COLORS["single_agent"], label="Attack success")
    ax2.bar(x + w / 2, util, width=w, color=figstyle.COLORS["quantigence"], label="Utility")
    for xi, a in zip(x, asr):
        ax2.text(xi - w / 2, a + 0.02, f"{a:.0%}", ha="center", fontsize=9)
    for xi, u in zip(x, util):
        ax2.text(xi + w / 2, u + 0.02, f"{u:.0%}", ha="center", fontsize=9)
    ax2.set_xticks(x); ax2.set_xticklabels([nice[c] for c in configs])
    ax2.set_ylabel("Rate")
    ax2.set_ylim(0, 1.08)
    ax2.set_title("(b) Generation condition")
    ax2.legend(loc="upper center", fontsize=9, ncol=2, bbox_to_anchor=(0.5, -0.16))

    figstyle.save(fig, "fig_robustness")


if __name__ == "__main__":
    main(*sys.argv[1:])
