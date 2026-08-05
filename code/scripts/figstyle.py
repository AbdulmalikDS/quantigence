"""Shared matplotlib style for publication figures.

One clean, serif, vector-first style so every figure in the paper reads as one
system. Colours are the CVD-validated Okabe-Ito subset (checked with the dataviz
validator): assigned to conditions in fixed order, never cycled.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Condition identity colours (fixed order; CVD-safe).
COLORS = {
    "zeroshot": "#0072B2",       # blue
    "single_agent": "#E69F00",   # orange
    "quantigence": "#009E73",    # green (the system under study)
}
LABELS = {
    "zeroshot": "Zero-shot",
    "single_agent": "Single agent + tools",
    "quantigence": "Quantigence (multi-agent)",
}
SEQ = "#1b6ca8"   # single-hue sequential base for heatmaps
INK = "#222222"
MUTED = "#6b6b6b"
FIG = Path(__file__).resolve().parents[1].parent / "figures"


def apply() -> None:
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.labelcolor": INK,
        "legend.frameon": False,
        "legend.fontsize": 10,
    })


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"wrote figures/{name}.pdf and .png")
