"""Figure: QARS temporal-urgency curve and composite sensitivity heatmap.

Left  - the sigmoid urgency factor T(a) vs the Mosca ratio r, for several alpha,
        showing the cliff at r = 1.
Right - the composite QARS over (urgency ratio, exploitability) at fixed
        sensitivity, showing how time dominates but the other factors shift it.
Replaces v1's hand-drawn landscape.png with a generated, exact figure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt  # noqa: E402
from quantigence.qars import Asset, qars, temporal_urgency  # noqa: E402
import figstyle  # noqa: E402


def main() -> None:
    figstyle.apply()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.9))

    # --- left: urgency sigmoid for several steepness alphas ---
    r = np.linspace(0, 2, 400)
    for alpha, shade in [(4, "#9ecae1"), (10, figstyle.SEQ), (20, "#08519c")]:
        T = [temporal_urgency(Asset("x", rr, 0, 1, 0, 0), alpha=alpha) for rr in r]
        ax1.plot(r, T, color=shade, lw=2, label=fr"$\alpha={alpha}$")
    ax1.axvline(1.0, color=figstyle.MUTED, lw=1, ls="--")
    ax1.text(1.02, 0.06, "Mosca cliff\n$X+Y=Z$", color=figstyle.MUTED, fontsize=9)
    ax1.set_xlabel(r"Urgency ratio $r=(X+Y)/Z$")
    ax1.set_ylabel(r"Temporal urgency $T(a)$")
    ax1.set_title("(a) Urgency saturation")
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(loc="upper left")

    # --- right: composite QARS heatmap over ratio x exploitability, S fixed ---
    ratios = np.linspace(0, 2, 120)
    expl = np.linspace(0, 1, 120)
    S_fixed = 0.7
    grid = np.zeros((len(expl), len(ratios)))
    for i, e in enumerate(expl):
        for j, rr in enumerate(ratios):
            # Encode a target ratio via X+Y with Z=1.
            grid[i, j] = qars(Asset("x", migration_time=rr, shelf_life=0.0,
                                    collapse_time=1.0, sensitivity=S_fixed,
                                    exploitability=e))
    im = ax2.imshow(grid, origin="lower", aspect="auto", cmap="viridis",
                    extent=[0, 2, 0, 1], vmin=0, vmax=1)
    ax2.axvline(1.0, color="white", lw=1, ls="--", alpha=0.8)
    ax2.set_xlabel(r"Urgency ratio $r$")
    ax2.set_ylabel(r"Exploitability $E(a)$")
    ax2.set_title(fr"(b) Composite QARS  ($S={S_fixed}$)")
    ax2.grid(False)
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label(r"$R_{\mathrm{QARS}}$")

    figstyle.save(fig, "fig_qars")


if __name__ == "__main__":
    main()
