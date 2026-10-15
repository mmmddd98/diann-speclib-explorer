"""Regenerate the README figures from deterministic synthetic data only."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


OUTPUT = Path(__file__).parent / "images"
BLUE = "#4477AA"
ORANGE = "#EE7733"
DARK = "#202938"
GREY = "#7E8794"


def paired_rt_demo() -> None:
    rng = np.random.default_rng(12)
    free_rt = rng.uniform(0, 100, 32_000)
    delta_rt = rng.normal(1.0, 4.0, free_rt.size)
    dimethyl_rt = free_rt + delta_rt

    fig, (paired, delta) = plt.subplots(
        1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.15, 0.85]}
    )
    counts, x_edges, y_edges = np.histogram2d(
        free_rt, dimethyl_rt, bins=90, range=[[0, 100], [0, 100]]
    )
    paired.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_less_equal(counts.T, 0),
        norm=LogNorm(vmin=1, vmax=max(2, counts.max())),
        cmap="Blues",
        shading="auto",
        rasterized=True,
    )
    paired.plot([0, 100], [0, 100], color=DARK, lw=1, ls="--")
    paired.set(
        xlim=(0, 100),
        ylim=(0, 100),
        xlabel="Free N-terminus RT (stored units)",
        ylabel="N-terminal dimethyl RT (stored units)",
        title="A  Matched peptide forms",
    )
    paired.set_aspect("equal", adjustable="box")
    paired.text(0.04, 0.96, "Synthetic example only", transform=paired.transAxes, va="top", color=GREY)

    delta.hist(delta_rt, bins=70, density=True, color=BLUE, alpha=0.25)
    delta.hist(delta_rt, bins=70, density=True, histtype="step", lw=1.8, color=BLUE)
    delta.axvline(0, color=DARK, lw=1, ls="--")
    delta.axvline(np.median(delta_rt), color=ORANGE, lw=1.4)
    delta.set(
        xlabel="ΔRT = dimethyl − free (stored units)",
        ylabel="Probability density",
        title="B  Paired RT difference",
    )
    fig.suptitle("Exact modification pairs", fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT / "demo_paired_rt.svg", bbox_inches="tight")
    plt.close(fig)


def rt_shift_by_length_demo() -> None:
    lengths = np.arange(7, 31)
    median = 1.2 - 0.34 * (lengths - 7) + 0.004 * (lengths - 18) ** 2
    spread = 4.0 + 0.50 * (lengths - 7)

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.fill_between(
        lengths,
        median - spread,
        median + spread,
        color=BLUE,
        alpha=0.17,
        label="Illustrative 10th–90th percentile band",
    )
    ax.plot(lengths, median, color=BLUE, marker="o", markersize=3.5, lw=1.8, label="Illustrative median")
    ax.axhline(0, color=DARK, lw=1, ls="--")
    ax.set(
        xlabel="Peptide length (amino acids)",
        ylabel="RT-tuned − baseline (stored units)",
        title="RT shift by peptide length",
    )
    ax.legend(frameon=False, loc="best")
    ax.text(
        0.99,
        0.03,
        "SIMULATED DATA — not a study result",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=GREY,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT / "demo_rt_shift_by_length.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    paired_rt_demo()
    rt_shift_by_length_demo()


if __name__ == "__main__":
    main()
