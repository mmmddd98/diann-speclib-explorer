#!/usr/bin/env python3
"""Create exact paired RT and dimethyl-only density plots from DIA-NN libraries.

Pairs are joined with Apache Arrow on a canonical modified sequence. For the
N-terminal comparison, only the leading UniMod:510 is removed; every other
modification and the decoy flag must match exactly. This keeps charge-collapsed
peptide forms while preventing mismatched oxidation, acetylation, or lysine
states from entering the pair.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


COLORS = {
    "blue": "#4477AA",
    "orange": "#EE7733",
    "green": "#228833",
    "magenta": "#AA3377",
    "neutral": "#7E8794",
    "dark": "#202938",
    "grid": "#D9DEE7",
    "blue_light": "#EAF1F8",
}
LIBRARIES = ("baseline", "tuned_rt", "tuned_rt_fr")
LIBRARY_LABELS = {
    "baseline": "Baseline",
    "tuned_rt": "RT-tuned",
    "tuned_rt_fr": "RT + fragment-tuned",
}
RT_COLUMNS = {
    "baseline": "rt_baseline",
    "tuned_rt": "rt_tuned_rt",
    "tuned_rt_fr": "rt_tuned_rt_fr",
}
SITE_GROUPS = (
    ("N-terminal only", "N-terminal dimethyl", COLORS["blue"]),
    ("Lysine only", "Lysine dimethyl", COLORS["orange"]),
    ("N-terminus + lysine", "N-terminus + lysine dimethyl", COLORS["magenta"]),
)
INPUT_COLUMNS = [
    "modified_sequence",
    "decoy",
    "has_unimod_510",
    "has_unimod_510_nterm",
    "has_unimod_510_k",
    "rt",
]


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "text.color": COLORS["dark"],
            "axes.labelcolor": COLORS["dark"],
            "axes.edgecolor": COLORS["dark"],
            "xtick.color": COLORS["dark"],
            "ytick.color": COLORS["dark"],
            "axes.linewidth": 0.7,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def configure_logger(out_dir: Path) -> logging.Logger:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"modification_pairs_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.log"
    logger = logging.getLogger("plot_diann_modification_pairs")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def peak_rss_gib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024**2) if sys.platform != "darwin" else value / (1024**3)


def save_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def save_figure(fig, prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def numpy_column(table: pa.Table, name: str, dtype=None) -> np.ndarray:
    arr = table[name].combine_chunks().to_numpy(zero_copy_only=False)
    return arr.astype(dtype, copy=False) if dtype is not None else arr


def numpy_arrow_array(arr, dtype=None) -> np.ndarray:
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    values = arr.to_numpy(zero_copy_only=False)
    return values.astype(dtype, copy=False) if dtype is not None else values


def read_library(data_dir: Path, label: str, logger: logging.Logger) -> pa.Table:
    path = data_dir / f"library={label}" / "rt.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    logger.info("Reading %s (%0.2f GB compressed)", path, path.stat().st_size / 1e9)
    table = pq.read_table(path, columns=INPUT_COLUMNS, use_threads=True)
    if table.num_rows == 0:
        raise ValueError(f"{label}: empty RT table")
    for name in ("modified_sequence", "decoy", "has_unimod_510_nterm", "has_unimod_510_k", "rt"):
        if table[name].null_count:
            raise ValueError(f"{label}: {name} contains null values")
    if not pc.all(pc.is_finite(table["rt"])).as_py():
        raise ValueError(f"{label}: RT contains non-finite values")
    logger.info("Loaded %s: %s rows; peak RSS %.1f GiB", label, f"{table.num_rows:,}", peak_rss_gib())
    return table


def make_nterm_pairs(table: pa.Table, label: str, out_dir: Path, logger: logging.Logger):
    seq = table["modified_sequence"]
    nterm = table["has_unimod_510_nterm"]
    # Remove only a leading heavy-dimethyl token. The bare counterpart must
    # equal this exact canonical key; e.g. an acetylated N-terminus cannot match.
    pair_key = pc.replace_substring_regex(
        seq, pattern=r"^\(UniMod:510\)", replacement="", max_replacements=1
    )
    dim_mask = nterm
    free_mask = pc.and_(pc.invert(nterm), pc.equal(seq, pair_key))
    dim = table.filter(dim_mask).select(["decoy", "has_unimod_510_k", "rt"])
    dim = dim.append_column("pair_key", pc.filter(pair_key, dim_mask))
    free = table.filter(free_mask).select(["decoy", "has_unimod_510_k", "rt"])
    free = free.append_column("pair_key", pc.filter(pair_key, free_mask))
    dim = dim.select(["pair_key", "decoy", "has_unimod_510_k", "rt"]).rename_columns(
        ["pair_key", "decoy", "has_k", "rt_dimethyl"]
    )
    free = free.select(["pair_key", "decoy", "has_unimod_510_k", "rt"]).rename_columns(
        ["pair_key", "decoy", "has_k_free", "rt_free"]
    )
    logger.info(
        "%s candidates: %s N-terminal dimethyl forms; %s exact bare-N-terminal counterparts",
        label,
        f"{dim.num_rows:,}",
        f"{free.num_rows:,}",
    )
    pairs = dim.join(free, keys=["pair_key", "decoy"], join_type="inner", use_threads=True)
    if pairs.num_rows > dim.num_rows:
        raise ValueError(f"{label}: normalized peptide keys create ambiguous many-to-many pairs")
    if pairs.num_rows == 0:
        raise ValueError(f"{label}: no exact N-terminal dimethyl/free pairs found")
    if not pc.all(pc.equal(pairs["has_k"], pairs["has_k_free"])).as_py():
        raise ValueError(f"{label}: an N-terminal pair changes lysine dimethylation state")
    pairs = pairs.select([name for name in pairs.column_names if name != "has_k_free"])
    pairs = pairs.append_column("delta_rt", pc.subtract(pairs["rt_dimethyl"], pairs["rt_free"]))
    output = out_dir / f"paired_nterm_rt_{label}.parquet"
    output_table = pairs.select(["pair_key", "decoy", "has_k", "rt_free", "rt_dimethyl", "delta_rt"]).replace_schema_metadata(
        {b"pairing_rule": b"strip leading UniMod:510 only; bare key, decoy and all other modifications identical"}
    )
    pq.write_table(output_table, output, compression="zstd", row_group_size=250_000)
    logger.info("Wrote %s pairs to %s; peak RSS %.1f GiB", f"{pairs.num_rows:,}", output, peak_rss_gib())
    return pairs, output


def k_site_coverage(table: pa.Table) -> dict:
    seq = table["modified_sequence"]
    k_sites = pc.count_substring_regex(seq, pattern="K")
    k_dimethyl_sites = pc.count_substring(seq, pattern="K(UniMod:510)")
    k_sites_np = numpy_arrow_array(k_sites)
    k_mod_np = numpy_arrow_array(k_dimethyl_sites)
    has_k = k_sites_np > 0
    return {
        "forms_with_lysine": int(has_k.sum()),
        "forms_with_any_dimethylated_lysine": int((k_mod_np > 0).sum()),
        "forms_with_lysine_but_no_dimethylated_lysine": int((has_k & (k_mod_np == 0)).sum()),
        "forms_with_partially_dimethylated_lysines": int(((k_mod_np > 0) & (k_mod_np < k_sites_np)).sum()),
        "lysine_sites": int(k_sites_np.sum()),
        "dimethylated_lysine_sites": int(k_mod_np.sum()),
    }


def quantile_summary(values: np.ndarray) -> dict:
    q05, q25, q50, q75, q95 = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n": int(values.size),
        "mean_delta_rt": float(np.mean(values, dtype=np.float64)),
        "median_delta_rt": float(q50),
        "sd_delta_rt": float(np.std(values, dtype=np.float64)),
        "positive_delta_fraction": float(np.mean(values > 0)),
        "q05": float(q05),
        "q25": float(q25),
        "q75": float(q75),
        "q95": float(q95),
    }


def smooth_density(values: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts, _ = np.histogram(values, bins=edges)
    density = counts.astype(np.float64)
    if density.sum():
        density /= density.sum() * (edges[1] - edges[0])
    kernel_x = np.arange(-12, 13, dtype=float)
    kernel = np.exp(-0.5 * (kernel_x / 2.2) ** 2)
    kernel /= kernel.sum()
    density = np.convolve(density, kernel, mode="same")
    return (edges[:-1] + edges[1:]) / 2, density


def plot_single_library_pairs(pairs: pa.Table, label: str, out_dir: Path, stats: dict) -> None:
    configure_plot_style()
    free = numpy_column(pairs, "rt_free", np.float64)
    dimethyl = numpy_column(pairs, "rt_dimethyl", np.float64)
    delta = numpy_column(pairs, "delta_rt", np.float64)
    has_k = numpy_column(pairs, "has_k", bool)
    bounds = (
        float(min(free.min(), dimethyl.min())),
        float(max(free.max(), dimethyl.max())),
    )
    fig, (scatter, distribution) = plt.subplots(
        1, 2, figsize=(7.1, 3.35), gridspec_kw={"width_ratios": [1.15, 0.85]}
    )
    hist, xedges, yedges = np.histogram2d(free, dimethyl, bins=240, range=[bounds, bounds])
    mesh = scatter.pcolormesh(
        xedges,
        yedges,
        np.ma.masked_less_equal(hist.T, 0),
        norm=LogNorm(vmin=1, vmax=max(1.0, float(hist.max()))),
        cmap="Blues",
        shading="auto",
        rasterized=True,
    )
    if has_k.any():
        khist, xe, ye = np.histogram2d(free[has_k], dimethyl[has_k], bins=240, range=[bounds, bounds])
        positive = khist[khist > 0]
        if positive.size > 8:
            levels = np.unique(np.quantile(positive, [0.65, 0.88, 0.98]))
            levels = levels[levels > 0]
            if levels.size:
                scatter.contour(
                    (xe[:-1] + xe[1:]) / 2,
                    (ye[:-1] + ye[1:]) / 2,
                    khist.T,
                    levels=levels,
                    colors=[COLORS["orange"]],
                    linewidths=0.8,
                )
    scatter.plot(bounds, bounds, color=COLORS["dark"], lw=0.8, ls=(0, (3, 2)))
    scatter.set(xlim=bounds, ylim=bounds, xlabel="N-terminally free RT (stored units)")
    scatter.set_ylabel("N-terminal UniMod:510 RT (stored units)")
    scatter.set_aspect("equal", adjustable="box")
    scatter.set_title("A  Matched peptide-form RTs", loc="left", fontweight="bold")
    scatter.text(
        0.97,
        0.03,
        "Colour: log pair count",
        transform=scatter.transAxes,
        va="bottom",
        ha="right",
        fontsize=6.5,
        color=COLORS["neutral"],
    )
    scatter.text(
        0.04,
        0.96,
        f"n = {len(delta):,}\nmedian ΔRT = {np.median(delta):+.2f}",
        transform=scatter.transAxes,
        va="top",
        color=COLORS["dark"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2},
    )
    finite = np.isfinite(delta)
    dlim = tuple(float(x) for x in np.quantile(delta[finite], [0.0025, 0.9975]))
    if dlim[0] == dlim[1]:
        pad = max(abs(dlim[0]) * 0.02, 0.5)
        dlim = (dlim[0] - pad, dlim[1] + pad)
    edges = np.linspace(dlim[0], dlim[1], 450)
    group_spec = [
        (~has_k, "No lysine in peptide", COLORS["blue"]),
        (has_k, "Lysine dimethylated", COLORS["orange"]),
    ]
    for mask, name, color in group_spec:
        values = delta[mask]
        if values.size == 0:
            continue
        median = float(np.median(values))
        if np.min(values) == np.max(values):
            distribution.axvline(
                median,
                color=color,
                lw=1.4,
                label=f"{name}: ΔRT = {median:+.2f} (n={len(values):,})",
            )
        else:
            xx, density = smooth_density(values, edges)
            distribution.fill_between(xx, density, color=color, alpha=0.16)
            distribution.plot(xx, density, color=color, lw=1.25, label=f"{name} (n={len(values):,})")
            distribution.axvline(median, color=color, lw=0.9, ls=(0, (3, 2)))
        stats["by_lysine_context"][name] = quantile_summary(values)
    distribution.axvline(0, color=COLORS["dark"], lw=0.8, ls=(0, (3, 2)))
    distribution.set_xlim(dlim)
    distribution.set_xlabel("ΔRT = N-terminal dimethyl − free (stored units)")
    if np.min(delta) == np.max(delta):
        distribution.set_ylabel("")
        distribution.set_yticks([])
        distribution.set_ylim(0, 1)
    else:
        distribution.set_ylabel("Probability density")
    distribution.set_title("B  Paired ΔRT by lysine state", loc="left", fontweight="bold")
    distribution.legend(frameon=False, loc="upper right")
    distribution.grid(True, axis="y")
    fig.suptitle(f"{LIBRARY_LABELS[label]}: paired N-terminal dimethyl RT", fontsize=10, fontweight="bold")
    pairing_note = (
        "Baseline: all matched forms have exactly the same stored RT (ΔRT = 0). "
        if np.min(delta) == np.max(delta)
        else ""
    )
    fig.text(
        0.01,
        0.015,
        "Pairs differ only by the leading N-terminal UniMod:510; K state and all other modifications are held fixed. "
        "Orange contours mark pairs with dimethylated lysine. " + pairing_note,
        fontsize=6.4,
        color=COLORS["neutral"],
    )
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.23, top=0.82, wspace=0.43)
    save_figure(fig, out_dir / "figures" / f"nterm_pairs_{label}")


def plot_combined_pairs(pairs: pa.Table, out_dir: Path, stats: dict) -> None:
    configure_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.55), sharex=True, sharey=True)
    common_n = pairs.num_rows
    bounds = (float("inf"), float("-inf"))
    arrays = {}
    for label in LIBRARIES:
        free = numpy_column(pairs, f"rt_free_{label}", np.float64)
        dim = numpy_column(pairs, f"rt_dimethyl_{label}", np.float64)
        arrays[label] = (free, dim)
        bounds = (min(bounds[0], float(free.min()), float(dim.min())), max(bounds[1], float(free.max()), float(dim.max())))
    for ax, label in zip(axes, LIBRARIES):
        free, dim = arrays[label]
        hist, xe, ye = np.histogram2d(free, dim, bins=190, range=[bounds, bounds])
        ax.pcolormesh(
            xe,
            ye,
            np.ma.masked_less_equal(hist.T, 0),
            norm=LogNorm(vmin=1, vmax=max(1.0, float(hist.max()))),
            cmap="Blues",
            shading="auto",
            rasterized=True,
        )
        ax.plot(bounds, bounds, color=COLORS["dark"], lw=0.7, ls=(0, (3, 2)))
        delta = dim - free
        title = LIBRARY_LABELS[label]
        if label == "tuned_rt_fr" and stats.get("tuned_rt_and_tuned_rt_fr_identical_exactly"):
            title += "\n(same RT values as RT-tuned)"
        else:
            title += f"\nmedian ΔRT = {np.median(delta):+.2f}"
        ax.set_title(title, loc="left", fontsize=8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
        ax.set_xlabel("Free N-term RT")
    axes[0].set_ylabel("N-term dimethyl RT (stored units)")
    fig.suptitle("The same N-terminally paired peptide forms across libraries", fontsize=10, fontweight="bold")
    fig.text(0.01, -0.02, f"Three-way shared pairs, n = {common_n:,}; each pair differs only by N-terminal UniMod:510.", fontsize=7, color=COLORS["neutral"])
    save_figure(fig, out_dir / "figures" / "nterm_pairs_all_libraries")

    fig, ax = plt.subplots(figsize=(7.1, 3.2))
    colors = {"baseline": COLORS["blue"], "tuned_rt": COLORS["orange"], "tuned_rt_fr": COLORS["green"]}
    ax.axvline(0, color=COLORS["dark"], lw=0.8, ls=(0, (3, 2)), zorder=1)
    all_delta = np.concatenate([arrays[label][1] - arrays[label][0] for label in LIBRARIES])
    dlim = tuple(float(x) for x in np.quantile(all_delta, [0.0025, 0.9975]))
    if dlim[0] == dlim[1]:
        pad = max(abs(dlim[0]) * 0.02, 0.5)
        dlim = (dlim[0] - pad, dlim[1] + pad)
    edges = np.linspace(dlim[0], dlim[1], 500)
    drawn_duplicate = False
    for label in LIBRARIES:
        if label == "tuned_rt_fr" and stats.get("tuned_rt_and_tuned_rt_fr_identical_exactly"):
            drawn_duplicate = True
            continue
        delta = arrays[label][1] - arrays[label][0]
        if np.min(delta) == np.max(delta):
            ax.axvline(
                delta[0],
                color=colors[label],
                lw=1.5,
                zorder=3,
                label=f"{LIBRARY_LABELS[label]}: all ΔRT = {delta[0]:+.2f}",
            )
        else:
            x, density = smooth_density(delta, edges)
            ax.plot(x, density, color=colors[label], lw=1.5, label=LIBRARY_LABELS[label])
            ax.fill_between(x, density, color=colors[label], alpha=0.10)
        stats["common_pairs"][label]["delta_rt"] = quantile_summary(delta)
    if drawn_duplicate:
        fig.text(
            0.01,
            0.015,
            "RT-tuned and RT + fragment-tuned curves are exactly identical.",
            va="bottom",
            ha="left",
            fontsize=7,
            color=COLORS["neutral"],
        )
    ax.set(xlim=dlim, xlabel="ΔRT = N-terminal dimethyl − free (stored units)", ylabel="Probability density")
    ax.set_title("Paired N-terminal dimethyl shift on the same forms in all libraries", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(True, axis="y")
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.22, top=0.82)
    save_figure(fig, out_dir / "figures" / "nterm_delta_density_all_libraries")


def density_counts(table: pa.Table, edges: np.ndarray) -> dict[str, np.ndarray]:
    nterm = table["has_unimod_510_nterm"]
    lysine = table["has_unimod_510_k"]
    decoy = table["decoy"]
    target = pc.invert(decoy)
    groups = {
        "N-terminal only": pc.and_(pc.and_(nterm, pc.invert(lysine)), target),
        "Lysine only": pc.and_(pc.and_(pc.invert(nterm), lysine), target),
        "N-terminus + lysine": pc.and_(pc.and_(nterm, lysine), target),
    }
    rt = table["rt"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    result = {}
    for name, mask in groups.items():
        mask_np = numpy_arrow_array(mask)
        values = rt[mask_np]
        counts, _ = np.histogram(values, bins=edges)
        result[name] = counts
    return result


def smooth_hist_counts(counts: np.ndarray, bin_width: float) -> np.ndarray:
    density = counts.astype(np.float64)
    if density.sum():
        density /= density.sum() * bin_width
    grid = np.arange(-12, 13, dtype=float)
    kernel = np.exp(-0.5 * (grid / 2.2) ** 2)
    kernel /= kernel.sum()
    return np.convolve(density, kernel, mode="same")


def plot_single_dimethyl_density(label: str, counts: dict, centers: np.ndarray, out_dir: Path) -> None:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(7.1, 3.35))
    for group, legend, color in SITE_GROUPS:
        density = smooth_hist_counts(counts[group], centers[1] - centers[0])
        if counts[group].sum() == 0:
            continue
        ax.plot(centers, density, color=color, lw=1.5, label=f"{legend} (n={counts[group].sum():,})")
        ax.fill_between(centers, density, color=color, alpha=0.12)
    ax.set_xlabel("Predicted RT (stored DIA-NN units)")
    ax.set_ylabel("Probability density")
    ax.set_title(f"{LIBRARY_LABELS[label]}: dimethylated peptide forms only", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(True, axis="y")
    fig.text(0.01, -0.025, "All plotted forms contain UniMod:510; curves separate N-terminal and lysine sites.", fontsize=7, color=COLORS["neutral"])
    save_figure(fig, out_dir / "figures" / f"dimethyl_rt_density_{label}")


def plot_combined_dimethyl_density(hist_by_library: dict, centers: np.ndarray, stats: dict, out_dir: Path) -> None:
    configure_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.55), sharex=True, sharey=True)
    colors = {"baseline": COLORS["blue"], "tuned_rt": COLORS["orange"], "tuned_rt_fr": COLORS["green"]}
    for ax, (group, title, _color) in zip(axes, SITE_GROUPS):
        for label in LIBRARIES:
            if label == "tuned_rt_fr" and stats.get("tuned_rt_and_tuned_rt_fr_identical_exactly"):
                continue
            counts = hist_by_library[label][group]
            density = smooth_hist_counts(counts, centers[1] - centers[0])
            ax.plot(centers, density, lw=1.35, color=colors[label], label=LIBRARY_LABELS[label])
        ax.set_title(title, loc="left")
        ax.set_xlabel("Predicted RT (stored units)")
        ax.grid(True, axis="y")
    axes[0].set_ylabel("Probability density")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.55, 1.02))
    if stats.get("tuned_rt_and_tuned_rt_fr_identical_exactly"):
        fig.text(0.01, -0.025, "All plotted forms contain UniMod:510. The RT + fragment-tuned curve exactly overlaps RT-tuned.", fontsize=7, color=COLORS["neutral"])
    fig.suptitle("Dimethylated RT distributions by site across libraries", y=1.08, fontsize=10, fontweight="bold")
    save_figure(fig, out_dir / "figures" / "dimethyl_rt_density_all_libraries")


def plot_followup_common_form_patterns(common_path: Path, rt_range: tuple[float, float], out_dir: Path, logger: logging.Logger) -> dict:
    """Plot how RT tuning shifts depend on modification context, RT, and length.

    Read the common three-library table in batches. Target forms only are used;
    histograms and length-bin quantiles are accumulated without a dataframe join.
    """
    configure_plot_style()
    if not common_path.is_file():
        raise FileNotFoundError(
            f"Common paired-form table not found: {common_path}. Run plot_diann_rt_comparison.py first."
        )
    pf = pq.ParquetFile(common_path)
    columns = ["modified_sequence", "decoy", "has_unimod_510_nterm", "has_unimod_510_k", "rt_baseline", "rt_tuned_rt"]
    logger.info("Reading shared forms in batches for follow-up plots: %s", common_path)
    # First pass finds data-driven delta bounds and exact per-length values.
    min_delta, max_delta = float("inf"), float("-inf")
    length_deltas: dict[int, list[np.ndarray]] = {}
    category_counts = {name: 0 for name in ("Neither", "N-terminus", "Lysine", "Both")}
    n_target = 0
    for batch in pf.iter_batches(batch_size=500_000, columns=columns, use_threads=True):
        decoy = numpy_arrow_array(batch.column("decoy"), bool)
        target = ~decoy
        if not target.any():
            continue
        nterm = numpy_arrow_array(batch.column("has_unimod_510_nterm"), bool)
        has_k = numpy_arrow_array(batch.column("has_unimod_510_k"), bool)
        delta = numpy_arrow_array(batch.column("rt_tuned_rt"), np.float64) - numpy_arrow_array(batch.column("rt_baseline"), np.float64)
        delta = delta[target]
        nterm, has_k = nterm[target], has_k[target]
        n_target += int(delta.size)
        min_delta = min(min_delta, float(delta.min()))
        max_delta = max(max_delta, float(delta.max()))
        masks = {
            "Neither": ~nterm & ~has_k,
            "N-terminus": nterm & ~has_k,
            "Lysine": ~nterm & has_k,
            "Both": nterm & has_k,
        }
        for name, mask in masks.items():
            category_counts[name] += int(mask.sum())
        # Strip UniMod annotations to count amino acids, not modification text.
        seq = pc.replace_substring_regex(batch.column("modified_sequence"), pattern=r"\(UniMod:\d+\)", replacement="")
        lengths = numpy_arrow_array(pc.utf8_length(seq), np.int16)[target]
        for length in np.unique(lengths):
            selected = lengths == length
            length_deltas.setdefault(int(length), []).append(delta[selected].copy())

    if n_target == 0 or not np.isfinite(min_delta + max_delta):
        raise ValueError("No finite target forms in common paired-form table")
    pad = max((max_delta - min_delta) * 0.01, 0.05)
    d_edges = np.linspace(min_delta - pad, max_delta + pad, 501)
    group_counts = {name: np.zeros(len(d_edges) - 1, dtype=np.int64) for name in category_counts}
    rt_edges = np.linspace(rt_range[0], rt_range[1], 251)
    shift_rt_hist = np.zeros((250, 300), dtype=np.int64)
    shift_edges = np.linspace(min_delta - pad, max_delta + pad, 301)
    # Second pass accumulates full-data distributions and an RT-position heatmap.
    for batch in pf.iter_batches(batch_size=500_000, columns=columns, use_threads=True):
        decoy = numpy_arrow_array(batch.column("decoy"), bool)
        target = ~decoy
        nterm = numpy_arrow_array(batch.column("has_unimod_510_nterm"), bool)
        has_k = numpy_arrow_array(batch.column("has_unimod_510_k"), bool)
        baseline = numpy_arrow_array(batch.column("rt_baseline"), np.float64)
        tuned = numpy_arrow_array(batch.column("rt_tuned_rt"), np.float64)
        delta = tuned - baseline
        masks = {
            "Neither": ~nterm & ~has_k,
            "N-terminus": nterm & ~has_k,
            "Lysine": ~nterm & has_k,
            "Both": nterm & has_k,
        }
        for name, mask in masks.items():
            group_counts[name] += np.histogram(delta[target & mask], bins=d_edges)[0]
        avg_rt = (baseline + tuned) / 2
        shift_rt_hist += np.histogram2d(
            avg_rt[target], delta[target], bins=(rt_edges, shift_edges)
        )[0].astype(np.int64)

    # Modification-context density of the paired tuning shift.
    fig, ax = plt.subplots(figsize=(7.1, 3.45))
    palette = {"Neither": COLORS["neutral"], "N-terminus": COLORS["blue"], "Lysine": COLORS["orange"], "Both": COLORS["magenta"]}
    centers = (d_edges[:-1] + d_edges[1:]) / 2
    for name in ("Neither", "N-terminus", "Lysine", "Both"):
        counts = group_counts[name]
        if counts.sum() == 0:
            continue
        density = smooth_hist_counts(counts, centers[1] - centers[0])
        ax.plot(centers, density, color=palette[name], lw=1.45, label=f"{name} (n={counts.sum():,})")
    ax.axvline(0, color=COLORS["dark"], lw=0.8, ls=(0, (3, 2)))
    ax.set(xlabel="RT-tuned − baseline (stored units)", ylabel="Probability density")
    ax.set_title("Tuning shift by dimethylation site", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=2)
    ax.grid(True, axis="y")
    fig.text(0.01, -0.02, "Three-library shared target forms; density curves compare modification contexts, not independent experiments.", fontsize=6.8, color=COLORS["neutral"])
    save_figure(fig, out_dir / "figures" / "tuning_shift_by_modification_site")

    # 2D count map shows whether tuning shifts vary along the RT range.
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    positive = shift_rt_hist[shift_rt_hist > 0]
    mesh = ax.pcolormesh(
        rt_edges, shift_edges,
        np.ma.masked_less_equal(shift_rt_hist.T, 0),
        norm=LogNorm(vmin=1, vmax=max(1.0, float(positive.max()) if positive.size else 1.0)),
        cmap="Blues", shading="auto", rasterized=True,
    )
    fig.colorbar(mesh, ax=ax, label="Peptide-form count (log scale)")
    ax.axhline(0, color=COLORS["dark"], lw=0.8, ls=(0, (3, 2)))
    ax.set(xlabel="Mean of baseline and RT-tuned RT (stored units)", ylabel="RT-tuned − baseline (stored units)")
    ax.set_title("Tuning shift across the RT range", loc="left", fontweight="bold")
    ax.text(0.99, 0.02, f"Target forms: {n_target:,}", transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=COLORS["neutral"])
    save_figure(fig, out_dir / "figures" / "tuning_shift_by_rt_position")

    # Exact per-length medians and 10th–90th percentile bands.
    lengths_sorted = sorted(length_deltas)
    length_stats = []
    for length in lengths_sorted:
        vals = np.concatenate(length_deltas[length])
        length_stats.append((length, vals.size, *np.quantile(vals, [0.1, 0.5, 0.9])))
    stats_array = np.asarray(length_stats, dtype=float)
    keep = (stats_array[:, 0] >= 5) & (stats_array[:, 0] <= 40) & (stats_array[:, 1] >= 1000)
    selected = stats_array[keep]
    fig, ax = plt.subplots(figsize=(7.1, 3.45))
    if selected.size:
        ax.fill_between(selected[:, 0], selected[:, 2], selected[:, 4], color=COLORS["blue"], alpha=0.17, label="10th–90th percentile")
        ax.plot(selected[:, 0], selected[:, 3], color=COLORS["blue"], marker="o", ms=2.8, lw=1.4, label="Median")
        ax.axhline(0, color=COLORS["dark"], lw=0.8, ls=(0, (3, 2)))
        ax.set_xlim(4.5, 40.5)
    ax.set(xlabel="Peptide length (amino acids)", ylabel="RT-tuned − baseline (stored units)")
    ax.set_title("Tuning shift by peptide length", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(True, axis="y")
    fig.text(0.01, -0.02, "Target forms shared by all three libraries; only lengths with ≥1,000 forms are shown.", fontsize=6.8, color=COLORS["neutral"])
    save_figure(fig, out_dir / "figures" / "tuning_shift_by_peptide_length")
    result = {
        "common_target_forms": n_target,
        "tuning_shift_by_modification_site_counts": category_counts,
        "tuning_shift_rt_bounds": [min_delta, max_delta],
        "length_quantiles": [
            {"peptide_length": int(row[0]), "n": int(row[1]), "q10": float(row[2]), "median": float(row[3]), "q90": float(row[4])}
            for row in length_stats
        ],
        "method": "Three-way shared target forms; RT-tuned minus baseline; length is amino-acid count after removing UniMod annotations.",
    }
    logger.info("Follow-up plots used %s shared target forms", f"{n_target:,}")
    return result


def make_common_pair_table(pair_paths: dict[str, Path], out_dir: Path, logger: logging.Logger) -> pa.Table:
    renamed = {}
    for label in LIBRARIES:
        t = pq.read_table(pair_paths[label], columns=["pair_key", "decoy", "has_k", "rt_free", "rt_dimethyl"])
        t = t.rename_columns(
            [
                "pair_key",
                "decoy",
                f"has_k_{label}",
                f"rt_free_{label}",
                f"rt_dimethyl_{label}",
            ]
        )
        renamed[label] = t
    common = renamed["baseline"].join(renamed["tuned_rt"], keys=["pair_key", "decoy"], join_type="inner", use_threads=True)
    common = common.join(renamed["tuned_rt_fr"], keys=["pair_key", "decoy"], join_type="inner", use_threads=True)
    k0 = numpy_column(common, "has_k_baseline", bool)
    for label in LIBRARIES[1:]:
        if not np.array_equal(k0, numpy_column(common, f"has_k_{label}", bool)):
            raise ValueError(f"K-modification state differs across libraries for common pair keys ({label})")
    path = out_dir / "paired_nterm_rt_all_libraries.parquet"
    pq.write_table(common, path, compression="zstd", row_group_size=250_000)
    logger.info("Wrote %s three-way common N-terminal pairs to %s", f"{common.num_rows:,}", path)
    return common


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extracted-dir",
        type=Path,
        required=True,
        help="data directory containing library=<name>/rt.parquet",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="directory for pair data, summaries, logs, and figures",
    )
    parser.add_argument(
        "--common-forms",
        type=Path,
        help="shared forms table (default: <out-dir parent>/paired_rt_common.parquet)",
    )
    parser.add_argument(
        "--reuse-pairs",
        action="store_true",
        help="reuse existing paired_nterm_rt_<library>.parquet files and rebuild plots only",
    )
    args = parser.parse_args()
    data_dir = args.extracted_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    common_forms = (
        args.common_forms.expanduser().resolve()
        if args.common_forms is not None
        else out_dir.parent / "paired_rt_common.parquet"
    )
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(out_dir)
    started = time.monotonic()
    try:
        metadata = {}
        pair_paths = {}
        hist_by_library = {}
        rt_mins, rt_maxes = [], []
        # Parquet statistics give a shared histogram range without an extra scan.
        for label in LIBRARIES:
            path = data_dir / f"library={label}" / "rt.parquet"
            pf = pq.ParquetFile(path)
            rt_index = pf.schema.names.index("rt")
            for i in range(pf.metadata.num_row_groups):
                stat = pf.metadata.row_group(i).column(rt_index).statistics
                if stat is not None and stat.has_min_max:
                    rt_mins.append(float(stat.min))
                    rt_maxes.append(float(stat.max))
        rt_range = (min(rt_mins), max(rt_maxes))
        edges = np.linspace(rt_range[0], rt_range[1], 701)
        centers = (edges[:-1] + edges[1:]) / 2

        for label in LIBRARIES:
            table = read_library(data_dir, label, logger)
            pair_path = out_dir / f"paired_nterm_rt_{label}.parquet"
            if args.reuse_pairs:
                if not pair_path.is_file():
                    raise FileNotFoundError(f"--reuse-pairs requested, but pair dataset is missing: {pair_path}")
                pairs = pq.read_table(pair_path)
                logger.info("Reusing %s paired forms from %s", f"{pairs.num_rows:,}", pair_path)
            else:
                pairs, pair_path = make_nterm_pairs(table, label, out_dir, logger)
            pair_paths[label] = pair_path
            has_k = numpy_column(pairs, "has_k", bool)
            delta = numpy_column(pairs, "delta_rt", np.float64)
            lib_stats = {
                "input_forms": int(table.num_rows),
                "dimethylated_forms_by_site_group": {
                    "N-terminal only": int((numpy_column(table, "has_unimod_510_nterm", bool) & ~numpy_column(table, "has_unimod_510_k", bool)).sum()),
                    "Lysine only": int((~numpy_column(table, "has_unimod_510_nterm", bool) & numpy_column(table, "has_unimod_510_k", bool)).sum()),
                    "N-terminus + lysine": int((numpy_column(table, "has_unimod_510_nterm", bool) & numpy_column(table, "has_unimod_510_k", bool)).sum()),
                },
                "nterm_pairs": int(pairs.num_rows),
                "nterm_pairs_by_lysine_context": {
                    "no_lysine": int((~has_k).sum()),
                    "lysine_dimethylated": int(has_k.sum()),
                },
                "nterm_delta_rt": quantile_summary(delta),
                "by_lysine_context": {},
                "lysine_site_coverage": k_site_coverage(table),
            }
            plot_single_library_pairs(pairs, label, out_dir, lib_stats)
            hist_by_library[label] = density_counts(table, edges)
            plot_single_dimethyl_density(label, hist_by_library[label], centers, out_dir)
            metadata[label] = lib_stats
            del table, pairs
            gc.collect()
            logger.info("Finished %s plots; peak RSS %.1f GiB", label, peak_rss_gib())

        common = make_common_pair_table(pair_paths, out_dir, logger)
        stats = {
            "libraries": metadata,
            "nterm_pairing_rule": "Leading UniMod:510 removed from modified sequence; exact bare N-terminus, decoy, K modification state, and all other modifications matched.",
            "K_pairing_status": "unavailable: every K-containing peptide form in the checked libraries has UniMod:510 on its lysines; there are no unmodified-K candidate forms to pair.",
            "both_vs_neither_status": "unavailable: with lysine dimethylation fixed across K-containing candidates, no exact both-vs-neither peptide-form pair exists.",
            "combined_pair_count": int(common.num_rows),
            "common_pair_columns": common.column_names,
            "rt_range_stored_units": list(rt_range),
            "tuned_rt_and_tuned_rt_fr_identical_exactly": None,
        }
        if common.num_rows:
            tuned_a = numpy_column(common, "rt_dimethyl_tuned_rt", np.float32)
            tuned_b = numpy_column(common, "rt_dimethyl_tuned_rt_fr", np.float32)
            free_a = numpy_column(common, "rt_free_tuned_rt", np.float32)
            free_b = numpy_column(common, "rt_free_tuned_rt_fr", np.float32)
            stats["tuned_rt_and_tuned_rt_fr_identical_exactly"] = bool(
                np.array_equal(tuned_a, tuned_b) and np.array_equal(free_a, free_b)
            )
            stats["common_pairs"] = {}
            for label in LIBRARIES:
                free = numpy_column(common, f"rt_free_{label}", np.float64)
                dim = numpy_column(common, f"rt_dimethyl_{label}", np.float64)
                stats["common_pairs"][label] = {
                    "n": int(common.num_rows),
                    "nterm_delta_rt": quantile_summary(dim - free),
                }
        plot_combined_pairs(common, out_dir, stats)
        plot_combined_dimethyl_density(hist_by_library, centers, stats, out_dir)
        stats["followup_plots"] = plot_followup_common_form_patterns(
            common_forms, rt_range, out_dir, logger
        )

        # The requested K-only and both-vs-neither contrasts are audited as zero
        # matched pairs rather than emitted as blank scatter plots.
        stats["requested_pair_families"] = {
            "N-terminal dimethyl vs free": {"status": "paired", "counts": {label: metadata[label]["nterm_pairs"] for label in LIBRARIES}},
            "lysine dimethyl vs lysine-unmodified": {"status": "not pairable in these libraries", "counts": {label: 0 for label in LIBRARIES}},
            "N-terminus + lysine dimethyl vs neither": {"status": "not pairable in these libraries", "counts": {label: 0 for label in LIBRARIES}},
        }
        stats["elapsed_seconds"] = round(time.monotonic() - started, 1)
        stats["peak_rss_gib"] = round(peak_rss_gib(), 2)
        save_json(out_dir / "modification_pair_summary.json", stats)
        logger.info(
            "Finished: %s common N-terminal pairs; %0.1f min; peak RSS %.1f GiB",
            f"{common.num_rows:,}",
            stats["elapsed_seconds"] / 60,
            peak_rss_gib(),
        )
    except Exception:
        logger.exception("Modification pairing and plot generation failed")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
