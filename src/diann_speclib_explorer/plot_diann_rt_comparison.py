#!/usr/bin/env python3
"""Pair extracted DIA-NN RT libraries by peptide key and create publication-style plots.

The full outputs are joined on (modified_sequence, decoy), never on row position.
Only peptide forms present in all three libraries are included in the like-for-like
figures. The shared table and full-count summaries are saved for later replotting.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
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
import pandas as pd


COLORS = {
    "blue": "#4477AA",
    "orange": "#EE7733",
    "magenta": "#AA3377",
    "purple": "#6F4C9B",
    "neutral": "#7E8794",
    "dark": "#202938",
    "grid": "#D9DEE7",
    "background": "#FFFFFF",
}
KEYS = ["modified_sequence", "decoy"]
BASELINE = "baseline"
TUNED_RT = "tuned_rt"
TUNED_RT_FR = "tuned_rt_fr"
RT_COLUMNS = {
    BASELINE: "rt_baseline",
    TUNED_RT: "rt_tuned_rt",
    TUNED_RT_FR: "rt_tuned_rt_fr",
}


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
            "figure.facecolor": COLORS["background"],
            "axes.facecolor": COLORS["background"],
            "savefig.facecolor": COLORS["background"],
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def configure_logger(output_dir: Path) -> logging.Logger:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"plot_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
    logger = logging.getLogger("plot_diann_rt_comparison")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def peak_rss_gb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024**2) if sys.platform != "darwin" else value / (1024**3)


def save_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_extracted(library_dir: Path, label: str, logger: logging.Logger) -> pd.DataFrame:
    path = library_dir / f"library={label}" / "rt.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"missing extracted library parquet: {path}")
    columns = KEYS + [
        "has_unimod_510",
        "has_unimod_510_nterm",
        "has_unimod_510_k",
        "rt",
    ]
    logger.info("Loading %s from %s (%.2f GB compressed)", label, path, path.stat().st_size / 1e9)
    frame = pd.read_parquet(path, columns=columns)
    if frame[KEYS].isna().any().any() or frame["rt"].isna().any():
        raise ValueError(f"{label}: key or RT contains null values")
    if not np.isfinite(frame["rt"].to_numpy(copy=False)).all():
        raise ValueError(f"{label}: RT contains non-finite values")
    logger.info("Loaded %s: %s rows; peak RSS %.1f GiB", label, f"{len(frame):,}", peak_rss_gb())
    return frame


def merge_three_libraries(
    extracted_dir: Path,
    paired_path: Path,
    summary_path: Path,
    logger: logging.Logger,
    *,
    baseline_label: str = BASELINE,
    tuned_rt_label: str = TUNED_RT,
    tuned_rt_fr_label: str = TUNED_RT_FR,
) -> dict:
    baseline = load_extracted(extracted_dir, baseline_label, logger).rename(columns={"rt": RT_COLUMNS[BASELINE]})
    tuned_rt = load_extracted(extracted_dir, tuned_rt_label, logger)[KEYS + ["rt"]].rename(
        columns={"rt": RT_COLUMNS[TUNED_RT]}
    )
    tuned_rt_fr = load_extracted(extracted_dir, tuned_rt_fr_label, logger)[KEYS + ["rt"]].rename(
        columns={"rt": RT_COLUMNS[TUNED_RT_FR]}
    )

    input_counts = {
        BASELINE: int(len(baseline)),
        TUNED_RT: int(len(tuned_rt)),
        TUNED_RT_FR: int(len(tuned_rt_fr)),
    }
    logger.info("Joining by the actual peptide key, not by row position")
    common = baseline.merge(tuned_rt, on=KEYS, how="inner", sort=False, validate="one_to_one")
    del baseline, tuned_rt
    gc.collect()
    logger.info("First key join: %s shared forms; peak RSS %.1f GiB", f"{len(common):,}", peak_rss_gb())
    common = common.merge(tuned_rt_fr, on=KEYS, how="inner", sort=False, validate="one_to_one")
    del tuned_rt_fr
    gc.collect()
    logger.info("Three-way key intersection: %s forms; peak RSS %.1f GiB", f"{len(common):,}", peak_rss_gb())
    if common.empty:
        raise ValueError("the three libraries have no shared peptide forms")

    out_columns = KEYS + [
        "has_unimod_510",
        "has_unimod_510_nterm",
        "has_unimod_510_k",
        RT_COLUMNS[BASELINE],
        RT_COLUMNS[TUNED_RT],
        RT_COLUMNS[TUNED_RT_FR],
    ]
    common = common[out_columns]
    paired_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing shared keyed dataset to %s", paired_path)
    common.to_parquet(
        paired_path,
        engine="pyarrow",
        compression="zstd",
        index=False,
        row_group_size=250_000,
    )

    common_count = int(len(common))
    unmatched = {label: input_counts[label] - common_count for label in input_counts}
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "matching_key": KEYS,
        "join_method": "one-to-one inner joins on the peptide-form key",
        "row_order_used_for_matching": False,
        "input_rows": input_counts,
        "shared_by_all_three": common_count,
        "not_in_three_way_intersection": unmatched,
        "paired_data_file": str(paired_path.resolve()),
        "paired_data_size_bytes": paired_path.stat().st_size,
        "rt_columns": RT_COLUMNS,
        "input_labels": {
            BASELINE: baseline_label,
            TUNED_RT: tuned_rt_label,
            TUNED_RT_FR: tuned_rt_fr_label,
        },
        "memory_peak_gb_during_join": round(peak_rss_gb(), 2),
    }
    save_json(summary_path, summary)
    return summary


def modification_groups(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    has = frame["has_unimod_510"].to_numpy(dtype=bool, copy=False)
    nterm = frame["has_unimod_510_nterm"].to_numpy(dtype=bool, copy=False)
    lysine = frame["has_unimod_510_k"].to_numpy(dtype=bool, copy=False)
    return {
        "All shared forms": np.ones(len(frame), dtype=bool),
        "No UniMod:510": ~has,
        "UniMod:510, any site": has,
        "N-terminal only": has & nterm & ~lysine,
        "Lysine only": has & ~nterm & lysine,
        "N-terminus + lysine": has & nterm & lysine,
        "Other UniMod:510 placement": has & ~nterm & ~lysine,
    }


def summarize_delta(delta: np.ndarray, mask: np.ndarray) -> dict:
    values = delta[mask]
    if values.size == 0:
        return {"n": 0}
    abs_values = np.abs(values)
    return {
        "n": int(values.size),
        "mean_delta_rt": float(np.mean(values, dtype=np.float64)),
        "median_delta_rt": float(np.median(values)),
        "sd_delta_rt": float(np.std(values, dtype=np.float64)),
        "mean_absolute_delta_rt": float(np.mean(abs_values, dtype=np.float64)),
        "rmse_delta_rt": float(np.sqrt(np.mean(values.astype(np.float64) ** 2))),
        "p95_absolute_delta_rt": float(np.quantile(abs_values, 0.95)),
        "p99_absolute_delta_rt": float(np.quantile(abs_values, 0.99)),
    }


def pair_statistics(frame: pd.DataFrame) -> tuple[dict, dict[str, np.ndarray]]:
    baseline = frame[RT_COLUMNS[BASELINE]].to_numpy(dtype=np.float64, copy=False)
    variants = {
        TUNED_RT: frame[RT_COLUMNS[TUNED_RT]].to_numpy(dtype=np.float64, copy=False),
        TUNED_RT_FR: frame[RT_COLUMNS[TUNED_RT_FR]].to_numpy(dtype=np.float64, copy=False),
    }
    groups = modification_groups(frame)
    summaries = {}
    deltas = {}
    for label, tuned in variants.items():
        delta = tuned - baseline
        deltas[label] = delta
        pearson = float(np.corrcoef(baseline, tuned)[0, 1])
        summaries[label] = {
            "comparison": f"{label} minus baseline",
            "pearson_r": pearson,
            "overall": summarize_delta(delta, groups["All shared forms"]),
            "by_modification_group": {
                group: summarize_delta(delta, mask) for group, mask in groups.items() if group != "All shared forms"
            },
        }
    return summaries, deltas


def add_dimethyl_contour(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    bins: int,
    ranges: tuple[tuple[float, float], tuple[float, float]],
):
    if not np.any(mask):
        return
    hist, xedges, yedges = np.histogram2d(x[mask], y[mask], bins=bins, range=ranges)
    positive = hist[hist > 0]
    if positive.size < 8:
        return
    levels = np.unique(np.quantile(positive, [0.60, 0.85, 0.97]))
    levels = levels[levels > 0]
    if levels.size:
        centers_x = (xedges[:-1] + xedges[1:]) / 2
        centers_y = (yedges[:-1] + yedges[1:]) / 2
        ax.contour(
            centers_x,
            centers_y,
            hist.T,
            levels=levels,
            colors=[COLORS["orange"]],
            linewidths=0.8,
            alpha=0.95,
        )


def save_figure(fig, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def create_figures(
    frame: pd.DataFrame,
    figures_dir: Path,
    summary: dict,
    logger: logging.Logger,
) -> dict:
    configure_plot_style()
    groups = modification_groups(frame)
    dimethyl = groups["UniMod:510, any site"]
    base_rt = frame[RT_COLUMNS[BASELINE]].to_numpy(dtype=np.float64, copy=False)
    rt_only = frame[RT_COLUMNS[TUNED_RT]].to_numpy(dtype=np.float64, copy=False)
    rt_fr = frame[RT_COLUMNS[TUNED_RT_FR]].to_numpy(dtype=np.float64, copy=False)
    delta_rt = rt_only - base_rt
    delta_rt_fr = rt_fr - base_rt
    limits = (
        float(min(base_rt.min(), rt_only.min(), rt_fr.min())),
        float(max(base_rt.max(), rt_only.max(), rt_fr.max())),
    )
    comparisons = [
        (TUNED_RT, rt_only, delta_rt, "RT-tuned"),
        (TUNED_RT_FR, rt_fr, delta_rt_fr, "RT + fragment-tuned"),
    ]
    n = len(frame)
    logger.info("Creating figures from %s matched peptide forms", f"{n:,}")

    # Figure 1: paired RT distributions, with the dimethylated subset outlined.
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.15), sharex=True, sharey=True, constrained_layout=True)
    last_mesh = None
    for ax, (label, tuned, _delta, title) in zip(axes, comparisons):
        hist, xedges, yedges = np.histogram2d(base_rt, tuned, bins=260, range=[limits, limits])
        masked = np.ma.masked_less_equal(hist.T, 0)
        vmax = max(1.0, float(hist.max()))
        last_mesh = ax.pcolormesh(
            xedges,
            yedges,
            masked,
            norm=LogNorm(vmin=1, vmax=vmax),
            cmap="Blues",
            shading="auto",
            rasterized=True,
        )
        add_dimethyl_contour(ax, base_rt, tuned, dimethyl, 260, (limits, limits))
        ax.plot(limits, limits, color=COLORS["dark"], lw=0.8, ls=(0, (3, 2)), alpha=0.85)
        r = summary["comparisons"][label]["pearson_r"]
        ax.set_title(f"{title}  |  r = {r:.4f}", loc="left", pad=7)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Baseline predicted RT (stored units)")
        ax.grid(False)
    axes[0].set_ylabel("Tuned predicted RT (stored units)")
    fig.colorbar(last_mesh, ax=axes, shrink=0.82, pad=0.025, label="Peptide forms per bin (log scale)")
    fig.suptitle("Predicted retention times across DIA-NN libraries", fontsize=10, fontweight="bold")
    tuned_identity_note = (
        "RT-only and RT + fragment-tuned values are identical. "
        if summary.get("tuned_rt_and_tuned_rt_fr_identical_exactly")
        else ""
    )
    fig.text(
        0.01,
        -0.015,
        f"N = {n:,} shared peptide forms; decoy flags are retained but not FASTA-validated. "
        f"Orange contours mark UniMod:510 forms. {tuned_identity_note}",
        fontsize=7,
        color=COLORS["neutral"],
    )
    save_figure(fig, figures_dir / "figure1_rt_pairwise_density")

    # Figure 2: where RT changes occur across the baseline RT range.
    all_delta = np.concatenate((delta_rt, delta_rt_fr))
    delta_limits = tuple(float(x) for x in np.quantile(all_delta, [0.0025, 0.9975]))
    if delta_limits[0] == delta_limits[1]:
        delta_limits = (delta_limits[0] - 1.0, delta_limits[1] + 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.1), sharex=True, sharey=True, constrained_layout=True)
    for ax, (label, _tuned, delta, title) in zip(axes, comparisons):
        hist, xedges, yedges = np.histogram2d(
            base_rt,
            delta,
            bins=(260, 220),
            range=[limits, delta_limits],
        )
        mesh = ax.pcolormesh(
            xedges,
            yedges,
            np.ma.masked_less_equal(hist.T, 0),
            norm=LogNorm(vmin=1, vmax=max(1.0, float(hist.max()))),
            cmap="Blues",
            shading="auto",
            rasterized=True,
        )
        add_dimethyl_contour(ax, base_rt, delta, dimethyl, 220, (limits, delta_limits))
        ax.axhline(0, color=COLORS["dark"], lw=0.8, ls=(0, (3, 2)))
        ax.set_title(title, loc="left", pad=7)
        ax.set_xlim(limits)
        ax.set_ylim(delta_limits)
        ax.set_xlabel("Baseline predicted RT (stored units)")
        ax.grid(False)
    axes[0].set_ylabel("Tuned − baseline RT (stored units)")
    fig.colorbar(mesh, ax=axes, shrink=0.82, pad=0.025, label="Peptide forms per bin (log scale)")
    fig.suptitle("RT shifts across the baseline prediction range", fontsize=10, fontweight="bold")
    fig.text(
        0.01,
        -0.015,
        "Axes show the central 99.5% of RT shifts; decoy flags are retained but not FASTA-validated; "
        "orange contours mark UniMod:510 forms. "
        + tuned_identity_note,
        fontsize=7,
        color=COLORS["neutral"],
    )
    save_figure(fig, figures_dir / "figure2_rt_shift_by_baseline")

    # Figure 3: signed RT-shift intervals by UniMod:510 placement.
    categories = [
        ("No UniMod:510", "No 510"),
        ("N-terminal only", "N-term"),
        ("Lysine only", "Lysine"),
        ("N-terminus + lysine", "N-term + K"),
        ("Other UniMod:510 placement", "Other site"),
    ]
    categories = [
        (group, label) for group, label in categories if groups[group].any()
    ]
    fig, ax = plt.subplots(figsize=(7.1, 3.6), constrained_layout=True)
    positions = np.arange(len(categories), dtype=float)
    offsets = {TUNED_RT: -0.12, TUNED_RT_FR: 0.12}
    variant_colors = {TUNED_RT: COLORS["blue"], TUNED_RT_FR: COLORS["orange"]}
    interval_summary = {}
    for variant, delta, display in (
        (TUNED_RT, delta_rt, "RT-tuned"),
        (TUNED_RT_FR, delta_rt_fr, "RT + fragment-tuned"),
    ):
        interval_summary[variant] = {}
        for index, (group_name, short_name) in enumerate(categories):
            mask = groups[group_name]
            values = delta[mask]
            if values.size == 0:
                interval_summary[variant][group_name] = {"n": 0}
                continue
            q05, q25, q50, q75, q95 = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])
            x = positions[index] + offsets[variant]
            color = variant_colors[variant]
            ax.vlines(x, q05, q95, color=color, linewidth=1.0, alpha=0.8)
            ax.vlines(x, q25, q75, color=color, linewidth=4.0, alpha=0.9)
            ax.scatter([x], [q50], s=20, color=color, edgecolor="white", linewidth=0.5, zorder=3)
            interval_summary[variant][group_name] = {
                "n": int(values.size),
                "q05": float(q05),
                "q25": float(q25),
                "median": float(q50),
                "q75": float(q75),
                "q95": float(q95),
            }
    ax.axhline(0, color=COLORS["dark"], lw=0.8, ls=(0, (3, 2)))
    ax.set_xticks(positions, [short for _long, short in categories])
    ax.set_ylabel("Tuned − baseline RT (stored units)")
    ax.set_xlabel("Peptide modification group")
    ax.grid(True, axis="y")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=COLORS["blue"], marker="o", lw=2, label="RT-tuned"),
            plt.Line2D([0], [0], color=COLORS["orange"], marker="o", lw=2, label="RT + fragment-tuned"),
        ],
        loc="upper right",
    )
    fig.suptitle("RT shifts by UniMod:510 placement", fontsize=10, fontweight="bold")
    fig.text(
        0.01,
        -0.025,
        "Points show medians; thick bars show the interquartile range; thin bars show the 5th–95th percentiles. "
        "Decoy flags are retained but not FASTA-validated. "
        + tuned_identity_note,
        fontsize=7,
        color=COLORS["neutral"],
    )
    save_figure(fig, figures_dir / "figure3_rt_shift_by_dimethyl_site")

    summary["intervals_by_modification_group"] = interval_summary
    summary["plot_axes"] = {
        "rt_limits": list(limits),
        "rt_shift_limits_central_99_5_percent": list(delta_limits),
        "rt_unit": "stored DIA-NN RT units; the source format does not encode/display a unit label here",
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extracted-dir",
        type=Path,
        required=True,
        help="data directory produced by extract_diann_rt.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="output folder for paired data, summaries, and figures",
    )
    parser.add_argument(
        "--reuse-paired",
        action="store_true",
        help="reuse an existing paired_rt_common.parquet instead of repeating the full key join",
    )
    parser.add_argument("--baseline-label", default=BASELINE, help=f"extracted baseline label (default: {BASELINE})")
    parser.add_argument("--tuned-rt-label", default=TUNED_RT, help=f"RT-only library label (default: {TUNED_RT})")
    parser.add_argument("--tuned-rt-fr-label", default=TUNED_RT_FR, help=f"RT+fragment library label (default: {TUNED_RT_FR})")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    extracted_dir = args.extracted_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(out_dir)
    started = time.monotonic()
    try:
        paired_path = out_dir / "paired_rt_common.parquet"
        summary_path = out_dir / "summary.json"
        if args.reuse_paired:
            if not paired_path.is_file():
                raise FileNotFoundError(f"--reuse-paired requested, but no paired dataset exists at {paired_path}")
            summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
            logger.info("Reusing existing paired dataset %s", paired_path)
        else:
            summary = merge_three_libraries(
                extracted_dir,
                paired_path,
                summary_path,
                logger,
                baseline_label=args.baseline_label,
                tuned_rt_label=args.tuned_rt_label,
                tuned_rt_fr_label=args.tuned_rt_fr_label,
            )
        paired_frame = pd.read_parquet(paired_path)
        tuned_rt_identical = bool(
            np.array_equal(
                paired_frame[RT_COLUMNS[TUNED_RT]].to_numpy(copy=False),
                paired_frame[RT_COLUMNS[TUNED_RT_FR]].to_numpy(copy=False),
            )
        )
        summary["tuned_rt_and_tuned_rt_fr_identical_exactly"] = tuned_rt_identical
        logger.info("RT-only and RT+fragment tuned values identical for every shared form: %s", tuned_rt_identical)
        pair_stats, _deltas = pair_statistics(paired_frame)
        del _deltas
        summary["comparisons"] = pair_stats
        summary["all_three_common_forms"] = int(len(paired_frame))
        summary["shared_forms_by_modification_group"] = {
            label: int(mask.sum()) for label, mask in modification_groups(paired_frame).items()
        }
        del paired_frame
        gc.collect()

        paired_frame = pd.read_parquet(paired_path)
        summary = create_figures(paired_frame, figures_dir, summary, logger)
        summary["elapsed_seconds"] = round(time.monotonic() - started, 1)
        save_json(summary_path, summary)
        summary_csv_rows = []
        for comparison, result in summary["comparisons"].items():
            overall = result["overall"]
            summary_csv_rows.append(
                {
                    "comparison": result["comparison"],
                    "group": "All shared forms",
                    **overall,
                    "pearson_r": result["pearson_r"],
                }
            )
            for group, values in result["by_modification_group"].items():
                summary_csv_rows.append(
                    {"comparison": result["comparison"], "group": group, **values, "pearson_r": None}
                )
        pd.DataFrame(summary_csv_rows).to_csv(out_dir / "summary.csv", index=False)
        logger.info("Finished: %s shared forms; figures in %s", f"{summary['all_three_common_forms']:,}", figures_dir)
        logger.info("Elapsed %.1f min; peak RSS %.1f GiB", summary["elapsed_seconds"] / 60, peak_rss_gb())
    except Exception:
        logger.exception("RT comparison plot generation failed")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
