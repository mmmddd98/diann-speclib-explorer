#!/usr/bin/env python3
"""Compare DIA-NN RT and RT+fragment tuning runs using final-report RT fields.

For each balanced/fast/slow quantification mode, the report rows are collapsed
across precursor charge on (Run, Modified.Sequence, Decoy). DIA-NN's report
Decoy flag is preserved and plotted separately; it is not FASTA-validated.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


REPORTS = {
    "balanced": {
        "RT": "balanced_rt_lr_0p00011_d0",
        "RT+FR": "balanced_rt_fr_lr_0p00011_d0",
    },
    "fast": {
        "RT": "fast_rt_lr_0p00011_d0",
        "RT+FR": "fast_rt_fr_lr_0p00011_d0",
    },
    "slow": {
        "RT": "slow_rt_lr_0p00011_d0",
        "RT+FR": "slow_rt_fr_lr_0p00011_d0",
    },
}
KEYS = ["Run", "Modified.Sequence", "Decoy"]
COLORS = {0: "#4477AA", 1: "#EE7733"}
MODE_ORDER = ["balanced", "fast", "slow"]
METHOD_ORDER = ["RT", "RT+FR"]
MODE_DETAILS = {
    "balanced": "DIA-NN default RT window",
    "fast": "RT window multiplier 1.7; factor 100",
    "slow": "--no-rt-window",
}


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.edgecolor": "#202938",
            "text.color": "#202938",
            "axes.labelcolor": "#202938",
            "xtick.color": "#202938",
            "ytick.color": "#202938",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D9DEE7",
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def configure_logger(out_dir: Path) -> logging.Logger:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logger = logging.getLogger("plot_integrated_tuning")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_dir / f"integrated_tuning_{stamp}.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def read_report(path: Path, mode: str, method: str, logger: logging.Logger) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"missing completed DIA-NN report: {path}")
    parquet = pq.ParquetFile(path)
    required = {*KEYS, "RT", "Predicted.RT", "Q.Value"}
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    raw = pd.read_parquet(path, columns=[*KEYS, "RT", "Predicted.RT", "Q.Value"])
    raw["Decoy"] = pd.to_numeric(raw["Decoy"], errors="raise").astype("int8")
    if not raw["Decoy"].isin([0, 1]).all():
        raise ValueError(f"{path}: expected DIA-NN Decoy values 0/1")
    for column in ("RT", "Predicted.RT", "Q.Value"):
        raw[column] = pd.to_numeric(raw[column], errors="raise")
    raw = raw.replace([np.inf, -np.inf], np.nan).dropna(subset=[*KEYS, "RT", "Predicted.RT"])
    # The report has precursor-level rows; summarize multiple charges as one
    # peptide form per run and preserve the explicit DIA-NN decoy state.
    forms = (
        raw.groupby(KEYS, as_index=False, sort=False, observed=True)
        .agg(
            observed_rt=("RT", "median"),
            predicted_rt=("Predicted.RT", "median"),
            min_q_value=("Q.Value", "min"),
            precursor_charge_rows=("RT", "size"),
        )
    )
    forms["mode"] = mode
    forms["method"] = method
    forms["rt_error"] = forms["observed_rt"] - forms["predicted_rt"]
    forms["absolute_rt_error"] = forms["rt_error"].abs()
    logger.info(
        "%s/%s: %s report rows collapsed to %s run-peptidoform keys (%s DIA-NN decoy-flag 1)",
        mode,
        method,
        f"{len(raw):,}",
        f"{len(forms):,}",
        f"{int(forms['Decoy'].sum()):,}",
    )
    return forms


def error_metrics(frame: pd.DataFrame) -> dict:
    error = frame["rt_error"].to_numpy(dtype=np.float64, copy=False)
    absolute = np.abs(error)
    return {
        "n_run_peptidoforms": int(len(frame)),
        "n_runs": int(frame["Run"].nunique()),
        "median_signed_error_observed_minus_predicted": float(np.median(error)),
        "mean_signed_error_observed_minus_predicted": float(np.mean(error)),
        "median_absolute_error": float(np.median(absolute)),
        "p90_absolute_error": float(np.quantile(absolute, 0.90)),
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
        "median_min_q_value": float(frame["min_q_value"].median()),
    }


def paired_mode_forms(forms: pd.DataFrame, mode: str) -> pd.DataFrame:
    mode_forms = forms.loc[forms["mode"] == mode]
    fields = ["observed_rt", "predicted_rt", "rt_error", "absolute_rt_error", "min_q_value"]
    rt = mode_forms.loc[mode_forms["method"] == "RT", KEYS + fields]
    rt_fr = mode_forms.loc[mode_forms["method"] == "RT+FR", KEYS + fields]
    return rt.merge(rt_fr, on=KEYS, how="inner", validate="one_to_one", suffixes=("_rt", "_rt_fr"))


def summarize(forms: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows = []
    for mode in MODE_ORDER:
        paired = paired_mode_forms(forms, mode)
        for method in METHOD_ORDER:
            suffix = "rt" if method == "RT" else "rt_fr"
            metric_frame = paired[KEYS + [
                f"rt_error_{suffix}",
                f"absolute_rt_error_{suffix}",
                f"min_q_value_{suffix}",
            ]].rename(
                columns={
                    f"rt_error_{suffix}": "rt_error",
                    f"absolute_rt_error_{suffix}": "absolute_rt_error",
                    f"min_q_value_{suffix}": "min_q_value",
                }
            )
            for decoy in (0, 1):
                subset = metric_frame.loc[metric_frame["Decoy"] == decoy]
                empty_metrics = {
                    "n_run_peptidoforms": 0,
                    "n_runs": 0,
                    "median_signed_error_observed_minus_predicted": math.nan,
                    "mean_signed_error_observed_minus_predicted": math.nan,
                    "median_absolute_error": math.nan,
                    "p90_absolute_error": math.nan,
                    "p95_absolute_error": math.nan,
                    "median_min_q_value": math.nan,
                }
                rows.append({"mode": mode, "method": method, "diann_decoy_flag": decoy,
                             **(error_metrics(subset) if len(subset) else empty_metrics)})
    summary = pd.DataFrame(rows)

    paired_rows = []
    for mode in MODE_ORDER:
        paired = paired_mode_forms(forms, mode)
        paired["delta_absolute_error_rt_fr_minus_rt"] = paired["absolute_rt_error_rt_fr"] - paired["absolute_rt_error_rt"]
        for decoy in (0, 1):
            subset = paired.loc[paired["Decoy"] == decoy]
            difference = subset["delta_absolute_error_rt_fr_minus_rt"].to_numpy(dtype=np.float64, copy=False)
            paired_rows.append(
                {
                    "mode": mode,
                    "diann_decoy_flag": decoy,
                    "n_paired_run_peptidoforms": int(len(subset)),
                    "median_delta_absolute_error_rt_fr_minus_rt": float(np.median(difference)) if len(difference) else math.nan,
                    "p90_absolute_delta_absolute_error": float(np.quantile(np.abs(difference), 0.90)) if len(difference) else math.nan,
                    "fraction_rt_fr_lower_absolute_error": float(np.mean(difference < 0)) if len(difference) else math.nan,
                    "fraction_equal_absolute_error": float(np.mean(difference == 0)) if len(difference) else math.nan,
                    "fraction_rt_lower_absolute_error": float(np.mean(difference > 0)) if len(difference) else math.nan,
                }
            )
    paired_summary = pd.DataFrame(paired_rows)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "DIA-NN step 5 final_report.parquet files",
        "key": KEYS,
        "unit_of_analysis": "one (Run, Modified.Sequence, DIA-NN Decoy) row after collapsing precursor charges by median RT",
        "rt_error_definition": "observed RT minus Predicted.RT from the DIA-NN report",
        "q_value_filter": None,
        "error_summary_basis": "exact run-peptidoform intersection between RT and RT+FR reports within each quant mode",
        "decoys": "retained and summarized separately using the DIA-NN Decoy field; not FASTA-validated",
        "absolute_counts_caveat": "report rows describe DIA-NN report observations, not total library candidates or verified target counts",
        "reports": {mode: labels for mode, labels in REPORTS.items()},
        "quant_mode_settings_from_startcmd": MODE_DETAILS,
    }
    return summary, paired_summary, metadata


def plot_error_distributions(forms: pd.DataFrame, output: Path) -> None:
    configure_plot_style()
    errors = forms["rt_error"].to_numpy(dtype=np.float64, copy=False)
    limit = max(0.25, float(np.quantile(np.abs(errors), 0.95)))
    bins = np.linspace(-limit, limit, 181)
    fig, axes = plt.subplots(3, 2, figsize=(8.2, 8.2), sharex=True, sharey=True, constrained_layout=True)
    for row, mode in enumerate(MODE_ORDER):
        paired = paired_mode_forms(forms, mode)
        for col, method in enumerate(METHOD_ORDER):
            ax = axes[row, col]
            suffix = "rt" if method == "RT" else "rt_fr"
            panel = paired[KEYS + [f"rt_error_{suffix}"]].rename(columns={f"rt_error_{suffix}": "rt_error"})
            for decoy in (0, 1):
                values = panel.loc[panel["Decoy"] == decoy, "rt_error"].to_numpy(dtype=np.float64, copy=False)
                if len(values):
                    ax.hist(
                        values,
                        bins=bins,
                        density=True,
                        histtype="step",
                        linewidth=1.25,
                        color=COLORS[decoy],
                        linestyle="-" if decoy == 0 else "--",
                        label=f"DIA-NN Decoy={decoy} (n={len(values):,})",
                    )
            ax.axvline(0, color="#202938", linewidth=0.8, linestyle=(0, (3, 2)))
            ax.set_title(f"{mode.capitalize()} · {method}", loc="left")
            ax.set_xlim(-limit, limit)
            if row == 2:
                ax.set_xlabel("Observed RT − Predicted.RT (DIA-NN units)")
            if col == 0:
                ax.set_ylabel("Density")
            ax.legend(frameon=False, loc="upper right")
    fig.suptitle("RT prediction residuals by integrated tuning and quant mode", fontsize=11, fontweight="bold")
    fig.text(
        0.01,
        -0.012,
        "Only exact RT/RT+FR shared run-peptidoforms are shown; charge collapsed per run and peptidoform. "
        "DIA-NN decoy flags shown separately, not FASTA-validated. "
        "X-range is central 95%; tail metrics remain in the CSV. Windows: balanced default, fast 1.7×/100, slow none.",
        fontsize=7,
        color="#687386",
    )
    save_figure(fig, output / "rt_prediction_residuals_by_mode")


def plot_paired_error_change(forms: pd.DataFrame, output: Path) -> None:
    configure_plot_style()
    fig, axes = plt.subplots(2, 3, figsize=(8.2, 5.6), sharey="row", constrained_layout=True)
    for row, decoy in enumerate((0, 1)):
        for col, mode in enumerate(MODE_ORDER):
            ax = axes[row, col]
            mode_forms = forms.loc[forms["mode"] == mode]
            rt = mode_forms.loc[mode_forms["method"] == "RT", KEYS + ["absolute_rt_error"]]
            rt_fr = mode_forms.loc[mode_forms["method"] == "RT+FR", KEYS + ["absolute_rt_error"]]
            paired = rt.merge(rt_fr, on=KEYS, how="inner", validate="one_to_one", suffixes=("_rt", "_rt_fr"))
            subset = paired.loc[paired["Decoy"] == decoy]
            delta = subset["absolute_rt_error_rt_fr"] - subset["absolute_rt_error_rt"]
            if len(delta):
                ax.boxplot(
                    delta.to_numpy(),
                    vert=True,
                    widths=0.45,
                    showfliers=False,
                    whis=(5, 95),
                    patch_artist=True,
                    boxprops={"facecolor": "#DCE6F1", "edgecolor": "#4477AA"},
                    medianprops={"color": "#202938", "linewidth": 1.2},
                    whiskerprops={"color": "#4477AA"},
                    capprops={"color": "#4477AA"},
                )
                median = float(delta.median())
                ax.text(0.97, 0.96, f"n={len(delta):,}\nmedian={median:.3g}", transform=ax.transAxes, ha="right", va="top", fontsize=7)
            ax.axhline(0, color="#202938", linewidth=0.8, linestyle=(0, (3, 2)))
            ax.set_title(mode.capitalize(), loc="left")
            ax.set_xticks([1], ["RT+FR − RT"])
            if col == 0:
                ax.set_ylabel(f"Decoy={decoy}\nΔ absolute RT error\n(DIA-NN units)")
    fig.suptitle("Paired change in absolute RT error: negative favors RT+FR", fontsize=11, fontweight="bold")
    fig.text(
        0.01,
        -0.012,
        "Pairs share Run, Modified.Sequence, and DIA-NN Decoy flag; charges collapsed by median. "
        "Boxes show IQR and 5th–95th percentile whiskers. Windows: balanced default, fast 1.7×/100, slow none.",
        fontsize=7,
        color="#687386",
    )
    save_figure(fig, output / "paired_absolute_rt_error_change")


def plot_per_run_medians(forms: pd.DataFrame, output: Path) -> None:
    configure_plot_style()
    runs = sorted(forms["Run"].astype(str).unique())
    palette = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 3, figsize=(8.2, 5.6), sharey="row", constrained_layout=True)
    for row, decoy in enumerate((0, 1)):
        for col, mode in enumerate(MODE_ORDER):
            ax = axes[row, col]
            paired = paired_mode_forms(forms, mode)
            subset = paired.loc[paired["Decoy"] == decoy]
            per_run = []
            for method in METHOD_ORDER:
                suffix = "rt" if method == "RT" else "rt_fr"
                method_medians = subset.groupby("Run", observed=True)[f"absolute_rt_error_{suffix}"].median()
                per_run.append(method_medians.rename(method))
            medians = pd.concat(per_run, axis=1)
            for run_index, run in enumerate(runs):
                if run not in medians.index or not all(method in medians.columns for method in METHOD_ORDER):
                    continue
                values = medians.loc[run, METHOD_ORDER].to_numpy(dtype=np.float64)
                if not np.isfinite(values).all():
                    continue
                color = palette(run_index % 10)
                ax.plot([0, 1], values, marker="o", markersize=3.2, linewidth=0.8, alpha=0.78, color=color, label=run)
            ax.set_xticks([0, 1], METHOD_ORDER)
            ax.set_title(mode.capitalize(), loc="left")
            if col == 0:
                ax.set_ylabel(f"Decoy={decoy}\nPer-run median absolute error\n(DIA-NN units)")
            if row == 0 and col == 2:
                ax.legend(title="Run", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Per-run median RT prediction error", fontsize=11, fontweight="bold")
    fig.text(
        0.01,
        -0.012,
        "Lines connect the same run; both DIA-NN Decoy states are retained and shown separately (not FASTA-validated). "
        "Windows: balanced default, fast 1.7×/100, slow none.",
        fontsize=7,
        color="#687386",
    )
    save_figure(fig, output / "per_run_median_absolute_rt_error")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant-dir", required=True, type=Path, help="DIA-NN 05_quant/tuned directory")
    parser.add_argument("--out-dir", required=True, type=Path, help="directory for plots, summaries, and logs")
    args = parser.parse_args()
    quant_dir = args.quant_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(out_dir)
    started = time.monotonic()
    try:
        frames = []
        for mode in MODE_ORDER:
            for method in METHOD_ORDER:
                label = REPORTS[mode][method]
                frames.append(read_report(quant_dir / label / "final_report.parquet", mode, method, logger))
        forms = pd.concat(frames, ignore_index=True)
        summary, paired_summary, metadata = summarize(forms)
        summary.to_csv(out_dir / "rt_error_summary_by_mode_method_decoy.csv", index=False)
        paired_summary.to_csv(out_dir / "paired_rt_fr_vs_rt_error_summary.csv", index=False)
        metadata["elapsed_seconds"] = round(time.monotonic() - started, 2)
        metadata["report_row_peptidoform_totals"] = {
            f"{mode}/{method}": int(len(forms.loc[(forms["mode"] == mode) & (forms["method"] == method)]))
            for mode in MODE_ORDER
            for method in METHOD_ORDER
        }
        (out_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        plot_error_distributions(forms, out_dir / "figures")
        plot_paired_error_change(forms, out_dir / "figures")
        plot_per_run_medians(forms, out_dir / "figures")
        logger.info("Wrote integrated-tuning figures and summaries to %s", out_dir)
        logger.info("Elapsed %.1f s", metadata["elapsed_seconds"])
    except Exception:
        logger.exception("Integrated tuning analysis failed")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
