#!/usr/bin/env python3
"""Create a sampled, key-matched comparison across every extracted RT library.

The plot compares within-library RT percentiles rather than raw RT values, since
CARAFE and DIA-NN outputs can use different coordinate scales. A reproducible
sample of peptidoforms is taken from the original DIA-NN baseline, then
matched by exact Modified.Sequence and the extracted decoy flag in every
library. This keeps memory bounded while checking identical library forms.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import resource
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


PALETTE = {"blue": "#4477AA", "orange": "#EE7733", "teal": "#009E73", "neutral": "#7E8794", "dark": "#202938", "grid": "#D9DEE7"}
KEY = "modified_sequence"
BASELINE = "baseline"


def peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def parquet_rows(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def sample_baseline(path: Path, wanted: int, seed: int) -> pa.Table:
    """Sample from every row group using a fixed seed, then cap exactly."""
    rng = np.random.default_rng(seed)
    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    fraction = min(1.0, max((wanted * 1.35) / max(total, 1), 1e-6))
    pieces: list[pa.Table] = []
    collected = 0
    columns = [KEY, "decoy", "has_unimod_510", "rt"]
    if "has_unimod_255" in pf.schema_arrow.names:
        columns.append("has_unimod_255")
    for batch in pf.iter_batches(batch_size=500_000, columns=columns):
        if "has_unimod_255" not in batch.schema.names:
            batch = batch.append_column("has_unimod_255", pa.array([False] * batch.num_rows))
        mask = rng.random(batch.num_rows) < fraction
        if mask.any():
            filtered = pc.filter(batch, pa.array(mask))
            pieces.append(pa.Table.from_batches([filtered]))
            collected += filtered.num_rows
    if not pieces:
        raise ValueError(f"sampled no rows from {path}")
    sample = pa.concat_tables(pieces)
    if sample.num_rows > wanted:
        indices = np.sort(rng.choice(sample.num_rows, wanted, replace=False))
        sample = pc.take(sample, pa.array(indices, type=pa.int64()))
    return sample


def read_matching_sample(path: Path, keys_by_decoy: dict[bool, pa.Array]) -> tuple[pa.Table, int]:
    pf = pq.ParquetFile(path)
    found: list[pa.Table] = []
    total_255 = 0
    columns = [KEY, "decoy", "has_unimod_510", "rt"]
    has_255_flag = "has_unimod_255" in pf.schema_arrow.names
    if has_255_flag:
        columns.append("has_unimod_255")
    for batch in pf.iter_batches(batch_size=500_000, columns=columns):
        if has_255_flag:
            total_255 += int(pc.sum(batch.column("has_unimod_255")).as_py() or 0)
        else:
            total_255 += int(pc.sum(pc.match_substring_regex(batch.column(KEY), r"\(UniMod:255\)")).as_py() or 0)
            batch = batch.append_column(
                "has_unimod_255",
                pc.match_substring_regex(batch.column(KEY), r"\(UniMod:255\)"),
            )
        target_mask = pc.is_in(batch.column(KEY), value_set=keys_by_decoy[False])
        target_mask = pc.and_(target_mask, pc.equal(batch.column("decoy"), False))
        mask = target_mask
        if len(keys_by_decoy[True]):
            decoy_mask = pc.is_in(batch.column(KEY), value_set=keys_by_decoy[True])
            decoy_mask = pc.and_(decoy_mask, pc.equal(batch.column("decoy"), True))
            mask = pc.or_(mask, decoy_mask)
        if pc.any(mask).as_py():
            found.append(pa.Table.from_batches([pc.filter(batch, mask)]))
    if found:
        return pa.concat_tables(found), total_255
    return pa.table({KEY: pa.array([], type=pa.string()), "has_unimod_510": pa.array([], type=pa.bool_()), "has_unimod_255": pa.array([], type=pa.bool_()), "rt": pa.array([], type=pa.float32())}), total_255


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = ((start + stop - 1) / 2 + 0.5) / max(len(values), 1)
        start = stop
    return ranks


def numpy_column(table: pa.Table, name: str, dtype) -> np.ndarray:
    return np.asarray(
        table.column(name).combine_chunks().to_numpy(zero_copy_only=False), dtype=dtype
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path, help="full RT dataset directory")
    parser.add_argument("--output-dir", type=Path, help="defaults to OUT_DIR/analysis/all_libraries")
    parser.add_argument("--sample-size", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--baseline-label", default=BASELINE)
    parser.add_argument("--cpus", type=int, default=4, help="Arrow CPU limit for keyed Parquet scans")
    args = parser.parse_args()
    if args.sample_size < 1 or args.cpus < 1:
        parser.error("--sample-size and --cpus must be positive")
    pa.set_cpu_count(args.cpus)
    data_dir = args.out_dir / "data"
    output = args.output_dir or (args.out_dir / "analysis" / "all_libraries")
    output.mkdir(parents=True, exist_ok=True)
    # Remove legacy absolute-entry-count charts so a stale figure cannot be
    # mistaken for the current sampled comparison.
    for stale_name in ("library_coverage.pdf", "library_coverage.png"):
        stale_path = output / stale_name
        if stale_path.exists():
            stale_path.unlink()
    manifest = load_manifest(args.out_dir / "manifest.json")
    baseline_path = data_dir / f"library={args.baseline_label}" / "rt.parquet"
    if not baseline_path.is_file():
        raise FileNotFoundError(f"reference library missing: {baseline_path}")

    print(f"Sampling {args.baseline_label} from {baseline_path}", flush=True)
    started = time.monotonic()
    baseline = sample_baseline(baseline_path, args.sample_size, args.seed)
    baseline = baseline.combine_chunks()
    baseline_sequences = baseline.column(KEY).combine_chunks().to_pylist()
    baseline_decoys = baseline.column("decoy").combine_chunks().to_pylist()
    baseline_keys = list(zip(baseline_sequences, baseline_decoys))
    baseline_index = {key: i for i, key in enumerate(baseline_keys)}
    keys_by_decoy = {
        flag: pa.array(
            [sequence for sequence, decoy in baseline_keys if decoy is flag],
            type=pa.string(),
        )
        for flag in (False, True)
    }
    baseline_rt = numpy_column(baseline, "rt", np.float64)
    baseline_dim = numpy_column(baseline, "has_unimod_510", bool)
    baseline_255 = numpy_column(baseline, "has_unimod_255", bool)
    baseline_rank = ranks(baseline_rt)
    print(f"Baseline sample: {len(baseline_rt):,}; peak RSS {peak_gb():.1f} GiB", flush=True)

    libraries = sorted(p.parent.name.removeprefix("library=") for p in data_dir.glob("library=*/rt.parquet"))
    libraries = [label for label in libraries if label != args.baseline_label]
    extra = manifest.get("carafe_libraries", {})
    library_meta = manifest.get("libraries", {}) | extra
    results: list[dict] = []
    panel_rows = max(1, math.ceil((len(libraries) + 1) / 4))
    figure, axes = plt.subplots(
        nrows=panel_rows,
        ncols=4,
        figsize=(17.6, max(9.0, 3.35 * panel_rows)),
        constrained_layout=False,
        sharex=True,
        sharey=True,
    )
    figure.subplots_adjust(left=0.075, right=0.99, top=0.94, bottom=0.085, wspace=0.30, hspace=0.55)
    axes_flat = list(np.asarray(axes).ravel())
    reference_ax = axes_flat[0]
    reference_ax.hexbin(
        baseline_rank, baseline_rank, gridsize=45, mincnt=1, bins="log", cmap="Blues", linewidths=0
    )
    if baseline_dim.any():
        reference_ax.scatter(
            baseline_rank[baseline_dim], baseline_rank[baseline_dim],
            s=0.05, c=PALETTE["orange"], alpha=0.14, rasterized=True,
        )
    if baseline_255.any():
        reference_ax.scatter(
            baseline_rank[baseline_255], baseline_rank[baseline_255],
            s=0.05, c=PALETTE["teal"], alpha=0.2, rasterized=True,
        )
    reference_ax.plot([0, 1], [0, 1], color=PALETTE["neutral"], linewidth=0.65, linestyle="--")
    reference_ax.set(
        xlim=(0, 1), ylim=(0, 1), xlabel="", ylabel=""
    )
    reference_ax.set_title(f"{args.baseline_label}\nrₛ=1.000; n={len(baseline_rank):,}", fontsize=9)
    reference_ax.grid(True, alpha=0.2)

    dim_figure, dim_axes = plt.subplots(
        nrows=panel_rows,
        ncols=4,
        figsize=(17.6, max(9.0, 3.35 * panel_rows)),
        constrained_layout=False,
        sharex=True,
        sharey=True,
    )
    dim_figure.subplots_adjust(left=0.075, right=0.99, top=0.94, bottom=0.085, wspace=0.30, hspace=0.55)
    dim_axes_flat = list(np.asarray(dim_axes).ravel())
    dim_reference_ax = dim_axes_flat[0]
    if baseline_dim.any():
        dim_reference_rank = ranks(baseline_rt[baseline_dim])
        dim_reference_ax.hexbin(
            dim_reference_rank, dim_reference_rank, gridsize=45, mincnt=1,
            bins="log", cmap="Oranges", linewidths=0,
        )
        dim_reference_ax.plot([0, 1], [0, 1], color=PALETTE["neutral"], linewidth=0.65, linestyle="--")
        dim_reference_ax.set(
            xlim=(0, 1), ylim=(0, 1), xlabel="", ylabel=""
        )
        dim_reference_ax.set_title(
            f"{args.baseline_label}\nrₛ=1.000; n={int(baseline_dim.sum()):,}", fontsize=9
        )
        dim_reference_ax.grid(True, alpha=0.2)
    else:
        dim_reference_ax.set_visible(False)

    baseline_meta = manifest.get("libraries", {}).get(args.baseline_label, {})
    results.append({
        "library": args.baseline_label,
        "source_type": baseline_meta.get("source_type", "DIA-NN compact .speclib"),
        "completeness": "complete",
        "extracted_peptidoform_records": parquet_rows(baseline_path),
        "sampled_baseline_forms": len(baseline_rt),
        "sample_matches": len(baseline_rt),
        "sample_match_fraction": 1.0,
        "sample_matches_dimethyl": int(baseline_dim.sum()),
        "sample_matches_unimod_255": int(baseline_255.sum()),
        "unimod_255_peptidoforms": baseline_meta.get("modification_counts", {}).get("unimod_255"),
        "sample_matches_unmodified": int((~baseline_dim).sum()),
        "spearman_percentile": 1.0,
        "spearman_dimethyl_percentile": 1.0 if baseline_dim.sum() > 1 else None,
        "median_abs_percentile_shift": 0.0,
        "median_abs_shift_dimethyl": 0.0 if baseline_dim.any() else None,
        "median_abs_shift_unmodified": 0.0 if (~baseline_dim).any() else None,
        "rt_value_definition": baseline_meta.get("rt_value", "DIA-NN stored compact-library RT"),
    })

    for index, label in enumerate(libraries, start=1):
        path = data_dir / f"library={label}" / "rt.parquet"
        print(f"Matching {label}: {path.stat().st_size / 1e9:.2f} GB", flush=True)
        matched, total_255 = read_matching_sample(path, keys_by_decoy)
        # The CARAFE data are normalized to [0,1], while DIA-NN stores another
        # RT/iRT coordinate. Percentile ranks make their pairwise ordering comparable.
        if matched.num_rows:
            matched = matched.combine_chunks()
            keys = list(zip(matched.column(KEY).to_pylist(), matched.column("decoy").to_pylist()))
            positions = np.fromiter((baseline_index[key] for key in keys), dtype=np.int64, count=len(keys))
            ref_values = baseline_rt[positions]
            target_values = numpy_column(matched, "rt", np.float64)
            dimethyl = numpy_column(matched, "has_unimod_510", bool)
            unimod_255 = numpy_column(matched, "has_unimod_255", bool)
            ref_rank = ranks(ref_values)
            target_rank = ranks(target_values)
            delta = target_rank - ref_rank
            spearman = float(np.corrcoef(ref_rank, target_rank)[0, 1]) if len(ref_rank) > 1 else float("nan")
            median_abs = float(np.median(np.abs(delta)))
            dim_abs = float(np.median(np.abs(delta[dimethyl]))) if dimethyl.any() else float("nan")
            unmod_abs = float(np.median(np.abs(delta[~dimethyl]))) if (~dimethyl).any() else float("nan")
            dimethyl_count = int(dimethyl.sum())
            unmodified_count = int((~dimethyl).sum())
            if dimethyl_count > 1:
                dim_ref_rank = ranks(ref_values[dimethyl])
                dim_target_rank = ranks(target_values[dimethyl])
                dim_spearman = float(np.corrcoef(dim_ref_rank, dim_target_rank)[0, 1])
                dim_ax = dim_axes_flat[index]
                dim_ax.hexbin(
                    dim_ref_rank, dim_target_rank, gridsize=45, mincnt=1,
                    bins="log", cmap="Oranges", linewidths=0,
                )
                dim_ax.plot([0, 1], [0, 1], color=PALETTE["neutral"], linewidth=0.65, linestyle="--")
                dim_ax.set(
                    xlim=(0, 1), ylim=(0, 1), xlabel="Baseline dimethyl RT percentile",
                    ylabel="",
                )
                dim_ax.set_xlabel("")
                dim_ax.set_title(
                    f"{textwrap.fill(label, width=24)}\nrₛ={dim_spearman:.3f}; n={dimethyl_count:,}", fontsize=9
                )
                dim_ax.grid(True, alpha=0.2)
            else:
                dim_spearman = float("nan")
                dim_axes_flat[index].set_visible(False)

            ax = axes_flat[index]
            ax.hexbin(ref_rank, target_rank, gridsize=45, mincnt=1, bins="log", cmap="Blues", linewidths=0)
            if dimethyl.any():
                # An orange outline layer marks the dimethylated subset.
                ax.scatter(ref_rank[dimethyl], target_rank[dimethyl], s=0.05, c=PALETTE["orange"], alpha=0.14, rasterized=True)
            if unimod_255.any():
                ax.scatter(ref_rank[unimod_255], target_rank[unimod_255], s=0.05, c=PALETTE["teal"], alpha=0.2, rasterized=True)
            ax.plot([0, 1], [0, 1], color=PALETTE["neutral"], linewidth=0.65, linestyle="--")
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="", ylabel="")
            wrapped_label = "\n".join(textwrap.wrap(label, width=24))
            ax.set_title(f"{wrapped_label}\nrₛ={spearman:.3f}; n={len(keys):,}", fontsize=9)
            ax.grid(True, alpha=0.2)
        else:
            spearman = median_abs = dim_abs = unmod_abs = float("nan")
            dimethyl_count = unmodified_count = 0
            dim_spearman = float("nan")
            axes_flat[index].set_visible(False)
            dim_axes_flat[index].set_visible(False)

        meta = library_meta.get(label, {})
        rows_total = parquet_rows(path)
        results.append({
            "library": label,
            "source_type": meta.get("source_type", "DIA-NN compact .speclib"),
            "completeness": meta.get("completeness", "complete"),
            "extracted_peptidoform_records": rows_total,
            "sampled_baseline_forms": len(baseline_rt),
            "sample_matches": matched.num_rows,
            "sample_match_fraction": matched.num_rows / max(len(baseline_rt), 1),
            "sample_matches_dimethyl": dimethyl_count,
            "sample_matches_unimod_255": int(unimod_255.sum()) if matched.num_rows else 0,
            "unimod_255_peptidoforms": total_255,
            "sample_matches_unmodified": unmodified_count,
            "spearman_percentile": spearman,
            "spearman_dimethyl_percentile": dim_spearman,
            "median_abs_percentile_shift": median_abs,
            "median_abs_shift_dimethyl": dim_abs,
            "median_abs_shift_unmodified": unmod_abs,
            "rt_value_definition": meta.get("rt_value", "DIA-NN stored compact-library RT"),
        })
        print(f"  matched sample {matched.num_rows:,}; r_s={spearman:.4f}; peak RSS {peak_gb():.1f} GiB", flush=True)

    figure.suptitle("Exact peptidoform RT ordering against DIA-NN baseline", fontsize=15, fontweight="bold")
    figure.supxlabel("DIA-NN baseline RT percentile", x=0.53, y=0.048, fontsize=10)
    figure.supylabel("Library RT percentile", x=0.025, y=0.52, fontsize=10)
    figure.text(
        0.5, 0.012,
        "Each point matches Modified.Sequence and the extracted decoy flag. Decoy status is not FASTA-validated; "
        "forms may include decoys. Axes use within-library percentiles; "
        "Orange marks UniMod:510; teal marks UniMod:255. Incomplete CARAFE runs are excluded unless explicitly imported.",
        ha="center", fontsize=8, color=PALETTE["neutral"],
    )
    for ax in axes_flat[len(libraries) + 1:]:
        ax.set_visible(False)
    figure.savefig(output / "all_library_rt_percentile_comparison.pdf", bbox_inches="tight")
    figure.savefig(output / "all_library_rt_percentile_comparison.png", bbox_inches="tight", dpi=300)
    plt.close(figure)
    dim_figure.suptitle("RT ordering for exact UniMod:510 peptidoforms", fontsize=15, fontweight="bold")
    dim_figure.supxlabel("Baseline dimethyl RT percentile", x=0.53, y=0.048, fontsize=10)
    dim_figure.supylabel("Library dimethyl RT percentile", x=0.025, y=0.52, fontsize=10)
    dim_figure.text(
        0.5, 0.012,
        "Only exact UniMod:510 forms shared with the baseline sample are shown; ranks are computed within this subset. "
        "Incomplete CARAFE runs are excluded unless explicitly imported.",
        ha="center", fontsize=8, color=PALETTE["neutral"],
    )
    for ax in dim_axes_flat[len(libraries) + 1:]:
        ax.set_visible(False)
    dim_figure.savefig(output / "all_library_dimethyl_rt_percentile_comparison.pdf", bbox_inches="tight")
    dim_figure.savefig(output / "all_library_dimethyl_rt_percentile_comparison.png", bbox_inches="tight", dpi=300)
    plt.close(dim_figure)

    mod255_label = "baseline_mod255"
    mod255_path = data_dir / f"library={mod255_label}" / "rt.parquet"
    if mod255_path.is_file():
        pf = pq.ParquetFile(mod255_path)
        with_255: list[np.ndarray] = []
        without_255: list[np.ndarray] = []
        for batch in pf.iter_batches(batch_size=500_000, columns=["has_unimod_255", "rt"]):
            flag = numpy_column(pa.Table.from_batches([batch]), "has_unimod_255", bool)
            rt_values = numpy_column(pa.Table.from_batches([batch]), "rt", np.float64)
            if flag.any():
                with_255.append(rt_values[flag])
            if (~flag).any():
                without_255.append(rt_values[~flag])
        if with_255 and without_255:
            fig, ax = plt.subplots(figsize=(8.4, 5.2))
            ax.hist(np.concatenate(without_255), bins=90, density=True, histtype="step", linewidth=1.4,
                    color=PALETTE["blue"], label="No UniMod:255")
            ax.hist(np.concatenate(with_255), bins=90, density=True, histtype="step", linewidth=1.6,
                    color=PALETTE["teal"], label="UniMod:255 present")
            ax.set_xlabel("Stored RT value (DIA-NN units)")
            ax.set_ylabel("Density")
            ax.set_title("RT distribution within the separate mod-255 baseline", loc="left", fontweight="bold")
            ax.grid(axis="y", alpha=0.2)
            ax.legend(frameon=False)
            fig.text(0.01, 0.005, "Within-library distribution only; UniMod:255 is reported separately from UniMod:510.",
                     fontsize=8, color=PALETTE["neutral"])
            fig.tight_layout(rect=(0, 0.025, 1, 1))
            fig.savefig(output / "baseline_mod255_rt_distribution.pdf", bbox_inches="tight")
            fig.savefig(output / "baseline_mod255_rt_distribution.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

    csv_path = output / "library_comparison_summary.csv"
    fields = list(results[0]) if results else []
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    ordered = sorted(results, key=lambda row: row["sample_match_fraction"])
    labels = [row["library"] for row in ordered]
    match_fraction = np.asarray([row["sample_match_fraction"] for row in ordered])
    colors = [PALETTE["orange"] if row["completeness"] == "partial" else PALETTE["blue"] for row in ordered]
    fig, ax = plt.subplots(figsize=(9.5, max(5.5, 0.34 * len(ordered))))
    y = np.arange(len(ordered))
    ax.barh(y, match_fraction * 100, color=colors, alpha=0.88, height=0.72)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Exact-key matches in fixed-seed baseline sample (%)")
    ax.set_title("Sampled exact-key match against DIA-NN baseline", loc="left", fontweight="bold")
    ax.set_xlim(0, 105)
    ax.grid(axis="x", alpha=0.22)
    ax.set_axisbelow(True)
    for yi, value in zip(y, match_fraction):
        ax.text(value * 100 + 0.7, yi, f"{value:.1%}", va="center", fontsize=7, color=PALETTE["dark"])
    fig.text(
        0.01, 0.005,
        "This is a key-match fraction in a fixed-seed sample, not a library-size or verified target-count comparison. "
        "Keys use Modified.Sequence and the source decoy flag; decoy status is not FASTA-validated.",
        fontsize=8, color=PALETTE["neutral"],
    )
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(output / "sampled_exact_key_match_fraction.pdf", bbox_inches="tight")
    fig.savefig(output / "sampled_exact_key_match_fraction.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    shift_rows = [row for row in results if row["library"] != args.baseline_label]
    shift_rows.sort(key=lambda row: (row["median_abs_percentile_shift"] is None, row["median_abs_percentile_shift"] or 0))
    if shift_rows:
        fig, ax = plt.subplots(figsize=(10.0, max(5.5, 0.38 * len(shift_rows))))
        y = np.arange(len(shift_rows))
        height = 0.34
        unmod = np.asarray([
            row["median_abs_shift_unmodified"]
            if row["median_abs_shift_unmodified"] is not None and math.isfinite(row["median_abs_shift_unmodified"])
            else 0
            for row in shift_rows
        ])
        dim = np.asarray([
            row["median_abs_shift_dimethyl"]
            if row["median_abs_shift_dimethyl"] is not None and math.isfinite(row["median_abs_shift_dimethyl"])
            else 0
            for row in shift_rows
        ])
        ax.barh(y - height / 2, unmod, height=height, color=PALETTE["blue"], label="No UniMod:510")
        ax.barh(y + height / 2, dim, height=height, color=PALETTE["orange"], label="UniMod:510")
        ax.set_yticks(y, [row["library"] for row in shift_rows])
        ax.set_xlabel("Median absolute RT-percentile shift vs baseline")
        ax.set_title("RT ordering changes by dimethylation state", loc="left", fontweight="bold")
        ax.grid(axis="x", alpha=0.22)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="lower right", fontsize=8)
        for yi, row in zip(y, shift_rows):
            if row["completeness"] == "partial":
                ax.get_yticklabels()[yi].set_color(PALETTE["orange"])
        fig.text(
            0.01, 0.005,
            "Statistics use exact forms in the sampled baseline intersection; incomplete CARAFE runs are excluded by default.",
            fontsize=8, color=PALETTE["neutral"],
        )
        fig.tight_layout(rect=(0, 0.025, 1, 1))
        fig.savefig(output / "library_rank_shift_by_dimethylation.pdf", bbox_inches="tight")
        fig.savefig(output / "library_rank_shift_by_dimethylation.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    summary = {
        "reference_library": args.baseline_label,
        "sampling": "fixed-seed uniform row sample; exact Modified.Sequence and decoy-flag matching",
        "decoy_handling": "All extracted records retained. Source decoy flags are grouping metadata only and are not FASTA-validated.",
        "requested_sample_size": args.sample_size,
        "actual_sample_size": len(baseline_rt),
        "seed": args.seed,
        "arrow_cpu_limit": args.cpus,
        "rank_comparison": "within matched sample; raw RT units are not compared across library formats",
        "libraries": [
            {key: (None if isinstance(value, float) and not math.isfinite(value) else value)
             for key, value in row.items()}
            for row in results
        ],
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "peak_rss_gib": round(peak_gb(), 2),
        "output_pdf": str(output / "all_library_rt_percentile_comparison.pdf"),
        "dimethyl_output_pdf": str(output / "all_library_dimethyl_rt_percentile_comparison.pdf"),
        "output_csv": str(csv_path),
        "sampled_match_plot": str(output / "sampled_exact_key_match_fraction.pdf"),
        "dimethylation_shift_plot": str(output / "library_rank_shift_by_dimethylation.pdf"),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote comparison outputs to {output}; elapsed {time.monotonic() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
