#!/usr/bin/env python3
"""Import CARAFE RTs into the same one-row-per-peptidoform RT dataset.

Completed runs are read from the final .diann2.parquet with only sequence,
decoy, precursor m/z, and RT columns. Fragment rows are collapsed. If no final
library exists, available prediction chunks are imported and their
sequence/modification fields are converted to DIA-NN Modified.Sequence.
Multiple precursor m/z variants can carry different RTs; the output stores
their median per modified sequence. Partial chunk coverage is recorded.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import resource
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA = pa.schema([
    ("modified_sequence", pa.string()),
    ("decoy", pa.bool_()),
    ("has_unimod_510", pa.bool_()),
    ("has_unimod_510_nterm", pa.bool_()),
    ("has_unimod_510_k", pa.bool_()),
    ("has_unimod_255", pa.bool_()),
    ("has_unimod_255_nterm", pa.bool_()),
    ("has_unimod_255_k", pa.bool_()),
    ("library", pa.string()),
    ("rt", pa.float32()),
])
MOD_IDS = {
    "Acetyl": 1,
    "Carbamidomethyl": 4,
    "Deamidated": 7,
    "Phospho": 21,
    "Oxidation": 35,
    "Dimethyl:2H(4)13C(2)": 510,
}
LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def modified_sequence(sequence: str, mods: str, mod_sites: str) -> str:
    if not mods:
        if mod_sites:
            raise ValueError(f"mod_sites without mods for {sequence!r}: {mod_sites!r}")
        return sequence
    mod_names = mods.split(";")
    site_strings = mod_sites.split(";") if mod_sites else []
    if len(mod_names) != len(site_strings):
        raise ValueError(f"mods/mod_sites length mismatch: {mods!r} / {mod_sites!r}")

    residue_mods: dict[int, list[int]] = {}
    terminal_n: list[int] = []
    terminal_c: list[int] = []
    for mod, site_string in zip(mod_names, site_strings):
        base, sep, residue = mod.rpartition("@")
        if not sep or base not in MOD_IDS:
            raise ValueError(f"unmapped CARAFE modification {mod!r}")
        try:
            site = int(site_string)
        except ValueError as exc:
            raise ValueError(f"invalid modification site {site_string!r} for {mod!r}") from exc
        uni = MOD_IDS[base]
        if residue == "N-term":
            terminal_n.append(uni)
        elif residue == "C-term":
            terminal_c.append(uni)
        elif 1 <= site <= len(sequence):
            if sequence[site - 1] != residue:
                raise ValueError(
                    f"site/residue mismatch for {mod!r} at {site} in {sequence!r}"
                )
            residue_mods.setdefault(site, []).append(uni)
        else:
            raise ValueError(f"site {site} is outside sequence {sequence!r} for {mod!r}")

    result = "".join(f"(UniMod:{uni})" for uni in terminal_n)
    for index, aa in enumerate(sequence, start=1):
        result += aa + "".join(f"(UniMod:{uni})" for uni in residue_mods.get(index, []))
    result += "".join(f"(UniMod:{uni})" for uni in terminal_c)
    return result


def parse_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("library must be LABEL=CARAFE_RUN_DIRECTORY")
    label, path_text = value.split("=", 1)
    if not LABEL_RE.fullmatch(label):
        raise argparse.ArgumentTypeError(f"invalid library label: {label!r}")
    path = Path(path_text)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"CARAFE run directory does not exist: {path}")
    return label, path


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"schema_version": 1, "libraries": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("libraries", {})
    return data


def write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def import_one(
    label: str, run_dir: Path, out_dir: Path, batch_size: int, max_ram_mb: int | None
) -> dict:
    chunks_dir = run_dir / "carafe_output" / "prediction_chunks"
    chunks = sorted(chunks_dir.glob("*_rt_pred.parquet"), key=lambda p: int(p.name.split("_", 1)[0]))
    total_chunks = len(list(chunks_dir.glob("peptide_forms_*.parquet")))
    if not total_chunks:
        total_chunks = 313
    final_library = run_dir / "carafe_output" / "carafe_spectral_library.diann2.parquet"
    use_final_library = final_library.is_file()
    if not use_final_library and not chunks:
        raise FileNotFoundError(f"no final CARAFE library or *_rt_pred.parquet chunks in {run_dir}")
    completeness = "complete" if use_final_library or len(chunks) == total_chunks else "partial"

    target = out_dir / "data" / f"library={label}" / "rt.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"output already exists: {target}")
    tmp = target.with_suffix(".parquet.tmp")

    rows: list[dict] = []
    # CARAFE final tables can contain multiple precursor masses for one
    # Modified.Sequence and their RT values may differ. Retain one prediction
    # per sequence/decoy/mass, then store one median RT per peptidoform.
    precursor_values: dict[tuple[str, bool], dict[float, float]] = {}
    written = 0
    source_rows = 0
    duplicate_rows = 0
    precursor_variant_rows = 0
    rt_variant_disagreements = 0
    rt_min = math.inf
    rt_max = -math.inf
    started = time.monotonic()

    def accept(key: tuple[str, bool], precursor_token: float, value: float) -> None:
        nonlocal duplicate_rows, precursor_variant_rows, rt_variant_disagreements
        if not math.isfinite(value):
            raise ValueError(f"non-finite RT for {key!r}")
        values = precursor_values.setdefault(key, {})
        if precursor_token in values:
            # Fragment rows repeat each precursor RT. Require one value for
            # one precursor mass; if source rows disagree, average those
            # repeated group summaries and record the discrepancy.
            if not math.isclose(value, values[precursor_token], rel_tol=1e-6, abs_tol=1e-6):
                rt_variant_disagreements += 1
                values[precursor_token] = (values[precursor_token] + value) / 2
            duplicate_rows += 1
            return
        values[precursor_token] = value
        precursor_variant_rows += 1

    def flush_forms(writer: pq.ParquetWriter) -> None:
        nonlocal written, rt_min, rt_max
        for key, charge_map in precursor_values.items():
            value = float(statistics.median(charge_map.values()))
            sequence, decoy = key
            has_510 = "(UniMod:510)" in sequence
            leading = re.match(r"^(?:\(UniMod:\d+\))*", sequence).group(0)
            rows.append({
                "modified_sequence": sequence,
                "decoy": decoy,
                "has_unimod_510": has_510,
                "has_unimod_510_nterm": "(UniMod:510)" in leading,
                "has_unimod_510_k": bool(re.search(r"K\(UniMod:510\)", sequence)),
                "has_unimod_255": "(UniMod:255)" in sequence,
                "has_unimod_255_nterm": "(UniMod:255)" in leading,
                "has_unimod_255_k": bool(re.search(r"K\(UniMod:255\)", sequence)),
                "library": label,
                "rt": value,
            })
            written += 1
            rt_min = min(rt_min, value)
            rt_max = max(rt_max, value)
            if len(rows) >= batch_size:
                writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
                rows.clear()
        precursor_values.clear()

    with pq.ParquetWriter(tmp, SCHEMA, compression="zstd", compression_level=3) as writer:
        if use_final_library:
            pf = pq.ParquetFile(final_library)
            expected = pf.metadata.num_rows
            for batch_no, batch in enumerate(pf.iter_batches(
                batch_size=batch_size,
                columns=["Modified.Sequence", "Decoy", "Precursor.Mz", "RT"],
            ), start=1):
                source_rows += batch.num_rows
                table = pa.Table.from_batches([batch])
                grouped = table.group_by(["Modified.Sequence", "Decoy", "Precursor.Mz"]).aggregate(
                    [("RT", "min"), ("RT", "max"), ("RT", "mean")]
                )
                duplicate_rows += table.num_rows - grouped.num_rows
                for record in grouped.to_pylist():
                    value = float(record["RT_mean"])
                    if not math.isclose(float(record["RT_min"]), float(record["RT_max"]), rel_tol=1e-6, abs_tol=1e-6):
                        rt_variant_disagreements += 1
                    accept(
                        (record["Modified.Sequence"], bool(record["Decoy"])),
                        round(float(record["Precursor.Mz"]), 6),
                        value,
                    )
                if batch_no % 20 == 0 or source_rows >= expected:
                    rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
                    if max_ram_mb is not None and rss_mib > max_ram_mb:
                        raise MemoryError(
                            f"{label}: peak RSS {rss_mib:.0f} MiB exceeded soft guard {max_ram_mb} MiB"
                        )
                    print(
                        f"{label}: final-library rows {source_rows:,}/{expected:,}; "
                        f"{len(precursor_values):,} forms indexed; RSS peak {rss_mib:.0f} MiB; "
                        f"{time.monotonic() - started:.0f}s",
                        flush=True,
                    )
        else:
            for chunk_no, path in enumerate(chunks, start=1):
                pf = pq.ParquetFile(path)
                for batch in pf.iter_batches(
                    batch_size=batch_size,
                    columns=["sequence", "mods", "mod_sites", "mz", "rt_norm_pred"],
                ):
                    cols = [batch.column(i).to_pylist() for i in range(batch.num_columns)]
                    for sequence, mods, sites, mz, rt in zip(*cols):
                        source_rows += 1
                        if rt is None:
                            raise ValueError(f"null normalized RT in {path}")
                        # Each pepID's precursor charge has a distinct m/z; use
                        # a stable rounded value to collapse its repeated rows.
                        accept(
                            (modified_sequence(sequence, mods or "", sites or ""), False),
                            round(float(mz), 6),
                            float(rt),
                        )
                print(
                    f"{label}: chunk {chunk_no}/{len(chunks)}; {len(precursor_values):,} forms indexed; "
                    f"{time.monotonic() - started:.0f}s",
                    flush=True,
                )
                rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
                if max_ram_mb is not None and rss_mib > max_ram_mb:
                    raise MemoryError(
                        f"{label}: peak RSS {rss_mib:.0f} MiB exceeded soft guard {max_ram_mb} MiB"
                    )
        flush_forms(writer)
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
    os.replace(tmp, target)
    return {
        "source_type": "CARAFE final library" if use_final_library else "CARAFE partial prediction chunks",
        "source_run_directory": str(run_dir),
        "source_file": str(final_library) if use_final_library else None,
        "source_chunk_count": len(chunks),
        "expected_chunk_count": total_chunks,
        "completeness": completeness,
        "source_rows_read": source_rows,
        "peptidoforms_written": written,
        "precursor_variant_rows": precursor_variant_rows,
        "rows_collapsed": source_rows - precursor_variant_rows,
        "duplicate_rows_collapsed": duplicate_rows,
        "rt_variant_disagreements": rt_variant_disagreements,
        "precursor_mz_variants_median_collapsed": precursor_variant_rows - written,
        "rt_value": (
            "median RT across distinct precursor m/z values per Modified.Sequence and Decoy from final .diann2.parquet"
            if use_final_library else "median rt_norm_pred across distinct CARAFE precursor m/z values; normalized coordinate (0..1) from incomplete prediction chunks"
        ),
        "decoy_handling": (
            "source Decoy flag retained; both true and false values included; not FASTA-validated"
            if use_final_library else "source chunks lack a validated decoy flag; all rows retained with status unknown"
        ),
        "rt_min": rt_min,
        "rt_max": rt_max,
        "output_file": str(target),
        "output_size_bytes": target.stat().st_size,
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--library", action="append", required=True, type=parse_spec)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--cpus", type=int, default=4, help="Arrow CPU limit for streamed group-by operations")
    parser.add_argument("--max-ram-mb", type=int, help="soft RSS guard checked between batches/chunks")
    parser.add_argument("--skip-existing", action="store_true", help="skip labels already recorded in manifest.json")
    parser.add_argument(
        "--include-partial",
        action="store_true",
        help="import incomplete chunk-only CARAFE runs; omitted by default",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.cpus < 1 or (args.max_ram_mb is not None and args.max_ram_mb < 1):
        parser.error("--batch-size, --cpus, and --max-ram-mb must be positive")
    pa.set_cpu_count(args.cpus)
    labels = [label for label, _ in args.library]
    if len(labels) != len(set(labels)):
        parser.error("library labels must be unique")
    manifest_path = args.out_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    manifest.setdefault("carafe_libraries", {})
    manifest["last_carafe_import_settings"] = {
        "batch_size": args.batch_size,
        "arrow_cpu_limit": args.cpus,
        "max_ram_mb_soft_guard": args.max_ram_mb,
        "include_partial": args.include_partial,
    }
    write_json_atomic(manifest_path, manifest)
    for label, run_dir in args.library:
        if label in manifest.get("libraries", {}) or label in manifest["carafe_libraries"]:
            if args.skip_existing:
                print(f"Skipping already imported library {label}", flush=True)
                continue
            raise ValueError(f"library label already present in manifest: {label}")
        final_library = run_dir / "carafe_output" / "carafe_spectral_library.diann2.parquet"
        chunks_dir = run_dir / "carafe_output" / "prediction_chunks"
        chunks = list(chunks_dir.glob("*_rt_pred.parquet"))
        if not final_library.is_file() and not chunks:
            raise FileNotFoundError(f"no final CARAFE library or *_rt_pred.parquet chunks in {run_dir}")
        expected_chunks = len(list(chunks_dir.glob("peptide_forms_*.parquet"))) or 313
        is_partial = not final_library.is_file() and len(chunks) != expected_chunks
        if is_partial and not args.include_partial:
            print(
                f"Skipping incomplete CARAFE run {label}: {len(chunks)}/{expected_chunks} chunks; "
                "pass --include-partial to import it",
                flush=True,
            )
            continue
        print(f"Importing {label} from {run_dir}", flush=True)
        result = import_one(label, run_dir, args.out_dir, args.batch_size, args.max_ram_mb)
        manifest["carafe_libraries"][label] = result
        write_json_atomic(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
