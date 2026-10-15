#!/usr/bin/env python3
"""Extract one RT value per modified peptide form from DIA-NN compact libraries.

This reads DIA-NN .speclib files directly. It does not call DIA-NN and does not
decode or write fragment, IM, precursor m/z, or other prediction fields.
Multiple libraries can be stored as a small Parquet dataset in one output
directory; later invocations can add new library labels without rewriting old
Parquet files.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import logging
import math
import os
import re
import resource
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

import pyarrow as pa
import pyarrow.parquet as pq

from .parse_diann_speclib import (
    F32,
    PRECURSOR_PREFIX_BYTES,
    POST_ID_BYTES,
    FRAGMENT_BYTES,
    MAX_PRECURSOR_ID_BYTES,
    U32,
    FormatError,
    decode_precursor_id,
    find_first_record,
    read_exact,
    read_u32,
)


SCHEMA_VERSION = 2
OUTPUT_SCHEMA = pa.schema(
    [
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
    ]
)
LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def log_string(fh: BinaryIO, *, max_len: int) -> str:
    n = read_u32(fh)
    if n > max_len:
        raise FormatError(f"implausible string length {n} at offset {fh.tell() - 4}")
    try:
        return read_exact(fh, n).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FormatError(f"invalid UTF-8 string at offset {fh.tell() - n}") from exc


def read_header_without_protein_rows(fh: BinaryIO) -> tuple[dict, int]:
    """Read header and skip proteins without retaining their rows in RAM."""
    fh.seek(0)
    fixed = struct.unpack("<iiii", read_exact(fh, 16))
    version, *header_words = fixed
    if version not in (-10, -11):
        if version == -3:
            raise FormatError("Skyline sidecar (-3) is not a standalone compact .speclib")
        raise FormatError(f"unsupported compact .speclib version {version}; expected -10 or -11")

    reserved = read_u32(fh)
    fasta_path = log_string(fh, max_len=1_000_000)
    protein_header = struct.unpack("<III", read_exact(fh, 12))
    protein_count, trailer1, trailer2 = protein_header
    if protein_count > 10_000_000:
        raise FormatError(f"implausible protein count {protein_count}")

    for _ in range(protein_count):
        for _ in range(3):
            n = read_u32(fh)
            if n > 4096:
                raise FormatError(f"implausible protein string length {n} at offset {fh.tell() - 4}")
            fh.seek(n, os.SEEK_CUR)
        fh.seek(16, os.SEEK_CUR)

    meta = {
        "format_version": version,
        "header_words": header_words,
        "reserved_u32": reserved,
        "fasta_path_in_header": fasta_path,
        "protein_count": protein_count,
        "header_trailer_words": [trailer1, trailer2],
        "protein_table_end_offset": fh.tell(),
    }
    return meta, fh.tell()


def iter_rt_records(
    fh: BinaryIO,
    first_record: int,
    file_size: int,
    end_offset: int | None = None,
) -> Iterator[tuple[str, bool, float, int, int]]:
    """Yield modified sequence, decoy flag, RT, charge, and next record offset."""
    if end_offset is None:
        end_offset = file_size
    record_prefix = first_record
    while record_prefix < end_offset and record_prefix + PRECURSOR_PREFIX_BYTES < file_size:
        fh.seek(record_prefix)
        n = read_u32(fh)
        if not 6 <= n <= MAX_PRECURSOR_ID_BYTES:
            return
        raw_id = read_exact(fh, n)
        decoded = decode_precursor_id(raw_id)
        if decoded is None:
            return
        _precursor_id, modified_sequence, _stripped_sequence, charge, decoy = decoded

        fh.seek(record_prefix - PRECURSOR_PREFIX_BYTES)
        prefix = read_exact(fh, PRECURSOR_PREFIX_BYTES)
        _precursor_index, raw_charge, peptide_length = struct.unpack_from("<III", prefix, 0)
        rt = F32.unpack_from(prefix, 16)[0]
        if raw_charge != charge or peptide_length != len(_stripped_sequence):
            raise FormatError(
                f"precursor prefix disagrees with ID at offset {record_prefix}: "
                f"charge {raw_charge}/{charge}, length {peptide_length}/{len(_stripped_sequence)}"
            )
        if not math.isfinite(rt):
            raise FormatError(f"non-finite RT at offset {record_prefix}: {rt}")

        after_id = record_prefix + 4 + n
        fh.seek(after_id + POST_ID_BYTES)
        n_fragments = read_u32(fh)
        fragments_start = fh.tell()
        next_record = fragments_start + n_fragments * FRAGMENT_BYTES + PRECURSOR_PREFIX_BYTES
        if n_fragments > 256 or next_record > file_size:
            raise FormatError(f"invalid fragment count {n_fragments} at {record_prefix}")
        if next_record > end_offset:
            raise FormatError(
                f"record at {record_prefix} crosses segment boundary {end_offset}; "
                "segment endpoints must be record starts"
            )
        yield modified_sequence, decoy, rt, charge, next_record
        record_prefix = next_record


def peak_rss_mb() -> float:
    # Linux reports ru_maxrss in KiB.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value) / 1024.0


def set_cpu_limit(cpu_count: int) -> list[int] | None:
    """Restrict process affinity where Linux exposes sched_setaffinity."""
    pa.set_cpu_count(cpu_count)
    try:
        available = sorted(os.sched_getaffinity(0))
        selected = available[:cpu_count]
        os.sched_setaffinity(0, selected)
        return selected
    except (AttributeError, OSError):
        return None


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"schema_version": SCHEMA_VERSION, "libraries": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise FormatError(f"manifest schema version {data.get('schema_version')} is not supported")
    data.setdefault("libraries", {})
    return data


def write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_worker_progress(path: Path | None, data: dict) -> None:
    if path is not None:
        write_json_atomic(path, data)


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    whole = int(seconds)
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def locate_form_limit(fh: BinaryIO, first_record: int, file_size: int, limit: int) -> int:
    """Find the exclusive record offset after `limit` complete peptide forms."""
    previous_key: tuple[str, bool] | None = None
    completed_forms = 0
    record_offset = first_record
    for sequence, decoy, _rt, _charge, next_record in iter_rt_records(fh, first_record, file_size):
        key = (sequence, decoy)
        if key != previous_key:
            if previous_key is not None:
                completed_forms += 1
                if completed_forms >= limit:
                    return record_offset
            previous_key = key
        record_offset = next_record
    if completed_forms + (1 if previous_key is not None else 0) < limit:
        raise FormatError(f"library ended before the requested {limit:,} peptide forms")
    return record_offset


def align_boundary_to_form(fh: BinaryIO, candidate: int, file_size: int) -> int:
    """Move a valid record boundary forward to the next modified-form boundary."""
    first_key: tuple[str, bool] | None = None
    record_offset = candidate
    for sequence, decoy, _rt, _charge, next_record in iter_rt_records(fh, candidate, file_size):
        key = (sequence, decoy)
        if first_key is None:
            first_key = key
        elif key != first_key:
            return record_offset
        record_offset = next_record
    raise FormatError(f"could not find a peptide-form transition after byte offset {candidate}")


def validate_record_chain(fh: BinaryIO, candidate: int, file_size: int, *, records: int = 8) -> bool:
    """Require several consecutive, structurally valid records at a candidate offset."""
    previous_index: int | None = None
    record_offset = candidate
    validated = 0
    try:
        for _sequence, _decoy, _rt, _charge, next_record in iter_rt_records(
            fh, candidate, file_size
        ):
            fh.seek(record_offset - PRECURSOR_PREFIX_BYTES)
            precursor_index = U32.unpack(read_exact(fh, 4))[0]
            if previous_index is not None and precursor_index != ((previous_index + 1) & 0xFFFFFFFF):
                return False
            previous_index = precursor_index
            validated += 1
            record_offset = next_record
            if validated >= records:
                return True
    except (FormatError, OSError):
        return False
    return False


def resolve_first_record(
    fh: BinaryIO,
    protein_end: int,
    file_size: int,
    override: int | None,
) -> int:
    """Use a supplied byte offset only after validating its position and record chain."""
    if override is None:
        return find_first_record(fh, protein_end, file_size)
    if override < protein_end or override >= file_size:
        raise FormatError(
            f"first-record offset {override} is outside the precursor region "
            f"[{protein_end}, {file_size})"
        )
    if not validate_record_chain(fh, override, file_size):
        raise FormatError(
            f"first-record offset {override} failed the 8-record structural check; "
            "remove --first-record-offset to use automatic scanning"
        )
    return override


def discover_chunk_bounds(
    fh: BinaryIO,
    *,
    protein_end: int,
    file_size: int,
    chunk_count: int,
    max_peptidoforms: int | None,
    first_record_override: int | None = None,
) -> list[tuple[int, int]]:
    """Find validated record starts and align each split to a peptide-form change."""
    first_record = resolve_first_record(fh, protein_end, file_size, first_record_override)
    if not validate_record_chain(fh, first_record, file_size):
        raise FormatError(f"first-record candidate at {first_record} failed the 8-record chain check")
    region_end = (
        locate_form_limit(fh, first_record, file_size, max_peptidoforms)
        if max_peptidoforms is not None
        else file_size
    )
    if chunk_count == 1:
        return [(first_record, region_end)]

    boundaries = [first_record]
    span = region_end - first_record
    for index in range(1, chunk_count):
        target = first_record + span * index // chunk_count
        candidate = find_first_record(fh, target, file_size)
        if not validate_record_chain(fh, candidate, file_size):
            raise FormatError(f"record candidate at {candidate} failed the 8-record chain check")
        boundary = align_boundary_to_form(fh, candidate, file_size)
        if boundary <= boundaries[-1] or boundary >= region_end:
            raise FormatError(
                f"could not make {chunk_count} non-empty chunks: boundary {boundary} "
                f"is outside ({boundaries[-1]}, {region_end})"
            )
        boundaries.append(boundary)
    boundaries.append(region_end)
    return list(zip(boundaries, boundaries[1:]))


def merge_parquet_parts(
    parts: list[Path],
    output_path: Path,
    *,
    progress_callback=None,
) -> int:
    """Merge ordered temporary Parquet chunks into one atomically replaced file."""
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.unlink(missing_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        total_rows = sum(pq.ParquetFile(part).metadata.num_rows for part in parts)
        rows_written = 0
        with pq.ParquetWriter(tmp_path, OUTPUT_SCHEMA, compression="zstd", compression_level=3) as writer:
            for part in parts:
                parquet = pq.ParquetFile(part)
                for batch in parquet.iter_batches(batch_size=25_000):
                    writer.write_batch(batch)
                    rows_written += batch.num_rows
                    if progress_callback is not None:
                        progress_callback(rows_written, total_rows)
        os.replace(tmp_path, output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return output_path.stat().st_size


def combine_segment_metadata(
    label: str,
    input_path: Path,
    output_path: Path,
    segment_results: list[dict],
    segment_bounds: list[tuple[int, int]],
    *,
    output_size_bytes: int,
    elapsed_seconds: float,
    limited_run: bool,
) -> dict:
    """Aggregate chunk counters while retaining the validated byte ranges."""
    metadata = dict(segment_results[0])
    sum_fields = (
        "precursor_rows_read",
        "peptidoforms_read",
        "charge_rows_collapsed",
        "mod_filtered_peptidoforms",
        "peptidoforms_written",
    )
    for field in sum_fields:
        metadata[field] = sum(result[field] for result in segment_results)
    metadata["input_file"] = str(input_path.resolve())
    metadata["input_size_bytes"] = input_path.stat().st_size
    metadata["first_precursor_record_offset"] = segment_bounds[0][0]
    metadata["parsed_record_end_offset"] = segment_results[-1]["parsed_record_end_offset"]
    metadata["trailing_bytes_after_record_block"] = max(
        0, metadata["input_size_bytes"] - metadata["parsed_record_end_offset"]
    )
    metadata["rt_min"] = min(result["rt_min"] for result in segment_results if result["rt_min"] is not None)
    metadata["rt_max"] = max(result["rt_max"] for result in segment_results if result["rt_max"] is not None)
    metadata["output_file"] = str(output_path.resolve())
    metadata["output_size_bytes"] = output_size_bytes
    metadata["limited_run"] = limited_run
    metadata["elapsed_seconds"] = round(elapsed_seconds, 3)
    metadata["peak_rss_mb"] = max(result["peak_rss_mb"] for result in segment_results)
    metadata["chunk_ranges"] = [
        {"start_offset": start, "end_offset": end, "peptidoforms": result["peptidoforms_read"]}
        for (start, end), result in zip(segment_bounds, segment_results)
    ]
    return metadata


def parse_library(
    label: str,
    input_path: Path,
    output_path: Path,
    *,
    batch_rows: int,
    max_peptidoforms: int | None,
    only_mod_ids: set[str],
    max_ram_mb: int | None,
    logger: logging.Logger,
    first_record_override: int | None = None,
    end_record_offset: int | None = None,
    progress_path: Path | None = None,
    progress_chunk_index: int = 0,
) -> dict:
    input_path = input_path.resolve()
    file_size = input_path.stat().st_size
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict] = []
    record_count = 0
    record_end_offset = 0
    form_count = 0
    written_forms = 0
    collapsed_charge_rows = 0
    mod_filtered_forms = 0
    min_rt: float | None = None
    max_rt: float | None = None
    start_time = time.monotonic()
    last_progress_write = start_time
    progress_first_record: int | None = None
    progress_end = end_record_offset or file_size

    def persist_progress(state: str, *, error: str | None = None) -> None:
        now = time.monotonic()
        start_offset = progress_first_record
        total_bytes = max(0, progress_end - start_offset) if start_offset is not None else 0
        current_offset = min(progress_end, max(start_offset or 0, record_end_offset))
        write_worker_progress(
            progress_path,
            {
                "label": label,
                "chunk_index": progress_chunk_index,
                "state": state,
                "start_offset": start_offset,
                "end_offset": progress_end,
                "current_offset": current_offset,
                "bytes_total": total_bytes,
                "bytes_done": max(0, current_offset - (start_offset or 0)),
                "precursor_rows_read": record_count,
                "peptidoforms_read": form_count,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(now - start_time, 1),
                "error": error,
            },
        )

    try:
        with input_path.open("rb") as fh:
            metadata, protein_end = read_header_without_protein_rows(fh)
            first_record = resolve_first_record(fh, protein_end, file_size, first_record_override)
            progress_first_record = first_record
            record_end_offset = first_record
            persist_progress("running")
            metadata.update(
                {
                    "input_file": str(input_path),
                    "input_size_bytes": file_size,
                    "first_precursor_record_offset": first_record,
                }
            )

            def flush_rows(writer: pq.ParquetWriter) -> None:
                nonlocal output_rows
                if not output_rows:
                    return
                writer.write_table(pa.Table.from_pylist(output_rows, schema=OUTPUT_SCHEMA))
                output_rows = []
                if max_ram_mb is not None:
                    rss = peak_rss_mb()
                    if rss > max_ram_mb:
                        raise MemoryError(
                            f"RSS guard reached: {rss:.0f} MiB > configured {max_ram_mb} MiB"
                        )

            previous_key: tuple[str, bool] | None = None
            previous_rt: float | None = None
            current_has_mod_510 = False
            current_has_mod_255 = False
            current_selected_by_mod = False
            current_should_write = False
            record_end_offset = first_record

            with pq.ParquetWriter(tmp_path, OUTPUT_SCHEMA, compression="zstd", compression_level=3) as writer:
                for modified_sequence, decoy, rt, _charge, next_record in iter_rt_records(
                    fh, first_record, file_size, end_record_offset
                ):
                    record_count += 1
                    record_end_offset = next_record
                    key = (modified_sequence, decoy)
                    if key != previous_key:
                        if previous_key is not None:
                            form_count += 1
                            if current_should_write:
                                output_rows.append(
                                    {
                                        "modified_sequence": previous_key[0],
                                        "decoy": previous_key[1],
                                        "has_unimod_510": current_has_mod_510,
                                        "has_unimod_510_nterm": current_has_mod_510_nterm,
                                        "has_unimod_510_k": current_has_mod_510_k,
                                        "has_unimod_255": current_has_mod_255,
                                        "has_unimod_255_nterm": current_has_mod_255_nterm,
                                        "has_unimod_255_k": current_has_mod_255_k,
                                        "library": label,
                                        "rt": previous_rt,
                                    }
                                )
                                written_forms += 1
                                if len(output_rows) >= batch_rows:
                                    flush_rows(writer)
                            if max_peptidoforms is not None and form_count >= max_peptidoforms:
                                break

                        previous_key = key
                        previous_rt = rt
                        current_has_mod_510 = "(UniMod:510)" in modified_sequence
                        current_has_mod_255 = "(UniMod:255)" in modified_sequence
                        leading_mods = re.match(r"^(?:\(UniMod:\d+\))*", modified_sequence).group(0)
                        current_has_mod_510_nterm = "(UniMod:510)" in leading_mods
                        current_has_mod_510_k = bool(re.search(r"K\(UniMod:510\)", modified_sequence))
                        current_has_mod_255_nterm = "(UniMod:255)" in leading_mods
                        current_has_mod_255_k = bool(re.search(r"K\(UniMod:255\)", modified_sequence))
                        mods = set(re.findall(r"\(UniMod:(\d+)\)", modified_sequence))
                        current_selected_by_mod = not only_mod_ids or bool(mods & only_mod_ids)
                        current_should_write = current_selected_by_mod
                        if not current_selected_by_mod:
                            mod_filtered_forms += 1
                        min_rt = rt if min_rt is None else min(min_rt, rt)
                        max_rt = rt if max_rt is None else max(max_rt, rt)
                    else:
                        collapsed_charge_rows += 1
                        if rt != previous_rt:
                            raise FormatError(
                                "RT differs across charge states for one modified peptide form: "
                                f"{modified_sequence!r}: {previous_rt} vs {rt}"
                            )

                    if record_count % 1_000_000 == 0:
                        logger.info(
                            "%s: %s precursor rows, %s peptide forms, input offset %.2f/%.2f GB, RSS %.0f MiB",
                            label,
                            f"{record_count:,}",
                            f"{form_count:,}",
                            next_record / 1e9,
                            file_size / 1e9,
                            peak_rss_mb(),
                        )

                    if progress_path is not None and time.monotonic() - last_progress_write >= 5:
                        persist_progress("running")
                        last_progress_write = time.monotonic()

                # Flush the last group unless max_peptidoforms stopped at its boundary.
                if previous_key is not None and (max_peptidoforms is None or form_count < max_peptidoforms):
                    form_count += 1
                    if current_should_write:
                        output_rows.append(
                            {
                                "modified_sequence": previous_key[0],
                                "decoy": previous_key[1],
                                "has_unimod_510": current_has_mod_510,
                                "has_unimod_510_nterm": current_has_mod_510_nterm,
                                "has_unimod_510_k": current_has_mod_510_k,
                                "has_unimod_255": current_has_mod_255,
                                "has_unimod_255_nterm": current_has_mod_255_nterm,
                                "has_unimod_255_k": current_has_mod_255_k,
                                "library": label,
                                "rt": previous_rt,
                            }
                        )
                        written_forms += 1
                        if len(output_rows) >= batch_rows:
                            flush_rows(writer)
                flush_rows(writer)

        if form_count == 0:
            raise FormatError("no precursor/peptidoform rows were found")
        os.replace(tmp_path, output_path)
    except Exception as exc:
        persist_progress("failed", error=f"{type(exc).__name__}: {exc}")
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    elapsed = time.monotonic() - start_time
    metadata.update(
        {
            "precursor_rows_read": record_count,
            "parsed_record_end_offset": record_end_offset,
            "trailing_bytes_after_record_block": max(0, file_size - record_end_offset),
            "peptidoforms_read": form_count,
            "charge_rows_collapsed": collapsed_charge_rows,
            "mod_filtered_peptidoforms": mod_filtered_forms,
            "peptidoforms_written": written_forms,
            "only_mod_ids": sorted(only_mod_ids),
            "rt_min": min_rt,
            "rt_max": max_rt,
            "output_file": str(output_path.resolve()),
            "output_size_bytes": output_path.stat().st_size,
            "limited_run": max_peptidoforms is not None,
            "elapsed_seconds": round(elapsed, 3),
            "peak_rss_mb": round(peak_rss_mb(), 1),
        }
    )
    logger.info(
        "%s done: %s rows -> %s peptidoforms written; %.1f MiB peak RSS; %.1f s; output %.2f MB",
        label,
        f"{record_count:,}",
        f"{written_forms:,}",
        metadata["peak_rss_mb"],
        elapsed,
        metadata["output_size_bytes"] / 1e6,
    )
    persist_progress("completed")
    return metadata


def parse_library_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("library must be LABEL=/path/to/library.speclib")
    label, filename = value.split("=", 1)
    if not LABEL_RE.fullmatch(label):
        raise argparse.ArgumentTypeError("LABEL may contain letters, numbers, dot, underscore, and hyphen")
    path = Path(filename).expanduser()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"library file does not exist: {path}")
    return label, path


def parse_offset_spec(value: str) -> tuple[str, int]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("offset must be LABEL=BYTE_OFFSET")
    label, raw_offset = value.split("=", 1)
    if not LABEL_RE.fullmatch(label):
        raise argparse.ArgumentTypeError("LABEL may contain letters, numbers, dot, underscore, and hyphen")
    try:
        offset = int(raw_offset)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("BYTE_OFFSET must be a non-negative integer") from exc
    if offset < 0:
        raise argparse.ArgumentTypeError("BYTE_OFFSET must be a non-negative integer")
    return label, offset


def parse_library_worker(
    label: str,
    chunk_index: int,
    input_path: Path,
    output_path: Path,
    first_record: int | None,
    end_record_offset: int,
    *,
    batch_rows: int,
    max_peptidoforms: int | None,
    only_mod_ids: set[str],
    max_ram_mb: int | None,
    cpu_ids: list[int] | None,
    cpu_count: int,
    log_path: Path,
    log_level: str,
    progress_path: Path | None = None,
) -> tuple[str, int, dict]:
    """Run one independent library chunk in a worker process."""
    if cpu_ids:
        os.sched_setaffinity(0, set(cpu_ids))
        pa.set_cpu_count(len(cpu_ids))
    else:
        pa.set_cpu_count(cpu_count)

    logger = logging.getLogger(f"extract_diann_rt.{label}.chunk{chunk_index}")
    logger.setLevel(getattr(logging, log_level))
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.info(
        "Worker chunk %d CPU affinity: %s; Arrow CPU count: %d; bytes [%s, %d)",
        chunk_index,
        cpu_ids,
        pa.cpu_count(),
        first_record,
        end_record_offset,
    )
    result = parse_library(
        label,
        input_path,
        output_path,
        batch_rows=batch_rows,
        max_peptidoforms=max_peptidoforms,
        only_mod_ids=only_mod_ids,
        max_ram_mb=max_ram_mb,
        logger=logger,
        first_record_override=first_record,
        end_record_offset=end_record_offset,
        progress_path=progress_path,
        progress_chunk_index=chunk_index,
    )
    return label, chunk_index, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path, help="experiment output folder")
    parser.add_argument(
        "--library",
        action="append",
        required=True,
        type=parse_library_spec,
        metavar="LABEL=PATH",
        help="library to extract; repeat to process baseline and tuned, or add a new label later",
    )
    parser.add_argument(
        "--first-record-offset",
        action="append",
        default=[],
        type=parse_offset_spec,
        metavar="LABEL=BYTE_OFFSET",
        help=(
            "optional validated precursor start offset for a library label; skips scanning the "
            "protein-to-precursor gap (repeatable; otherwise auto-detected)"
        ),
    )
    parser.add_argument("--replace-library", action="store_true", help="replace existing output for these labels")
    parser.add_argument("--max-peptidoforms", type=int, help="stop after this many peptide forms (small trial only)")
    parser.add_argument("--batch-rows", type=int, default=25_000, help="Parquet write batch size")
    parser.add_argument("--max-ram-mb", type=int, help="soft RSS guard; abort between batches if exceeded")
    parser.add_argument("--cpus", type=int, default=1, help="total CPU affinity limit (default: 1)")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="libraries to parse simultaneously; useful for baseline/tuned pairs (default: 1)",
    )
    parser.add_argument(
        "--chunks-per-library",
        type=int,
        default=1,
        help="split each binary library into this many record-aligned byte ranges (default: 1)",
    )
    parser.add_argument(
        "--only-mod-id",
        action="append",
        default=[],
        help="keep peptide forms containing this UniMod ID; repeatable (default: keep all forms)",
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    args = parser.parse_args()

    if args.batch_rows < 1 or args.cpus < 1 or args.jobs < 1 or args.chunks_per_library < 1:
        parser.error("--batch-rows, --cpus, --jobs, and --chunks-per-library must be positive")
    if args.jobs > args.cpus:
        parser.error("--jobs cannot exceed --cpus")
    if args.max_peptidoforms is not None and args.max_peptidoforms < 1:
        parser.error("--max-peptidoforms must be positive")
    if args.max_ram_mb is not None and args.max_ram_mb < 128:
        parser.error("--max-ram-mb must be at least 128 MiB; it is a soft guard, not an OS hard limit")

    labels = [label for label, _ in args.library]
    if len(set(labels)) != len(labels):
        parser.error("library labels must be unique within one invocation")
    offset_labels = [label for label, _offset in args.first_record_offset]
    if len(set(offset_labels)) != len(offset_labels):
        parser.error("first-record offset labels must be unique")
    unknown_offset_labels = set(offset_labels) - set(labels)
    if unknown_offset_labels:
        parser.error(f"first-record offsets refer to unknown library labels: {sorted(unknown_offset_labels)}")
    first_record_offsets = dict(args.first_record_offset)

    out_dir = args.out_dir.expanduser().resolve()
    data_dir = out_dir / "data"
    log_dir = out_dir / "logs"
    data_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"extract_{timestamp}.log"
    logger = logging.getLogger("extract_diann_rt")
    logger.setLevel(getattr(logging, args.log_level))
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    affinity = set_cpu_limit(args.cpus)
    logger.info("Output directory: %s", out_dir)
    logger.info("CPU limit: %d; effective affinity: %s", args.cpus, affinity)
    logger.info(
        "Parallel library jobs: %d; chunks per library: %d; batch rows: %s; soft RSS guard per worker: %s MiB",
        args.jobs,
        args.chunks_per_library,
        f"{args.batch_rows:,}",
        args.max_ram_mb,
    )

    progress_path = out_dir / "progress.json"
    worker_progress_dir = out_dir / ".progress" / timestamp
    worker_progress_dir.mkdir(parents=True, exist_ok=True)
    progress_steps = [
        {"name": "Prepare run and check input files", "state": "pending"},
        {"name": "Verify precursor offsets and find chunk boundaries", "state": "pending"},
        {"name": "Extract peptide RTs", "state": "pending"},
        {"name": "Merge chunk Parquet files", "state": "pending"},
        {"name": "Write final manifest", "state": "pending"},
    ]
    progress_libraries = {
        label: {
            "state": "queued",
            "chunks_total": 0,
            "bytes_total": 0,
            "bytes_done": 0,
            "precursor_rows_read": 0,
            "peptidoforms_read": 0,
        }
        for label, _path in args.library
    }
    progress_worker_paths: dict[str, list[Path]] = {label: [] for label, _path in args.library}
    progress_step_index = 0
    progress_started = time.monotonic()
    extraction_started: float | None = None
    last_progress_log = 0.0

    def update_progress(*, force_log: bool = False) -> None:
        nonlocal last_progress_log

        def progress_bar(value: float | None, width: int = 12) -> str:
            if value is None:
                return "[------------]"
            filled = min(width, max(0, int(width * value / 100.0)))
            return f"[{('#' * filled) + ('.' * (width - filled))}]"

        total_bytes = 0
        done_bytes = 0
        library_snapshots: dict[str, dict] = {}
        for label, state in progress_libraries.items():
            worker_states = []
            for status_path in progress_worker_paths[label]:
                try:
                    worker_states.append(json.loads(status_path.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError):
                    continue
            library_total = int(state.get("bytes_total", 0))
            library_done = min(
                library_total,
                sum(int(worker.get("bytes_done", 0)) for worker in worker_states),
            )
            if state["state"] == "completed":
                library_done = library_total
            library_rows = sum(int(worker.get("precursor_rows_read", 0)) for worker in worker_states)
            library_forms = sum(int(worker.get("peptidoforms_read", 0)) for worker in worker_states)
            chunks_done = sum(worker.get("state") == "completed" for worker in worker_states)
            library_percent = 100.0 * library_done / library_total if library_total else None
            library_eta = None
            if (
                library_percent
                and library_percent > 0
                and extraction_started is not None
                and progress_step_index == 2
            ):
                fraction = min(library_percent / 100.0, 0.9999)
                library_eta = (time.monotonic() - extraction_started) * (1.0 - fraction) / fraction
            total_bytes += library_total
            done_bytes += library_done
            library_snapshots[label] = {
                **state,
                "percent": round(library_percent, 2) if library_percent is not None else None,
                "bytes_done": library_done,
                "precursor_rows_read": library_rows,
                "peptidoforms_read": library_forms,
                "chunks_done": chunks_done,
                "estimated_remaining_seconds": (
                    round(library_eta, 1) if library_eta is not None else None
                ),
                "worker_states": [worker.get("state", "starting") for worker in worker_states],
            }

        percent = 100.0 * done_bytes / total_bytes if total_bytes else None
        elapsed = time.monotonic() - progress_started
        eta = None
        if (
            percent
            and percent > 0
            and extraction_started is not None
            and progress_step_index == 2
        ):
            extraction_elapsed = time.monotonic() - extraction_started
            fraction = min(percent / 100.0, 0.9999)
            eta = extraction_elapsed * (1.0 - fraction) / fraction
        snapshot = {
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(out_dir),
            "current_step": progress_steps[progress_step_index]["name"],
            "steps": progress_steps,
            "overall": {
                "state": progress_steps[progress_step_index]["state"],
                "percent_by_input_bytes": round(percent, 2) if percent is not None else None,
                "elapsed_seconds": round(elapsed, 1),
                "estimated_remaining_seconds": round(eta, 1) if eta is not None else None,
            },
            "libraries": library_snapshots,
        }
        write_json_atomic(progress_path, snapshot)

        now = time.monotonic()
        if force_log or now - last_progress_log >= 15:
            eta_text = format_duration(eta) if eta is not None else "warming up"
            percent_text = f"{percent:.1f}%" if percent is not None else "pending"
            overall_bar = progress_bar(percent)
            details = []
            for label, info in library_snapshots.items():
                lib_pct = f"{info['percent']:.1f}%" if info["percent"] is not None else "pending"
                if info.get("merge_percent") is not None:
                    lib_pct = f"merge {info['merge_percent']:.1f}% {progress_bar(info['merge_percent'])}"
                else:
                    lib_pct = f"{lib_pct} {progress_bar(info['percent'])}"
                details.append(
                    f"{label} {lib_pct} (ETA {format_duration(info['estimated_remaining_seconds'])}, "
                    f"{info['chunks_done']}/{info['chunks_total']} chunks, "
                    f"{info['precursor_rows_read']:,} precursors)"
                )
            logger.info(
                "PROGRESS %s %s | step %d/%d: %s | elapsed %s | ETA %s | %s",
                percent_text,
                overall_bar,
                progress_step_index + 1,
                len(progress_steps),
                progress_steps[progress_step_index]["name"],
                format_duration(elapsed),
                eta_text,
                " | ".join(details),
            )
            last_progress_log = now

    def set_progress_step(index: int, state: str, *, force_log: bool = True) -> None:
        nonlocal progress_step_index, extraction_started
        progress_step_index = index
        progress_steps[index]["state"] = state
        if state == "running":
            progress_steps[index]["started_utc"] = datetime.now(timezone.utc).isoformat()
        elif state in {"completed", "skipped"}:
            progress_steps[index]["completed_utc"] = datetime.now(timezone.utc).isoformat()
        if index == 2 and state == "running" and extraction_started is None:
            extraction_started = time.monotonic()
        update_progress(force_log=force_log)

    set_progress_step(0, "running")
    try:
        manifest_path = out_dir / "manifest.json"
        manifest = load_manifest(manifest_path)
        manifest.setdefault("created_utc", datetime.now(timezone.utc).isoformat())
        manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["last_run_settings"] = {
            "cpu_affinity_limit": args.cpus,
            "parallel_library_jobs": args.jobs,
            "chunks_per_library": args.chunks_per_library,
            "batch_rows": args.batch_rows,
            "max_ram_mb_per_worker_soft_guard": args.max_ram_mb,
        }
        manifest["dataset_description"] = (
            "RT-only DIA-NN compact-library extracts; one row per modified peptide form and library. "
            "Includes UniMod:510 and UniMod:255 presence and N-terminal/lysine placement flags. "
            "No fragments, IM, precursor m/z, or charge-specific duplicate rows are stored."
        )
        manifest["output_schema"] = str(OUTPUT_SCHEMA)
        only_mod_ids = set(args.only_mod_id)
        jobs = []
        for label, input_path in args.library:
            existing = manifest["libraries"].get(label)
            output_path = data_dir / f"library={label}" / "rt.parquet"
            if existing and not args.replace_library:
                raise FormatError(
                    f"library label {label!r} already exists in {manifest_path}; "
                    "choose a new label or use --replace-library"
                )
            jobs.append((label, input_path, output_path))

        active_jobs = min(args.jobs, len(jobs))
        set_progress_step(0, "completed")
        set_progress_step(1, "running")
        if args.chunks_per_library == 1 and active_jobs == 1:
            set_progress_step(1, "completed")
            set_progress_step(2, "running")
            for label, input_path, output_path in jobs:
                logger.info("Starting %s: %s (%.2f GB)", label, input_path, input_path.stat().st_size / 1e9)
                worker_progress_path = worker_progress_dir / f"{label}.chunk0000.json"
                progress_worker_paths[label] = [worker_progress_path]
                progress_libraries[label].update(
                    {"state": "extracting", "chunks_total": 1, "bytes_total": input_path.stat().st_size}
                )
                update_progress(force_log=True)
                result = parse_library(
                    label,
                    input_path,
                    output_path,
                    batch_rows=args.batch_rows,
                    max_peptidoforms=args.max_peptidoforms,
                    only_mod_ids=only_mod_ids,
                    max_ram_mb=args.max_ram_mb,
                    logger=logger,
                    first_record_override=first_record_offsets.get(label),
                    progress_path=worker_progress_path,
                )
                manifest["libraries"][label] = result
                progress_libraries[label]["state"] = "completed"
                write_json_atomic(manifest_path, manifest)
                update_progress(force_log=True)
            set_progress_step(2, "completed")
        elif args.chunks_per_library == 1:
            set_progress_step(1, "completed")
            set_progress_step(2, "running")
            worker_count = min(active_jobs, len(affinity)) if affinity else active_jobs
            cpu_groups = [
                [cpu for index, cpu in enumerate(affinity) if index % worker_count == worker_index]
                for worker_index in range(worker_count)
            ] if affinity else [None] * worker_count
            logger.info("Running %d library extractions concurrently", active_jobs)
            futures = {}
            with ProcessPoolExecutor(max_workers=worker_count) as pool:
                for job_index, (label, input_path, output_path) in enumerate(jobs):
                    logger.info("Queueing %s: %s (%.2f GB)", label, input_path, input_path.stat().st_size / 1e9)
                    worker_log = log_dir / f"extract_{timestamp}_{label}.log"
                    worker_progress_path = worker_progress_dir / f"{label}.chunk0000.json"
                    progress_worker_paths[label] = [worker_progress_path]
                    progress_libraries[label].update(
                        {"state": "extracting", "chunks_total": 1, "bytes_total": input_path.stat().st_size}
                    )
                    future = pool.submit(
                        parse_library_worker,
                        label,
                        0,
                        input_path,
                        output_path,
                        first_record_offsets.get(label),
                        input_path.stat().st_size,
                        batch_rows=args.batch_rows,
                        max_peptidoforms=args.max_peptidoforms,
                        only_mod_ids=only_mod_ids,
                        max_ram_mb=args.max_ram_mb,
                        cpu_ids=cpu_groups[job_index % worker_count],
                        cpu_count=args.cpus,
                        log_path=worker_log,
                        log_level=args.log_level,
                        progress_path=worker_progress_path,
                    )
                    futures[future] = (label, job_index % worker_count)

                failures = 0
                pending = set(futures)
                update_progress(force_log=True)
                while pending:
                    completed, pending = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                    update_progress()
                    for future in completed:
                        label, _worker_index = futures[future]
                        try:
                            _, _, result = future.result()
                        except Exception as exc:
                            failures += 1
                            progress_libraries[label]["state"] = "failed"
                            logger.exception("Worker failed for %s: %s", label, exc)
                            continue
                        manifest["libraries"][label] = result
                        progress_libraries[label]["state"] = "completed"
                        write_json_atomic(manifest_path, manifest)
                        update_progress(force_log=True)
            if failures:
                return 2
            set_progress_step(2, "completed")
        else:
            available_cpus = len(affinity) if affinity else args.cpus
            worker_count = min(args.cpus, available_cpus, max(args.chunks_per_library * active_jobs, 1))
            effective_affinity = affinity[:worker_count] if affinity else None
            cpu_groups = (
                [
                    [cpu for index, cpu in enumerate(effective_affinity) if index % worker_count == worker_index]
                    for worker_index in range(worker_count)
                ]
                if effective_affinity
                else [None] * worker_count
            )
            chunk_root = out_dir / ".rt_chunks" / timestamp
            chunk_root.mkdir(parents=True, exist_ok=True)
            failed = False

            for batch_start in range(0, len(jobs), active_jobs):
                library_batch = jobs[batch_start : batch_start + active_jobs]
                tasks: list[tuple[str, Path, Path, int, int, int, Path]] = []
                per_library: dict[str, dict] = {}
                for label, input_path, output_path in library_batch:
                    library_started = time.monotonic()
                    with input_path.open("rb") as fh:
                        _metadata, protein_end = read_header_without_protein_rows(fh)
                        bounds = discover_chunk_bounds(
                            fh,
                            protein_end=protein_end,
                            file_size=input_path.stat().st_size,
                            chunk_count=args.chunks_per_library,
                            max_peptidoforms=args.max_peptidoforms,
                            first_record_override=first_record_offsets.get(label),
                        )
                    logger.info(
                        "%s: found %d record-aligned chunks; boundaries=%s",
                        label,
                        len(bounds),
                        [start for start, _end in bounds] + [bounds[-1][1]],
                    )
                    chunk_paths = []
                    progress_paths = []
                    library_bytes = 0
                    for chunk_index, (chunk_start, chunk_end) in enumerate(bounds):
                        part_path = chunk_root / f"{label}.part-{chunk_index:04d}.parquet"
                        worker_progress_path = worker_progress_dir / f"{label}.chunk{chunk_index:04d}.json"
                        chunk_paths.append(part_path)
                        progress_paths.append(worker_progress_path)
                        library_bytes += chunk_end - chunk_start
                        tasks.append(
                            (label, input_path, part_path, chunk_start, chunk_end, chunk_index, worker_progress_path)
                        )
                    progress_worker_paths[label] = progress_paths
                    progress_libraries[label].update(
                        {"state": "extracting", "chunks_total": len(bounds), "bytes_total": library_bytes}
                    )
                    per_library[label] = {
                        "input_path": input_path,
                        "output_path": output_path,
                        "bounds": bounds,
                        "parts": chunk_paths,
                        "results": [None] * len(bounds),
                        "started": library_started,
                    }

                set_progress_step(1, "completed")
                set_progress_step(2, "running")
                logger.info(
                    "Running %d record chunks on %d worker processes",
                    len(tasks),
                    min(worker_count, len(tasks)),
                )
                futures = {}
                process_count = min(worker_count, len(tasks))
                with ProcessPoolExecutor(max_workers=process_count) as pool:
                    for task_index, (
                        label,
                        input_path,
                        part_path,
                        chunk_start,
                        chunk_end,
                        chunk_index,
                        worker_progress_path,
                    ) in enumerate(tasks):
                        worker_index = task_index % process_count
                        worker_log = log_dir / f"extract_{timestamp}_{label}_chunk{chunk_index:04d}.log"
                        future = pool.submit(
                            parse_library_worker,
                            label,
                            chunk_index,
                            input_path,
                            part_path,
                            chunk_start,
                            chunk_end,
                            batch_rows=args.batch_rows,
                            max_peptidoforms=None,
                            only_mod_ids=only_mod_ids,
                            max_ram_mb=args.max_ram_mb,
                            cpu_ids=cpu_groups[worker_index],
                            cpu_count=len(cpu_groups[worker_index]) if cpu_groups[worker_index] else 1,
                            log_path=worker_log,
                            log_level=args.log_level,
                            progress_path=worker_progress_path,
                        )
                        futures[future] = label

                    pending = set(futures)
                    update_progress(force_log=True)
                    while pending:
                        completed, pending = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                        update_progress()
                        for future in completed:
                            label = futures[future]
                            try:
                                _, chunk_index, result = future.result()
                            except Exception as exc:
                                failed = True
                                progress_libraries[label]["state"] = "failed"
                                logger.exception("Chunk worker failed for %s: %s", label, exc)
                                continue
                            per_library[label]["results"][chunk_index] = result
                            update_progress()

                if failed:
                    progress_steps[progress_step_index]["state"] = "failed"
                    update_progress(force_log=True)
                    logger.error("Chunk outputs retained under %s for diagnosis", chunk_root)
                    return 2

                set_progress_step(2, "completed")
                set_progress_step(3, "running")
                for label, _input_path, _output_path in library_batch:
                    state = per_library[label]
                    segment_results = state["results"]
                    if any(result is None for result in segment_results):
                        raise FormatError(f"one or more chunks were not produced for {label}")
                    progress_libraries[label]["state"] = "merging"
                    progress_libraries[label]["merge_percent"] = 0.0
                    update_progress(force_log=True)
                    last_merge_update = 0.0

                    def report_merge_progress(rows_written: int, total_rows: int) -> None:
                        nonlocal last_merge_update
                        progress_libraries[label]["merge_percent"] = (
                            100.0 * rows_written / total_rows if total_rows else 100.0
                        )
                        now = time.monotonic()
                        if now - last_merge_update >= 15 or rows_written >= total_rows:
                            update_progress(force_log=True)
                            last_merge_update = now

                    output_size = merge_parquet_parts(
                        state["parts"], state["output_path"], progress_callback=report_merge_progress
                    )
                    result = combine_segment_metadata(
                        label,
                        state["input_path"],
                        state["output_path"],
                        segment_results,
                        state["bounds"],
                        output_size_bytes=output_size,
                        elapsed_seconds=time.monotonic() - state["started"],
                        limited_run=args.max_peptidoforms is not None,
                    )
                    manifest["libraries"][label] = result
                    progress_libraries[label]["state"] = "completed"
                    progress_libraries[label]["merge_percent"] = 100.0
                    write_json_atomic(manifest_path, manifest)
                    for part in state["parts"]:
                        part.unlink(missing_ok=True)
                    logger.info(
                        "%s: merged %d chunks into %s (%d peptide forms)",
                        label,
                        len(state["parts"]),
                        state["output_path"],
                        result["peptidoforms_written"],
                    )
                    update_progress(force_log=True)
                set_progress_step(3, "completed")
            try:
                chunk_root.rmdir()
                chunk_root.parent.rmdir()
            except OSError:
                pass
        if progress_steps[3]["state"] == "pending":
            set_progress_step(3, "skipped")
        set_progress_step(4, "running")
        set_progress_step(4, "completed")
        logger.info("Finished. Manifest: %s", manifest_path)
    except (FormatError, OSError, pa.ArrowException, MemoryError, json.JSONDecodeError) as exc:
        logger.exception("Extraction failed: %s", exc)
        progress_steps[progress_step_index]["state"] = "failed"
        update_progress(force_log=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
