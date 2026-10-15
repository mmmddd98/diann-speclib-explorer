#!/usr/bin/env python3
"""Decode DIA-NN's compact binary .speclib format without invoking DIA-NN.

Currently supports the compact-library versions observed in DIA-NN 2.5 (-10)
and 2.6 (-11). The output is split into precursor, fragment, and protein
Parquet tables. Some compact per-record fields are retained as raw metadata
until their semantics are established.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
from pathlib import Path
from typing import BinaryIO, Iterator

import pyarrow as pa
import pyarrow.parquet as pq


I32 = struct.Struct("<i")
U32 = struct.Struct("<I")
F32 = struct.Struct("<f")
PROTEIN_COUNT_HEADER_BYTES = 12
PRECURSOR_PREFIX_BYTES = 48
POST_ID_BYTES = 16
FRAGMENT_BYTES = 12
MAX_PRECURSOR_ID_BYTES = 4096
PRECURSOR_ID_RE = re.compile(
    rb"\^?(?:\(UniMod:\d+\))*[A-Z](?:[A-Z]|\(UniMod:\d+\))*[1-6]"
)
UNIMOD_RE = re.compile(r"\(UniMod:\d+\)")


class FormatError(ValueError):
    pass


def read_exact(fh: BinaryIO, n: int) -> bytes:
    value = fh.read(n)
    if len(value) != n:
        raise FormatError(f"unexpected end of file: wanted {n} bytes, got {len(value)}")
    return value


def read_u32(fh: BinaryIO) -> int:
    return U32.unpack(read_exact(fh, 4))[0]


def read_string(fh: BinaryIO, *, max_len: int = 1_000_000) -> str:
    n = read_u32(fh)
    if n > max_len:
        raise FormatError(f"implausible string length {n} at offset {fh.tell() - 4}")
    try:
        return read_exact(fh, n).decode("utf-8")
    except UnicodeDecodeError as e:
        raise FormatError(f"invalid UTF-8 string at offset {fh.tell() - n}") from e


def parse_header_and_proteins(fh: BinaryIO) -> tuple[dict, list[dict], int]:
    fh.seek(0)
    header = struct.unpack("<iiii", read_exact(fh, 16))
    version, h1, h2, h3 = header
    if version not in (-10, -11):
        if version == -3:
            raise FormatError(
                "This is a Skyline .skyline.speclib sidecar (format -3), not a "
                "standalone DIA-NN .speclib. Use its referenced Parquet library."
            )
        raise FormatError(f"unsupported compact .speclib version {version}; expected -10 or -11")

    # Compact v2 libraries store a 32-bit reserved word between the fixed
    # header and the length-prefixed FASTA path.
    reserved = read_u32(fh)
    fasta_path = read_string(fh)
    trailer = read_exact(fh, PROTEIN_COUNT_HEADER_BYTES)
    protein_count, trailer_word1, trailer_word2 = struct.unpack("<III", trailer)
    if protein_count > 10_000_000:
        raise FormatError(f"implausible protein count {protein_count}")

    proteins: list[dict] = []
    for i in range(protein_count):
        accession = read_string(fh, max_len=4096)
        name = read_string(fh, max_len=4096)
        gene = read_string(fh, max_len=4096)
        pmeta = struct.unpack("<IIII", read_exact(fh, 16))
        proteins.append(
            {
                "protein_index": i,
                "protein_id": accession,
                "protein_name": name,
                "gene_name": gene,
                "protein_meta_0": pmeta[0],
                "protein_meta_1": pmeta[1],
                "protein_meta_2": pmeta[2],
                "protein_meta_3": pmeta[3],
            }
        )

    metadata = {
        "format_version": version,
        "header_words": [h1, h2, h3],
        "reserved_u32": reserved,
        "fasta_path_in_header": fasta_path,
        "protein_count": protein_count,
        "header_trailer_words": [trailer_word1, trailer_word2],
        "protein_table_end_offset": fh.tell(),
    }
    return metadata, proteins, fh.tell()


def decode_precursor_id(raw: bytes) -> tuple[str, str, str, int, bool] | None:
    if not PRECURSOR_ID_RE.fullmatch(raw):
        return None
    try:
        precursor_id = raw.decode("ascii")
    except UnicodeDecodeError:
        return None
    charge = int(precursor_id[-1])
    decoy = precursor_id.startswith("^")
    modified = precursor_id[1:-1] if decoy else precursor_id[:-1]
    stripped = UNIMOD_RE.sub("", modified)
    return precursor_id, modified, stripped, charge, decoy


def plausible_record_at(fh: BinaryIO, id_prefix_offset: int, file_size: int) -> bool:
    """Check the length-prefixed ID and adjacent record structure."""
    if id_prefix_offset < PRECURSOR_PREFIX_BYTES:
        return False
    fh.seek(id_prefix_offset)
    n_raw = fh.read(4)
    if len(n_raw) != 4:
        return False
    n = U32.unpack(n_raw)[0]
    if not 6 <= n <= MAX_PRECURSOR_ID_BYTES:
        return False
    raw = fh.read(n)
    decoded = decode_precursor_id(raw)
    if decoded is None:
        return False

    # The 48-byte prefix stores charge and peptide length before the ID.
    fh.seek(id_prefix_offset - PRECURSOR_PREFIX_BYTES)
    pre = fh.read(PRECURSOR_PREFIX_BYTES)
    if len(pre) != PRECURSOR_PREFIX_BYTES:
        return False
    charge = U32.unpack_from(pre, 4)[0]
    peptide_length = U32.unpack_from(pre, 8)[0]
    mz, rt, im = struct.unpack_from("<fff", pre, 12)
    if charge != decoded[3] or not 1 <= charge <= 6:
        return False
    if not 1 <= peptide_length <= 100 or peptide_length != len(decoded[2]):
        return False
    if not math.isfinite(mz) or not 100 <= mz <= 5000:
        return False
    if not math.isfinite(rt) or not math.isfinite(im):
        return False

    after_id = id_prefix_offset + 4 + n
    fh.seek(after_id + POST_ID_BYTES)
    n_raw = fh.read(4)
    if len(n_raw) != 4:
        return False
    n_fragments = U32.unpack(n_raw)[0]
    if not 0 <= n_fragments <= 256:
        return False
    end = after_id + POST_ID_BYTES + 4 + n_fragments * FRAGMENT_BYTES
    if end > file_size:
        return False
    # A few leading fragment records are enough to distinguish this from a
    # coincidental peptide-like string elsewhere in an index section.
    fh.seek(after_id + POST_ID_BYTES + 4)
    for _ in range(min(n_fragments, 3)):
        block = fh.read(FRAGMENT_BYTES)
        if len(block) != FRAGMENT_BYTES:
            return False
        product_mz, intensity = struct.unpack_from("<ff", block)
        if not (math.isfinite(product_mz) and 20 <= product_mz <= 5000):
            return False
        if not (math.isfinite(intensity) and -0.001 <= intensity <= 1.1):
            return False
    return True


def find_first_record(fh: BinaryIO, start: int, file_size: int, *, scan_limit: int | None = None) -> int:
    """Find the first precursor record by scanning for length-prefixed IDs."""
    chunk_size = 8 * 1024 * 1024
    overlap = 512
    pos = start
    carry = b""
    end_limit = min(file_size, start + scan_limit) if scan_limit is not None else file_size
    while pos < end_limit:
        fh.seek(pos)
        chunk = fh.read(min(chunk_size, end_limit - pos))
        if not chunk:
            break
        data = carry + chunk
        data_base = pos - len(carry)
        for match in PRECURSOR_ID_RE.finditer(data):
            string_offset = data_base + match.start()
            prefix_offset = string_offset - 4
            if prefix_offset < start:
                continue
            if not plausible_record_at(fh, prefix_offset, file_size):
                continue
            fh.seek(prefix_offset)
            if read_u32(fh) != len(match.group()):
                continue
            return prefix_offset
        carry = data[-overlap:]
        pos += len(chunk)
    raise FormatError(
        "could not locate a precursor record after the protein table; "
        "the version may use a different layout or the file may be truncated"
    )


def make_schemas() -> tuple[pa.Schema, pa.Schema, pa.Schema]:
    precursor_schema = pa.schema(
        [
            ("precursor_id", pa.string()),
            ("decoy", pa.bool_()),
            ("modified_sequence", pa.string()),
            ("stripped_sequence", pa.string()),
            ("precursor_charge", pa.int8()),
            ("peptide_length", pa.int16()),
            ("precursor_mz", pa.float32()),
            ("rt", pa.float32()),
            ("im", pa.float32()),
            ("precursor_index", pa.uint32()),
            ("metric_0", pa.float32()),
            ("metric_1", pa.float32()),
            ("metric_2", pa.float32()),
            ("metric_3", pa.float32()),
            ("metric_4", pa.float32()),
            ("peptide_index", pa.uint32()),
            ("post_id_word_0", pa.uint32()),
            ("post_id_metric_0", pa.float32()),
            ("post_id_word_1", pa.uint32()),
            ("post_id_word_2", pa.uint32()),
            ("fragment_count", pa.uint16()),
        ]
    )
    fragment_schema = pa.schema(
        [
            ("precursor_id", pa.string()),
            ("fragment_index", pa.uint16()),
            ("product_mz", pa.float32()),
            ("relative_intensity", pa.float32()),
            ("fragment_charge", pa.uint8()),
            ("fragment_type_code", pa.uint8()),
            ("fragment_type", pa.string()),
            ("series_number", pa.uint8()),
            ("loss_code", pa.uint8()),
        ]
    )
    protein_schema = pa.schema(
        [
            ("protein_index", pa.uint32()),
            ("protein_id", pa.string()),
            ("protein_name", pa.string()),
            ("gene_name", pa.string()),
            ("protein_meta_0", pa.uint32()),
            ("protein_meta_1", pa.uint32()),
            ("protein_meta_2", pa.uint32()),
            ("protein_meta_3", pa.uint32()),
        ]
    )
    return precursor_schema, fragment_schema, protein_schema


def write_rows(writer: pq.ParquetWriter, rows: list[dict], schema: pa.Schema) -> None:
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=schema))
        rows.clear()


def parse_file(
    input_path: Path,
    output_dir: Path,
    *,
    max_precursors: int | None = None,
    scan_limit: int | None = None,
    batch_precursors: int = 10_000,
    batch_fragments: int = 200_000,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    precursor_schema, fragment_schema, protein_schema = make_schemas()
    precursor_out = output_dir / "precursors.parquet"
    fragment_out = output_dir / "fragments.parquet"
    protein_out = output_dir / "proteins.parquet"

    # Metadata is decoded once; the spectrum records are streamed from disk.
    with input_path.open("rb") as fh:
        metadata, proteins, protein_end = parse_header_and_proteins(fh)
        first_record = find_first_record(fh, protein_end, input_path.stat().st_size, scan_limit=scan_limit)
        metadata["first_precursor_record_offset"] = first_record
        metadata["input_file"] = str(input_path.resolve())
        metadata["input_size_bytes"] = input_path.stat().st_size

        precursor_rows: list[dict] = []
        fragment_rows: list[dict] = []
        precursor_count = 0
        fragment_count = 0
        fragment_type_counts: dict[int, int] = {}
        fragment_charge_counts: dict[int, int] = {}
        min_fragments: int | None = None
        max_fragments = 0
        record_prefix = first_record
        file_size = input_path.stat().st_size

        with pq.ParquetWriter(precursor_out, precursor_schema, compression="zstd") as pw, pq.ParquetWriter(
            fragment_out, fragment_schema, compression="zstd"
        ) as fw:
            while record_prefix + PRECURSOR_PREFIX_BYTES < file_size:
                fh.seek(record_prefix)
                n = read_u32(fh)
                if not 6 <= n <= MAX_PRECURSOR_ID_BYTES:
                    break
                raw_id = read_exact(fh, n)
                decoded = decode_precursor_id(raw_id)
                if decoded is None:
                    break
                precursor_id, modified_sequence, stripped_sequence, charge, decoy = decoded

                fh.seek(record_prefix - PRECURSOR_PREFIX_BYTES)
                prefix = read_exact(fh, PRECURSOR_PREFIX_BYTES)
                precursor_index, raw_charge, peptide_len = struct.unpack_from("<III", prefix, 0)
                precursor_mz, rt, im, metric0, metric1, metric2, metric3, metric4 = struct.unpack_from("<8f", prefix, 12)
                peptide_index = U32.unpack_from(prefix, 44)[0]
                if raw_charge != charge or peptide_len != len(stripped_sequence):
                    raise FormatError(
                        f"precursor prefix disagrees with ID at offset {record_prefix}: "
                        f"charge {raw_charge}/{charge}, length {peptide_len}/{len(stripped_sequence)}"
                    )

                after_id = record_prefix + 4 + n
                fh.seek(after_id)
                post_word0 = read_u32(fh)
                post_metric0 = F32.unpack(read_exact(fh, 4))[0]
                post_word1 = read_u32(fh)
                post_word2 = read_u32(fh)
                n_fragments = read_u32(fh)
                fragments_start = fh.tell()
                if n_fragments > 256 or fragments_start + n_fragments * FRAGMENT_BYTES > file_size:
                    raise FormatError(f"invalid fragment count {n_fragments} at {record_prefix}")

                precursor_rows.append(
                    {
                        "precursor_id": precursor_id,
                        "decoy": decoy,
                        "modified_sequence": modified_sequence,
                        "stripped_sequence": stripped_sequence,
                        "precursor_charge": charge,
                        "peptide_length": peptide_len,
                        "precursor_mz": precursor_mz,
                        "rt": rt,
                        "im": im,
                        "precursor_index": precursor_index,
                        "metric_0": metric0,
                        "metric_1": metric1,
                        "metric_2": metric2,
                        "metric_3": metric3,
                        "metric_4": metric4,
                        "peptide_index": peptide_index,
                        "post_id_word_0": post_word0,
                        "post_id_metric_0": post_metric0,
                        "post_id_word_1": post_word1,
                        "post_id_word_2": post_word2,
                        "fragment_count": n_fragments,
                    }
                )

                for frag_idx in range(n_fragments):
                    raw_frag = read_exact(fh, FRAGMENT_BYTES)
                    product_mz, rel_intensity = struct.unpack_from("<ff", raw_frag, 0)
                    frag_charge, type_code, series, loss = raw_frag[8:12]
                    fragment_type = {1: "b", 2: "y"}.get(type_code)
                    fragment_rows.append(
                        {
                            "precursor_id": precursor_id,
                            "fragment_index": frag_idx,
                            "product_mz": product_mz,
                            "relative_intensity": rel_intensity,
                            "fragment_charge": frag_charge,
                            "fragment_type_code": type_code,
                            "fragment_type": fragment_type,
                            "series_number": series,
                            "loss_code": loss,
                        }
                    )
                    fragment_type_counts[type_code] = fragment_type_counts.get(type_code, 0) + 1
                    fragment_charge_counts[frag_charge] = fragment_charge_counts.get(frag_charge, 0) + 1
                next_record_prefix = fh.tell() + PRECURSOR_PREFIX_BYTES

                precursor_count += 1
                fragment_count += n_fragments
                min_fragments = n_fragments if min_fragments is None else min(min_fragments, n_fragments)
                max_fragments = max(max_fragments, n_fragments)
                if len(precursor_rows) >= batch_precursors:
                    write_rows(pw, precursor_rows, precursor_schema)
                if len(fragment_rows) >= batch_fragments:
                    write_rows(fw, fragment_rows, fragment_schema)

                if max_precursors is not None and precursor_count >= max_precursors:
                    record_prefix = next_record_prefix
                    break
                record_prefix = next_record_prefix

            write_rows(pw, precursor_rows, precursor_schema)
            write_rows(fw, fragment_rows, fragment_schema)

        pq.write_table(pa.Table.from_pylist(proteins, schema=protein_schema), protein_out, compression="zstd")

    metadata.update(
        {
            "parsed_precursor_records": precursor_count,
            "parsed_fragment_records": fragment_count,
            "min_fragments_per_precursor": min_fragments,
            "max_fragments_per_precursor": max_fragments,
            "fragment_type_code_counts": fragment_type_counts,
            "fragment_charge_counts": fragment_charge_counts,
            "parsed_record_end_offset": record_prefix,
            "trailing_bytes_after_record_block": max(0, input_path.stat().st_size - record_prefix),
            "precursors_parquet": str(precursor_out.resolve()),
            "fragments_parquet": str(fragment_out.resolve()),
            "proteins_parquet": str(protein_out.resolve()),
            "limited_run": max_precursors is not None,
        }
    )
    metadata_path = output_dir / "parse_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="DIA-NN compact .speclib file")
    parser.add_argument("--out-dir", type=Path, help="output directory (default: <input>.parsed)")
    parser.add_argument("--max-precursors", type=int, help="parse only this many precursors (for inspection)")
    parser.add_argument("--scan-limit-mb", type=int, help="limit initial record search to this many MB")
    parser.add_argument("--batch-precursors", type=int, default=10_000)
    parser.add_argument("--batch-fragments", type=int, default=200_000)
    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.is_file():
        parser.error(f"input file does not exist: {input_path}")
    output_dir = args.out_dir.resolve() if args.out_dir else input_path.parent / f"{input_path.name}.parsed"
    scan_limit = None if args.scan_limit_mb is None else args.scan_limit_mb * 1024 * 1024
    try:
        result = parse_file(
            input_path,
            output_dir,
            max_precursors=args.max_precursors,
            scan_limit=scan_limit,
            batch_precursors=args.batch_precursors,
            batch_fragments=args.batch_fragments,
        )
    except (FormatError, OSError, pa.ArrowException) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
