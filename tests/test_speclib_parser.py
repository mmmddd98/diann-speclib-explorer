from __future__ import annotations

import struct

import pyarrow.parquet as pq
import pytest

from diann_speclib_explorer.parse_diann_speclib import FormatError, decode_precursor_id, parse_file


def _write_u32(handle, value: int) -> None:
    handle.write(struct.pack("<I", value))


def _write_string(handle, value: str) -> None:
    raw = value.encode("utf-8")
    _write_u32(handle, len(raw))
    handle.write(raw)


def make_tiny_library(path) -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("<iiii", -10, 0, 0, 0))
        _write_u32(handle, 0)  # reserved header word
        _write_string(handle, "test.fasta")
        handle.write(struct.pack("<III", 1, 0, 0))
        _write_string(handle, "P00001")
        _write_string(handle, "Example protein")
        _write_string(handle, "EXAMPLE")
        handle.write(struct.pack("<IIII", 0, 0, 0, 0))

        # Record prefix is 48 bytes immediately before the length-prefixed ID.
        handle.write(struct.pack("<III8fI", 0, 2, 7, 500.0, 12.5, 0.8, 0, 0, 0, 0, 0, 0))
        _write_string(handle, "PEPTIDE2")
        handle.write(struct.pack("<IfIII", 0, 0.0, 0, 0, 1))
        handle.write(struct.pack("<ffBBBB", 200.0, 0.75, 1, 1, 3, 0))
        # The observed compact record block ends with a 48-byte sentinel/prefix.
        handle.write(b"\0" * 48)


def test_decode_precursor_id_preserves_sequence_charge_and_decoy():
    assert decode_precursor_id(b"PEPTIDE2") == ("PEPTIDE2", "PEPTIDE", "PEPTIDE", 2, False)
    assert decode_precursor_id(b"^(UniMod:510)PEPTIDE3") == (
        "^(UniMod:510)PEPTIDE3",
        "(UniMod:510)PEPTIDE",
        "PEPTIDE",
        3,
        True,
    )
    assert decode_precursor_id(b"not-a-precursor") is None


def test_parse_tiny_compact_library_to_parquet(tmp_path):
    source = tmp_path / "tiny.predicted.speclib"
    output = tmp_path / "parsed"
    make_tiny_library(source)

    metadata = parse_file(source, output)

    assert metadata["format_version"] == -10
    assert metadata["parsed_precursor_records"] == 1
    assert metadata["parsed_fragment_records"] == 1
    precursor = pq.read_table(output / "precursors.parquet").to_pylist()
    fragment = pq.read_table(output / "fragments.parquet").to_pylist()
    proteins = pq.read_table(output / "proteins.parquet").to_pylist()
    assert precursor[0]["modified_sequence"] == "PEPTIDE"
    assert precursor[0]["precursor_charge"] == 2
    assert precursor[0]["rt"] == pytest.approx(12.5)
    assert fragment[0]["fragment_type"] == "b"
    assert fragment[0]["relative_intensity"] == pytest.approx(0.75)
    assert proteins[0]["protein_id"] == "P00001"


def test_parser_rejects_skyline_sidecar(tmp_path):
    source = tmp_path / "sample.skyline.speclib"
    source.write_bytes(struct.pack("<iiii", -3, 0, 0, 0))
    with pytest.raises(FormatError, match="Skyline"):
        parse_file(source, tmp_path / "out")
