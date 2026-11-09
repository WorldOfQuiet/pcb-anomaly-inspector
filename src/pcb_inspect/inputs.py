from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .common import ValidationError, sha256

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MANIFEST_SUFFIXES = {".csv", ".jsonl"}


@dataclass(frozen=True)
class InputRow:
    index: int
    sample_id: str
    path: Path
    input_path: str
    label: int | None = None
    split: str = ""
    expected_sha256: str = ""

    def artifact_name(self) -> str:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.sample_id).strip("._-")[:48] or "sample"
        digest = hashlib.sha256((self.sample_id + "\0" + self.input_path).encode()).hexdigest()[:12]
        return f"{self.index:06d}-{slug}-{digest}"


def _records(filename: Path):
    with filename.open(encoding="utf-8-sig", newline="") as stream:
        if filename.suffix.lower() == ".csv":
            reader = csv.DictReader(stream)
            if not reader.fieldnames or "path" not in reader.fieldnames:
                raise ValidationError("CSV manifest must have a path column")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValidationError("CSV manifest contains duplicate column names")
            for number, row in enumerate(reader, 2):
                if None in row:
                    raise ValidationError(f"Extra CSV fields at line {number}")
                yield number, row
        else:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValidationError(f"Invalid JSON at line {number}: {exc.msg}") from exc
                if not isinstance(row, dict):
                    raise ValidationError(f"JSONL line {number} must be an object")
                yield number, row


def _label(value, number: int) -> int | None:
    if value is None or value == "":
        return None
    if str(value) not in {"0", "1"}:
        raise ValidationError(f"Manifest label at line {number} must be 0, 1, or empty")
    return int(value)


def load_inputs(source: Path, split: str | None = None, require_labels: bool = False,
                normal_only: bool = False) -> list[InputRow]:
    """Read metadata once; inference holds only one decoded image at a time."""
    if not source.exists():
        raise ValidationError(f"Input does not exist: {source}")
    rows: list[InputRow] = []
    manifest = source.is_file() and source.suffix.lower() in MANIFEST_SUFFIXES
    if (split is not None or require_labels or normal_only) and not manifest:
        raise ValidationError("This operation requires a CSV or JSONL manifest")
    if manifest:
        available = set()
        for number, item in _records(source):
            row_split = item.get("split") or ""
            if not isinstance(row_split, str):
                raise ValidationError(f"Manifest split at line {number} must be text")
            available.add(row_split)
            if split is not None and row_split != split:
                continue
            image_path = item.get("path")
            if not isinstance(image_path, str) or not image_path.strip():
                raise ValidationError(f"Missing path at manifest line {number}")
            label = _label(item.get("label"), number)
            if require_labels and label is None:
                raise ValidationError(f"Missing label at manifest line {number}; labeled inputs are required")
            if normal_only and label != 0:
                raise ValidationError(f"Normal-only input required; line {number} has label {label}")
            sample_id = item.get("sample_id") or hashlib.sha256(image_path.encode()).hexdigest()[:24]
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValidationError(f"Invalid sample_id at manifest line {number}")
            expected = item.get("sha256") or ""
            if not isinstance(expected, str) or (expected and not re.fullmatch(r"[a-fA-F0-9]{64}", expected)):
                raise ValidationError(f"Invalid SHA-256 at manifest line {number}")
            resolved = (source.parent / image_path).resolve()
            rows.append(InputRow(len(rows) + 1, sample_id, resolved, image_path, label, row_split, expected.lower()))
        if split is not None and not rows:
            raise ValidationError(f"No inputs for split '{split}'; available splits: {', '.join(sorted(available)) or '(none)'}")
    elif source.is_dir():
        for folder, directories, filenames in os.walk(source, followlinks=False):
            directories.sort()
            for name in sorted(filenames):
                filename = Path(folder) / name
                if filename.suffix.lower() in IMAGE_SUFFIXES:
                    input_path = str(filename)
                    sample_id = hashlib.sha256(str(filename.relative_to(source)).encode()).hexdigest()[:24]
                    rows.append(InputRow(len(rows) + 1, sample_id, filename.resolve(), input_path))
    elif source.suffix.lower() in IMAGE_SUFFIXES:
        rows.append(InputRow(1, source.stem, source.resolve(), str(source)))
    else:
        raise ValidationError("Input must be a JPG/PNG image, directory, CSV, or JSONL manifest")
    if not rows:
        raise ValidationError("Input selection is empty")
    ids, paths = set(), set()
    for row in rows:
        if row.sample_id in ids:
            raise ValidationError(f"Duplicate sample_id: {row.sample_id}")
        if row.path in paths:
            raise ValidationError(f"Duplicate image path: {row.input_path}")
        ids.add(row.sample_id)
        paths.add(row.path)
        if not row.path.is_file():
            raise ValidationError(f"Image does not exist or is not a file: {row.input_path}")
        if row.path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValidationError(f"Unsupported image extension: {row.input_path}")
    return rows


def verified_hash(row: InputRow) -> str:
    digest = sha256(row.path)
    if row.expected_sha256 and digest != row.expected_sha256:
        raise ValueError(f"Image checksum mismatch: {row.input_path}")
    return digest


def normal_hashes(rows: list[InputRow]) -> list[str]:
    hashes = [verified_hash(row) for row in rows]
    if len(hashes) != len(set(hashes)):
        raise ValidationError("Duplicate image contents in normal-only inputs")
    return hashes
