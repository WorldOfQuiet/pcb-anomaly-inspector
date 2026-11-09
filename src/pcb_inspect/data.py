from __future__ import annotations

import csv
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from .common import ValidationError, sha256

VISA_ARCHIVE_SHA256 = "2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362"


def prepare_data(archive: Path, destination: Path, manifest: Path) -> dict:
    if not archive.is_file():
        raise ValidationError(f"Archive does not exist: {archive}")
    if destination.exists():
        raise ValidationError(f"Dataset output already exists: {destination}; choose a new directory")
    if not manifest.is_file():
        raise ValidationError(f"Split manifest does not exist: {manifest}")
    print("stage: verifying_archive", file=sys.stderr)
    if sha256(archive) != VISA_ARCHIVE_SHA256:
        raise ValidationError("Official VisA archive checksum mismatch")
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = {row["source_path"]: row["sha256"] for row in rows}
    if len(expected) != len(rows) or not expected:
        raise ValidationError("Split manifest has empty or duplicate source paths")
    for name in expected:
        parts = PurePosixPath(name).parts
        if not parts or parts[0] != "pcb1" or ".." in parts or PurePosixPath(name).is_absolute():
            raise ValidationError("Unsafe source path in split manifest")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".visa-", dir=destination.parent) as temporary:
        output = Path(temporary) / "data"
        output.mkdir()
        extracted = set()
        with tarfile.open(archive, "r:*") as tar:
            for member in tar:
                name = str(PurePosixPath(member.name.removeprefix("./")))
                if name not in expected:
                    continue
                if not member.isfile() or member.issym() or member.islnk():
                    raise ValidationError(f"Unsafe archive member: {name}")
                target = output / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
                if sha256(target) != expected[name]:
                    raise ValidationError(f"Image checksum mismatch in archive: {name}")
                extracted.add(name)
        if extracted != set(expected):
            raise ValidationError(f"Archive is missing {len(set(expected) - extracted)} expected PCB1 images")
        output.rename(destination)
    return {"images": len(extracted), "dataset": str(destination), "archive_sha256": VISA_ARCHIVE_SHA256}
