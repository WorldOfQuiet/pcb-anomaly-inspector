from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path


class ValidationError(ValueError):
    """Invalid user input, detected before model execution."""


def sha256(filename: Path) -> str:
    digest = hashlib.sha256()
    with filename.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def read_json(filename: Path):
    return json.loads(filename.read_text(encoding="utf-8"))


def save_json(filename: Path, value) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".writing-", dir=filename.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, filename)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def check_output(filename: Path, overwrite: bool = False, directory: bool = False) -> None:
    if filename.exists():
        if filename.is_dir() != directory:
            raise ValidationError(f"Output has the wrong type: {filename}")
        if not overwrite:
            raise ValidationError(f"Output already exists: {filename}; choose a new path or use --overwrite")
    parent = filename.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    if not parent.is_dir():
        raise ValidationError(f"Output parent is not a directory: {parent}")


def environment(device: str, threads: int) -> dict:
    import psutil
    try:
        import resource
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024)
    except ImportError:
        peak_rss = None
    packages = {}
    for package in ("pcb-inspect", "torch", "torchvision", "anomalib", "numpy", "pillow",
                    "scikit-learn", "timm", "opencv-python-headless"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.partition(":")[2].strip()
                break
    info = {"recorded_at": now(), "python": platform.python_version(), "platform": platform.platform(),
            "cpu_model": cpu, "logical_cpus": os.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
            "cpu_affinity": list(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "ram_total_bytes": psutil.virtual_memory().total, "device": device, "torch_threads": threads,
            "precision": "float32", "packages": packages,
            "peak_process_rss_bytes": peak_rss}
    if device.startswith("cuda"):
        import torch
        info["gpu_model"] = torch.cuda.get_device_name(device)
    return info
