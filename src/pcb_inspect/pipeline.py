from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

from .artifacts import save_artifacts
from .inputs import verified_hash
from .model import decode_image

CSV_COLUMNS = [
    "schema_version", "sample_id", "input_path", "input_sha256", "split", "label", "status", "error",
    "score", "threshold", "decision", "model_id", "model_sha256", "device", "threads",
    "inference_seconds", "total_seconds", "original_path", "heatmap_path", "overlay_path", "raw_map_path",
]


def predict_rows(inspector, rows, output: Path, artifacts_dir: Path | None = None,
                 overlay_alpha: float = 0.45, fail_fast: bool = False, collect: bool = False):
    """Stream CSV rows. Keep scalar predictions only when computing metrics."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if artifacts_dir is not None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
    predictions = []
    counts = {"inputs": len(rows), "attempted": 0, "processed": 0, "failed": 0,
              "normal": 0, "suspect": 0, "needs_review": 0}
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            start = time.perf_counter()
            prediction = {"schema_version": 1, "sample_id": row.sample_id, "input_path": row.input_path,
                          "split": row.split, "label": row.label, "threshold": inspector.threshold,
                          "model_id": inspector.info["model_id"], "model_sha256": inspector.info["model_sha256"],
                          "device": str(inspector.device), "threads": inspector.threads}
            counts["attempted"] += 1
            try:
                digest = verified_hash(row)
                image = decode_image(row.path)
                score, raw_map, inference_seconds = inspector.forward(image)
                artifact_paths = save_artifacts(image, raw_map, artifacts_dir / row.artifact_name(),
                                               output.parent, overlay_alpha) if artifacts_dir is not None else {}
                prediction.update(status="ok", error="", score=score, decision=inspector.decision(score),
                                  input_sha256=digest, inference_seconds=inference_seconds,
                                  total_seconds=time.perf_counter() - start, **artifact_paths)
                counts["processed"] += 1
                counts[prediction["decision"]] += 1
            except Exception as exc:
                prediction.update(status="error", error=f"{type(exc).__name__}: {exc}", decision="error",
                                  total_seconds=time.perf_counter() - start)
                counts["failed"] += 1
                print(f"image_error: {row.sample_id}: {prediction['error']}", file=sys.stderr)
            writer.writerow(prediction)
            stream.flush()
            if collect:
                predictions.append(prediction)
            if counts["attempted"] % 25 == 0 or counts["attempted"] == len(rows):
                print(f"progress: {counts['attempted']}/{len(rows)}", file=sys.stderr)
            if prediction["status"] == "error" and fail_fast:
                break
    counts["unprocessed"] = counts["inputs"] - counts["attempted"]
    return counts, predictions
