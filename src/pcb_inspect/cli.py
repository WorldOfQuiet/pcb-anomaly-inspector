from __future__ import annotations

import argparse
import contextlib
import math
import re
import sys
import time
from pathlib import Path

from . import __version__
from .common import ValidationError, check_output, save_json, sha256
from .inputs import load_inputs


def positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def probability(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number between 0 and 1") from exc
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be finite and between 0 and 1")
    return number


def open_probability(value: str) -> float:
    number = probability(value)
    if not 0 < number < 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1, exclusive")
    return number


def device_name(value: str) -> str:
    if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", value):
        raise argparse.ArgumentTypeError("must be cpu, cuda, or cuda:N")
    return value


def split_name(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("split name must not be empty")
    return value


def _runtime(parser):
    parser.add_argument("--device", type=device_name, default="cpu", help="Execution device: cpu, cuda, or cuda:N (default: cpu).")
    parser.add_argument("--threads", type=positive_integer, default=4, help="PyTorch CPU threads (default: 4); does not change model identity.")


def _model(parser):
    parser.add_argument("--model", type=Path, default=Path("models/pcb1-v1"),
                        help="Model directory containing model.pt, model.json and threshold.json (default: models/pcb1-v1).")


def _prediction(parser):
    parser.add_argument("--input", required=True, type=Path, help="JPEG/PNG file, recursively read directory, or CSV/JSONL manifest.")
    parser.add_argument("--output", required=True, type=Path, help="Prediction CSV destination; parent directories are created.")
    _model(parser)
    _runtime(parser)
    parser.add_argument("--artifacts-dir", type=Path, help="Optional directory for per-image original, heatmap, overlay, and raw .npy map.")
    parser.add_argument("--overlay-alpha", type=probability, default=0.45, metavar="0..1",
                        help="Heatmap opacity: 0 is original, 1 is heatmap (default: 0.45); scores are unchanged.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing CSV/metrics and matching artifact files.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first image failure (exit 3); otherwise record errors and continue.")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="pcb-inspect", description="Inspect PCB images with a frozen PatchCore model.",
                                   epilog="Summaries use metric: value on stdout. Progress and errors go to stderr. No command downloads model weights.")
    root.add_argument("--version", action="version", version=f"pcb-inspect {__version__}")
    sub = root.add_subparsers(dest="command", required=True)
    predict = sub.add_parser("predict", help="Inspect images and write a CSV.",
                             description="Inspect a file, directory, or manifest. No ground-truth labels are required.")
    _prediction(predict)
    predict.add_argument("--split", type=split_name, help="Optional exact manifest split to select; omitted means all rows.")
    benchmark = sub.add_parser("benchmark", help="Evaluate a labeled manifest with a fixed model and threshold.",
                               description="Measure image-level quality and latency. Requires 0/1 labels for every selected row.")
    _prediction(benchmark)
    benchmark.add_argument("--split", required=True, type=split_name, help="Exact manifest split name to evaluate; never inferred from filenames.")
    benchmark.add_argument("--metrics", required=True, type=Path, help="JSON destination for counts, quality, confusion matrix, latency, and environment.")
    fit = sub.add_parser("fit", help="Fit a new normal-only memory bank.",
                         description="Fit PatchCore on normal images using local ImageNet ResNet18 weights. Existing models are never overwritten.")
    fit.add_argument("--input", required=True, type=Path, help="CSV/JSONL manifest with path, label=0, and split columns.")
    fit.add_argument("--split", default="fit_normal", type=split_name, help="Exact normal-only fit split (default: fit_normal).")
    fit.add_argument("--output-model", required=True, type=Path, help="New directory for the uncalibrated model; must not exist.")
    fit.add_argument("--backbone-weights", required=True, type=Path, help="Local resnet18-f37072fd.pth; verified against the official full SHA-256.")
    fit.add_argument("--batch-size", type=positive_integer, default=8, help="Images per feature-extraction batch (default: 8).")
    fit.add_argument("--coreset-ratio", type=open_probability, default=0.01, help="Fraction of normal patches retained (default: 0.01).")
    fit.add_argument("--seed", type=int, default=42, help="Training random seed, 0 through 4294967295 (default: 42).")
    _runtime(fit)
    calibrate = sub.add_parser("calibrate", help="Calibrate a new model copy on a separate normal-only split.",
                               description="Choose the fixed upper quantile on normal images; reject overlap with fit images by SHA-256.")
    calibrate.add_argument("--input", required=True, type=Path, help="CSV/JSONL manifest with path, label=0, and split columns.")
    calibrate.add_argument("--split", default="calibration_normal", type=split_name, help="Exact separate normal-only split (default: calibration_normal).")
    calibrate.add_argument("--output-model", required=True, type=Path, help="New model directory including a calibrated threshold; must not exist.")
    calibrate.add_argument("--alpha", type=open_probability, default=0.05, help="Target normal false-reject fraction (default: 0.05; at least 19 normal samples).")
    _model(calibrate)
    _runtime(calibrate)
    info = sub.add_parser("model-info", help="Verify model checksums and print its identity.",
                          description="Inspect portable model metadata and checksums without loading PyTorch.")
    _model(info)
    data = sub.add_parser("prepare-data", help="Verify and extract PCB1 images from a local official VisA archive.",
                          description="Extract only the checked-in split's images. Download the official archive explicitly first.")
    data.add_argument("--archive", required=True, type=Path, help="Local VisA_20220922.tar (official SHA-256 is verified).")
    data.add_argument("--output-dir", type=Path, default=Path("data/visa"), help="New extraction directory (default: data/visa); must not exist.")
    data.add_argument("--manifest", type=Path, default=Path("data/pcb1-splits.csv"), help="Frozen split manifest with source_path and sha256 (default: data/pcb1-splits.csv).")
    return root


def summary(values: dict) -> None:
    for key, value in values.items():
        if isinstance(value, float):
            value = f"{value:.9g}"
        print(f"{key}: {value if value is not None else 'null'}")


def _validate_destinations(args, rows):
    check_output(args.output, args.overwrite)
    destinations = [args.output]
    if args.command == "benchmark":
        check_output(args.metrics, args.overwrite)
        destinations.append(args.metrics)
    if len({destination.resolve() for destination in destinations}) != len(destinations):
        raise ValidationError("CSV and metrics must have different destinations")
    inputs = {row.path for row in rows} | {args.input.resolve()}
    model_root = args.model.resolve()
    for destination in destinations:
        resolved = destination.resolve()
        if resolved in inputs or resolved.is_relative_to(model_root):
            raise ValidationError("Output must not overwrite input or model files")
    if args.artifacts_dir is not None:
        check_output(args.artifacts_dir, args.overwrite, directory=True)
        artifacts = args.artifacts_dir.resolve()
        if any(artifacts.is_relative_to(destination.resolve()) for destination in destinations):
            raise ValidationError("Artifact directory must not be a CSV/metrics filename or descend from one")
        if artifacts.is_relative_to(model_root) or model_root.is_relative_to(artifacts):
            raise ValidationError("Artifacts directory must be separate from the model directory")
        if any(path.is_relative_to(artifacts) for path in inputs):
            raise ValidationError("Artifacts directory must not contain the input data")


def run(args) -> int:
    if args.command == "prepare-data":
        from .data import prepare_data
        summary(prepare_data(args.archive, args.output_dir, args.manifest))
        return 0
    if hasattr(args, "model") and not args.model.is_dir():
        raise ValidationError(f"Model directory does not exist: {args.model}")
    if args.command == "model-info":
        from .model import inspect_model
        info, policy = inspect_model(args.model, require_threshold=False)
        summary({"model_id": info["model_id"], "model_sha256": info["model_sha256"],
                 "weights_sha256": info["weights_sha256"], "fit_samples": info["fit_n"],
                 "memory_bank_shape": "x".join(map(str, info["memory_bank_shape"])),
                 "threshold": policy["threshold"] if policy else None,
                 "comparison": "greater_than", "checksums": "verified"})
        return 0
    training = args.command in {"fit", "calibrate"}
    rows = load_inputs(args.input, args.split, require_labels=training or args.command == "benchmark", normal_only=training)
    if training:
        if args.output_model.exists():
            raise ValidationError(f"Model output already exists: {args.output_model}; use a new directory")
        check_output(args.output_model, directory=True)
        if args.command == "calibrate" and args.output_model.resolve().is_relative_to(args.model.resolve()):
            raise ValidationError("Calibrated output must be outside the source model directory")
        if args.command == "fit":
            if not args.backbone_weights.is_file():
                raise ValidationError(f"Backbone weights do not exist: {args.backbone_weights}")
            if not 0 <= args.seed <= 2**32 - 1:
                raise ValidationError("Seed must be between 0 and 4294967295")
            from .training import fit
            with contextlib.redirect_stdout(sys.stderr):
                result = fit(args, rows)
        else:
            from .model import inspect_model
            from .training import calibrate, validate_calibration
            info, _ = inspect_model(args.model, require_threshold=False)
            hashes = validate_calibration(args, rows, info)
            with contextlib.redirect_stdout(sys.stderr):
                result = calibrate(args, rows, hashes)
        summary(result)
        return 0
    _validate_destinations(args, rows)
    from .model import Inspector
    from .pipeline import predict_rows
    started = time.perf_counter()
    with contextlib.redirect_stdout(sys.stderr):
        inspector = Inspector(args.model, args.device, args.threads)
        counts, predictions = predict_rows(inspector, rows, args.output, args.artifacts_dir, args.overlay_alpha,
                                          args.fail_fast, collect=args.command == "benchmark")
    values = {"inputs": counts["inputs"], "processed": counts["processed"], "failed": counts["failed"],
              "unprocessed": counts["unprocessed"], "normal": counts["normal"], "suspect": counts["suspect"],
              "model": inspector.info["model_id"], "device": str(inspector.device), "threads": args.threads,
              "output": str(args.output)}
    if args.command == "benchmark":
        from .common import environment, now
        from .metrics import benchmark_metrics
        with contextlib.redirect_stdout(sys.stderr):
            metrics = benchmark_metrics(rows, predictions)
        metrics.update(split=args.split, input_manifest_sha256=sha256(args.input),
                       model_id=inspector.info["model_id"], model_sha256=inspector.info["model_sha256"],
                       weights_sha256=inspector.info["weights_sha256"], threshold=inspector.threshold,
                       predictions_sha256=sha256(args.output), created_at=now(),
                       wall_seconds=time.perf_counter() - started,
                       environment=environment(str(inspector.device), args.threads))
        save_json(args.metrics, metrics)
        values.update({key: metrics[key] for key in ("accuracy", "complete_case_accuracy", "all_input_accuracy", "precision", "recall", "f1", "image_auroc", "image_average_precision")})
        values["inference_p50_seconds"] = metrics["inference_latency"]["p50"]
        values["inference_p95_seconds"] = metrics["inference_latency"]["p95"]
        values["metrics"] = str(args.metrics)
    summary(values)
    return 3 if counts["failed"] else 0


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        return run(args)
    except ValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
