from __future__ import annotations

import math
import shutil
import sys
import tempfile
import time
from pathlib import Path

from .common import ValidationError, environment, now, object_hash, read_json, save_json, sha256
from .inputs import normal_hashes
from .metrics import normal_threshold
from .model import BACKBONE_SHA256, SPEC, Inspector, configure, decode_image, fingerprint, make_model, preprocess, synchronize
from .pipeline import predict_rows


def _registry(rows, hashes):
    return [{"sample_id": row.sample_id, "sha256": digest} for row, digest in zip(rows, hashes, strict=True)]


def fit(args, rows) -> dict:
    import torch
    hashes = normal_hashes(rows)
    if sha256(args.backbone_weights) != BACKBONE_SHA256:
        raise ValidationError("Backbone checksum mismatch; expected official ResNet18 IMAGENET1K_V1 weights")
    device = configure(args.device, args.threads, args.seed)
    model = make_model()
    state = torch.load(args.backbone_weights, map_location="cpu", weights_only=True)
    loaded = model.feature_extractor.feature_extractor.load_state_dict(state, strict=False)
    if loaded.missing_keys or any(not key.startswith(("layer4.", "fc.")) for key in loaded.unexpected_keys):
        raise ValueError(f"Backbone weights do not match: {loaded}")
    model = model.to(device).train()
    start = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(rows), args.batch_size):
            batch = [preprocess(decode_image(row.path)) for row in rows[offset:offset + args.batch_size]]
            model(torch.stack(batch).to(device))
            print(f"features: {offset + len(batch)}/{len(rows)}", file=sys.stderr)
        print("stage: selecting_coreset", file=sys.stderr)
        model.subsample_embedding(args.coreset_ratio)
    synchronize(device)
    if model.memory_bank.shape[0] < SPEC["num_neighbors"]:
        raise ValueError("Coreset is too small for 9 neighbors; add normal images or increase --coreset-ratio")
    elapsed = time.perf_counter() - start
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fit-", dir=args.output_model.parent) as temporary:
        output = Path(temporary) / "model"
        output.mkdir()
        torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, output / "model.pt")
        save_json(output / "fit_inputs.json", _registry(rows, hashes))
        digest = sha256(output / "model.pt")
        info = {"schema_version": 1, "model_id": args.output_model.name, "frozen": False, "spec": SPEC,
                "weights_sha256": digest, "model_sha256": fingerprint(digest, SPEC),
                "fit_inputs_sha256": sha256(output / "fit_inputs.json"), "fit_n": len(rows),
                "fit_manifest_sha256": sha256(args.input), "fit_split": args.split,
                "backbone_weights_sha256": BACKBONE_SHA256, "memory_bank_shape": list(model.memory_bank.shape),
                "training": {"seed": args.seed, "coreset_sampling_ratio": args.coreset_ratio,
                             "batch_size": args.batch_size, "device": str(device), "seconds": elapsed},
                "created_at": now(), "environment": environment(str(device), args.threads),
                "result_origin": "real_inference"}
        save_json(output / "model.json", info)
        output.rename(args.output_model)
    return {"model": str(args.output_model), "fit_samples": len(rows), "memory_bank_rows": model.memory_bank.shape[0],
            "fit_seconds": elapsed, "status": "uncalibrated"}


def validate_calibration(args, rows, info) -> list[str]:
    hashes = normal_hashes(rows)
    fit_hashes = {item["sha256"] for item in read_json(args.model / "fit_inputs.json")}
    if fit_hashes.intersection(hashes):
        raise ValidationError("Calibration inputs overlap fitted images by SHA-256; use a separate normal split")
    k = math.ceil((len(rows) + 1) * (1 - args.alpha))
    if k > len(rows):
        raise ValidationError(f"Insufficient calibration samples: n={len(rows)}, k={k}; increase n or --alpha")
    return hashes


def calibrate(args, rows, hashes) -> dict:
    inspector = Inspector(args.model, args.device, args.threads, require_threshold=False)
    inspector.threshold = None
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".calibrate-", dir=args.output_model.parent) as temporary:
        output = Path(temporary) / "model"
        output.mkdir()
        counts, predictions = predict_rows(inspector, rows, output / "calibration_predictions.csv",
                                          fail_fast=True, collect=True)
        if counts["failed"]:
            raise ValueError("Calibration failed on an image; no model was written")
        policy = normal_threshold([prediction["score"] for prediction in predictions], args.alpha)
        policy.update(model_sha256=inspector.info["model_sha256"], weights_sha256=inspector.info["weights_sha256"],
                      calibration_inputs_sha256=object_hash(hashes), calibration_split=args.split,
                      calibration_manifest_sha256=sha256(args.input), device=str(inspector.device),
                      environment=environment(str(inspector.device), args.threads), created_at=now())
        save_json(output / "threshold.json", policy)
        save_json(output / "calibration_inputs.json", _registry(rows, hashes))
        for name in ("model.pt", "fit_inputs.json"):
            shutil.copyfile(args.model / name, output / name)
        info = dict(inspector.info, model_id=args.output_model.name, frozen=False,
                    threshold_sha256=sha256(output / "threshold.json"),
                    calibration_inputs_sha256=sha256(output / "calibration_inputs.json"))
        save_json(output / "model.json", info)
        output.rename(args.output_model)
    return {"model": str(args.output_model), "calibration_samples": len(rows), "threshold": policy["threshold"],
            "alpha": args.alpha, "status": "calibrated"}
