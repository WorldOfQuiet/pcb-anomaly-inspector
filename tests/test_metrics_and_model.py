import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pcb_inspect.common import object_hash, read_json, sha256
from pcb_inspect.inputs import InputRow
from pcb_inspect.metrics import benchmark_metrics, normal_threshold
from pcb_inspect.model import SPEC, fingerprint, inspect_model
from pcb_inspect.training import validate_calibration

ROOT = Path(__file__).resolve().parents[1]


def test_metrics_denominators_and_failures():
    rows = [InputRow(i, str(i), Path(f"{i}.png"), f"{i}.png", label) for i, label in enumerate([0, 0, 1, 1, 1])]
    predictions = [
        {"sample_id": "0", "status": "ok", "label": 0, "decision": "normal", "score": 0.1, "inference_seconds": .01, "total_seconds": .02},
        {"sample_id": "1", "status": "ok", "label": 0, "decision": "suspect", "score": 0.8, "inference_seconds": .01, "total_seconds": .02},
        {"sample_id": "2", "status": "ok", "label": 1, "decision": "suspect", "score": 0.9, "inference_seconds": .01, "total_seconds": .02},
        {"sample_id": "3", "status": "error", "input_path": "3.png", "error": "cannot decode"},
    ]
    metrics = benchmark_metrics(rows, predictions)
    assert metrics["confusion_matrix"] == {"TP": 1, "FP": 1, "TN": 1, "FN": 0}
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["complete_case_accuracy"] == pytest.approx(2 / 3)
    assert metrics["all_input_accuracy"] == pytest.approx(2 / 5)
    assert metrics["metric_denominators"]["accuracy"] == 3
    assert metrics["metric_denominators"]["all_input_accuracy"] == 5
    assert metrics["precision"] == .5 and metrics["recall"] == 1
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["counts"]["failed"] == 1 and metrics["counts"]["unprocessed"] == 1
    assert metrics["status"] == "incomplete" and metrics["image_auroc"] == 1


def test_calibration_quantile_and_minimum_count():
    assert normal_threshold(list(range(200)), .05)["threshold"] == 190
    assert normal_threshold(list(range(200)), .05)["k"] == 191
    with pytest.raises(ValueError, match="Insufficient"):
        normal_threshold([.1] * 18, .05)
    assert normal_threshold([.1] * 19, .05)["threshold"] == .1


def test_calibration_rejects_fit_overlap(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"test input")
    digest = sha256(image)
    (tmp_path / "fit_inputs.json").write_text(json.dumps([{"sample_id": "original", "sha256": digest}]))
    args = SimpleNamespace(model=tmp_path, alpha=.5)
    row = InputRow(1, "renamed", image, "image.png", 0, "calibration")
    with pytest.raises(ValueError, match="overlap"):
        validate_calibration(args, [row], {})


def test_bundled_frozen_model_checksums_and_policy():
    info, threshold = inspect_model(ROOT / "models/pcb1-v1")
    assert info["weights_sha256"] == "95b5dd11f458d7f05cff5e75d570092b0961c493325ecc53868239cac2b00e1c"
    assert info["fit_n"] == 604 and info["memory_bank_shape"] == [6184, 384]
    assert threshold["threshold"] == 1.5928595066070557 and threshold["n"] == 200
    assert "device" not in SPEC and "threads" not in SPEC
    assert fingerprint(info["weights_sha256"], SPEC) == info["model_sha256"]


def test_preserved_split_counts_and_nonoverlap():
    import csv
    from collections import Counter
    rows = list(csv.DictReader((ROOT / "data/pcb1-splits.csv").open()))
    assert Counter(row["split"] for row in rows) == {"fit_normal": 604, "calibration_normal": 200, "dev_normal": 100, "dev_anomaly": 20, "final_test": 180}
    assert len({row["sample_id"] for row in rows}) == len(rows)
    assert len({row["sha256"] for row in rows}) == len(rows)
    assert all(row["label"] == "0" for row in rows if row["split"] in {"fit_normal", "calibration_normal"})
