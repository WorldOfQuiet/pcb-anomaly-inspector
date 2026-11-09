import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pcb_inspect import cli
from pcb_inspect.inputs import InputRow, load_inputs
from pcb_inspect.pipeline import CSV_COLUMNS


@pytest.fixture
def images(tmp_path):
    first = tmp_path / "a" / "board.png"
    second = tmp_path / "b" / "board.png"
    first.parent.mkdir()
    second.parent.mkdir()
    Image.new("RGB", (32, 24), (40, 70, 90)).save(first)
    Image.new("RGB", (32, 24), (100, 70, 90)).save(second)
    return first, second


@pytest.fixture
def fake_model(monkeypatch, tmp_path):
    class Inspector:
        def __init__(self, directory, device="cpu", threads=4, require_threshold=True):
            self.info = {"model_id": "test", "model_sha256": "0" * 64, "weights_sha256": "1" * 64}
            self.threshold = 1.0
            self.device = device
            self.threads = threads

        def forward(self, image):
            return 1.0, np.arange(256, dtype=float).reshape(16, 16), 0.01

        def decision(self, score):
            return "suspect" if score > self.threshold else "normal"

    monkeypatch.setattr("pcb_inspect.model.Inspector", Inspector)
    directory = tmp_path / "model"
    directory.mkdir()
    return directory


def write_manifest(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "path", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.mark.parametrize("command", ["predict", "benchmark", "fit", "calibrate", "model-info", "prepare-data"])
def test_help_is_offline_and_does_not_import_ml(command):
    script = "from pcb_inspect.cli import main; import sys\ntry: main([sys.argv[1], '--help'])\nexcept SystemExit as e: assert e.code == 0\nassert 'torch' not in sys.modules\nassert 'anomalib' not in sys.modules\n"
    result = subprocess.run([sys.executable, "-c", script, command], text=True, capture_output=True,
                            env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")})
    assert result.returncode == 0, result.stderr
    assert "usage: pcb-inspect" in result.stdout
    assert not result.stderr


@pytest.mark.parametrize("args", [
    ["--overlay-alpha", "-0.1"], ["--overlay-alpha", "1.1"], ["--overlay-alpha", "nan"],
    ["--overlay-alpha", "inf"], ["--threads", "0"], ["--device", "auto"],
    ["--split", ""],
])
def test_invalid_runtime_arguments(args):
    with pytest.raises(SystemExit) as error:
        cli.main(["predict", "--input", "missing.jpg", "--output", "result.csv", *args])
    assert error.value.code == 2


def test_relative_manifest_and_duplicate_ids(images, tmp_path):
    first, second = images
    manifest = write_manifest(tmp_path / "manifest.csv", [
        {"sample_id": "a", "path": "a/board.png", "label": 0, "split": "dev"},
        {"sample_id": "b", "path": "b/board.png", "label": 1, "split": "dev"},
    ])
    rows = load_inputs(manifest, "dev", require_labels=True)
    assert [row.path for row in rows] == [first, second]
    assert rows[0].input_path == "a/board.png"
    assert rows[0].artifact_name() != rows[1].artifact_name()
    write_manifest(manifest, [
        {"sample_id": "a", "path": "a/board.png", "label": 0, "split": "dev"},
        {"sample_id": "a", "path": "b/board.png", "label": 1, "split": "dev"},
    ])
    with pytest.raises(ValueError, match="Duplicate sample_id"):
        load_inputs(manifest)


def test_artifact_name_cannot_escape_directory(tmp_path):
    row = InputRow(1, "../../bad/..\\id", tmp_path / "x.jpg", "../../x.jpg")
    assert "/" not in row.artifact_name() and "\\" not in row.artifact_name()
    assert row.artifact_name().startswith("000001-")


def test_predict_directory_schema_stdout_no_default_artifacts(images, fake_model, tmp_path, capsys):
    output = tmp_path / "result.csv"
    result = cli.main(["predict", "--input", str(tmp_path), "--output", str(output), "--model", str(fake_model)])
    assert result == 0
    with output.open() as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == CSV_COLUMNS
        rows = list(reader)
    assert len(rows) == 2
    assert all(row["status"] == "ok" and row["decision"] == "normal" for row in rows)
    assert all(row["overlay_path"] == "" for row in rows)
    captured = capsys.readouterr()
    assert "processed: 2" in captured.out
    assert "progress: 2/2" in captured.err
    assert all(": " in line for line in captured.out.splitlines())
    assert not list(tmp_path.rglob("overlay.png"))


def test_output_overwrite_is_explicit(images, fake_model, tmp_path):
    output = tmp_path / "result.csv"
    output.write_text("preserve me")
    args = ["predict", "--input", str(images[0]), "--output", str(output), "--model", str(fake_model)]
    assert cli.main(args) == 2
    assert output.read_text() == "preserve me"
    assert cli.main([*args, "--overwrite"]) == 0


def test_artifact_destination_cannot_descend_from_output_file(images, fake_model, tmp_path):
    output = tmp_path / "result.csv"
    assert cli.main(["predict", "--input", str(images[0]), "--output", str(output), "--model", str(fake_model),
                     "--artifacts-dir", str(output / "maps")]) == 2
    assert not output.exists()


@pytest.mark.parametrize("fail_fast,expected_rows,unprocessed", [(False, 3, 0), (True, 2, 1)])
def test_image_failure_is_explicit_and_nonzero(images, fake_model, tmp_path, fail_fast, expected_rows, unprocessed, capsys):
    (tmp_path / "bad.png").write_bytes(b"not an image")
    manifest = write_manifest(tmp_path / "input.csv", [
        {"sample_id": "good1", "path": "a/board.png"}, {"sample_id": "bad", "path": "bad.png"},
        {"sample_id": "good2", "path": "b/board.png"},
    ])
    output = tmp_path / "result.csv"
    args = ["predict", "--input", str(manifest), "--output", str(output), "--model", str(fake_model)]
    assert cli.main([*args, *(["--fail-fast"] if fail_fast else [])]) == 3
    rows = list(csv.DictReader(output.open()))
    assert len(rows) == expected_rows
    assert rows[1]["status"] == "error" and rows[1]["decision"] == "error"
    assert rows[1]["score"] == ""
    captured = capsys.readouterr()
    assert f"unprocessed: {unprocessed}" in captured.out
    assert "image_error: bad:" in captured.err


@pytest.mark.parametrize("change,expected", [
    ({"label": ""}, "Missing label"), ({"path": "missing.png"}, "Image does not exist"),
    ({"label": "normal"}, "must be 0, 1"), ({"split": "other"}, "No inputs for split"),
])
def test_benchmark_validates_before_writing(images, fake_model, tmp_path, change, expected, capsys):
    manifest = write_manifest(tmp_path / "input.csv", [{"sample_id": "a", "path": "a/board.png", "label": 0, "split": "dev", **change}])
    output, metrics = tmp_path / "result.csv", tmp_path / "metrics.json"
    assert cli.main(["benchmark", "--input", str(manifest), "--split", "dev", "--output", str(output),
                     "--metrics", str(metrics), "--model", str(fake_model)]) == 2
    assert not output.exists() and not metrics.exists()
    assert expected in capsys.readouterr().err


def test_empty_input_and_normal_only_rejection(images, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="empty"):
        load_inputs(empty)
    manifest = write_manifest(tmp_path / "input.csv", [{"path": "a/board.png", "label": 1, "split": "fit_normal"}])
    with pytest.raises(ValueError, match="Normal-only"):
        load_inputs(manifest, "fit_normal", require_labels=True, normal_only=True)


def test_jsonl_manifest(images, tmp_path):
    manifest = tmp_path / "input.jsonl"
    manifest.write_text(json.dumps({"sample_id": "a", "path": "a/board.png", "label": 0, "split": "dev"}) + "\n")
    assert load_inputs(manifest, "dev", require_labels=True)[0].path == images[0]
    manifest.write_text("{bad json}\n")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_inputs(manifest)


def test_overlay_alpha_changes_only_display(images, fake_model, tmp_path):
    outputs = []
    for alpha in (0, 1):
        output = tmp_path / f"result-{alpha}.csv"
        assert cli.main(["predict", "--input", str(images[0]), "--output", str(output), "--model", str(fake_model),
                         "--artifacts-dir", str(tmp_path / f"artifacts-{alpha}"), "--overlay-alpha", str(alpha)]) == 0
        row = next(csv.DictReader(output.open()))
        outputs.append(row)
        overlay = np.asarray(Image.open(tmp_path / row["overlay_path"]))
        target = np.asarray(Image.open(tmp_path / row["original_path" if alpha == 0 else "heatmap_path"]))
        np.testing.assert_array_equal(overlay, target)
    assert outputs[0]["score"] == outputs[1]["score"]
    assert outputs[0]["decision"] == outputs[1]["decision"]
    np.testing.assert_array_equal(np.load(tmp_path / outputs[0]["raw_map_path"]), np.load(tmp_path / outputs[1]["raw_map_path"]))
