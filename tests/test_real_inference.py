"""Optional real-model test: PCB_INSPECT_REAL_TEST=1 python -m pytest tests/test_real_inference.py."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pcb_inspect.artifacts import save_artifacts
from pcb_inspect.inputs import load_inputs
from pcb_inspect.model import Inspector, decode_image

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.environ.get("PCB_INSPECT_REAL_TEST") != "1", reason="Enable real CPU inference with PCB_INSPECT_REAL_TEST=1")


def test_real_cpu_scores_and_overlay(tmp_path):
    inspector = Inspector(ROOT / "models/pcb1-v1", device="cpu", threads=1)
    rows = load_inputs(ROOT / "examples/manifest.csv")
    reference = json.loads((ROOT / "examples/provenance.json").read_text())["examples"]
    for row, expected in zip(rows, reference, strict=True):
        image = decode_image(row.path)
        score, raw_map, elapsed = inspector.forward(image)
        assert score == pytest.approx(expected["source_score"], abs=1e-4)
        assert inspector.decision(score) == expected["expected_decision"]
        assert np.isfinite(raw_map).all() and raw_map.shape == (256, 256) and elapsed > 0
        if row == rows[0]:
            for alpha in (0, 1):
                output = tmp_path / str(alpha)
                save_artifacts(image, raw_map, output, tmp_path, alpha)
                target = "original.png" if alpha == 0 else "heatmap.png"
                np.testing.assert_array_equal(np.asarray(Image.open(output / "overlay.png")), np.asarray(Image.open(output / target)))
            np.testing.assert_array_equal(np.load(tmp_path / "0/raw_map.npy"), np.load(tmp_path / "1/raw_map.npy"))
