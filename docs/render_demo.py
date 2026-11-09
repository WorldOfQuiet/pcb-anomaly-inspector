"""Render the README demonstration from saved CLI outputs.

Run from any directory with docs/figure-requirements.txt installed.
No model inference, score editing or image generation is performed here.
"""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle

from figure_style import BG, GREEN, INK, LINE, MUTED, arrow, box, canvas, photo, save, step, text

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    with (DOCS / "example-prediction.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    row = rows[0]
    assert digest(ROOT / row["input_path"]) == row["input_sha256"]
    original_path, overlay_path = (DOCS / row[k] for k in ("original_path", "overlay_path"))
    original = np.asarray(Image.open(original_path).convert("RGB"))
    overlay = np.asarray(Image.open(overlay_path).convert("RGB"))
    response = np.load(DOCS / row["raw_map_path"], allow_pickle=False)
    py, px = np.unravel_index(response.argmax(), response.shape)
    cy = (py + .5) / response.shape[0] * original.shape[0]
    cx = (px + .5) / response.shape[1] * original.shape[1]
    side = 310
    x0 = int(np.clip(cx - side / 2, 0, original.shape[1] - side))
    y0 = int(np.clip(cy - side / 2, 0, original.shape[0] - side))
    roi = (x0, y0, side, side)

    fig, ax = canvas()
    text(ax, 80, 39, "PCB-INSPECT  /  EXAMPLE OUTPUT", 12, GREEN, mono=True)
    text(ax, 80, 75, "PCB anomaly inspection", 34, weight="bold")
    text(ax, 80, 132, "VisA / PCB1     ·     PatchCore + ResNet18     ·     CPU inference", 15, MUTED)
    ax.plot([80, 1720], [173, 173], color=LINE, linewidth=1)

    step(ax, 80, 208, "01", "Input photograph")
    step(ax, 740, 208, "02", "Anomaly overlay")
    step(ax, 1400, 208, "03", "Peak region")
    box(ax, 80, 250, 610, 490)
    box(ax, 740, 250, 610, 490)
    box(ax, 1400, 250, 320, 490)
    original_geo = photo(ax, original, 95, 265, 580, 450)
    overlay_geo = photo(ax, overlay, 755, 265, 580, 450)
    for geometry, color in [(original_geo, GREEN), (overlay_geo, "white")]:
        left, top, scale = geometry
        ax.add_patch(Rectangle((left + x0 * scale, top + y0 * scale), side * scale, side * scale,
                                fill=False, edgecolor=color, linewidth=1.9, zorder=4))
    arrow(ax, (700, 490), (730, 490))
    arrow(ax, (1360, 490), (1390, 490))
    photo(ax, original[y0:y0+side, x0:x0+side], 1420, 270, 280, 185)
    photo(ax, overlay[y0:y0+side, x0:x0+side], 1420, 475, 280, 185)
    text(ax, 1420, 678, "Same region,", 13, MUTED)
    text(ax, 1420, 703, "before and after overlay", 11, MUTED)
    text(ax, 105, 758, "Full input frame", 12, MUTED)
    text(ax, 765, 758, "Heatmap opacity: 0.45", 12, MUTED)
    text(ax, 1400, 758, "Crop centered on map maximum", 10.5, MUTED)

    ax.plot([80, 1720], [813, 813], color=LINE, linewidth=1)
    for x, label, value, color in [
        (80, "anomaly_score", f"{float(row['score']):.4f}", INK),
        (640, "threshold", f"{float(row['threshold']):.4f}", INK),
        (1200, "decision", row["decision"], "#98612b"),
    ]:
        text(ax, x, 843, label, 13, MUTED, mono=True)
        text(ax, x, 875, value, 32, color, mono=True)
    box(ax, 80, 962, 1640, 63, fill="#e9eee7", edge="#e9eee7", radius=9)
    command = "pcb-inspect predict --input examples/images/anomaly.jpg --output results.csv --artifacts-dir maps --overlay-alpha 0.45"
    text(ax, 102, 984, "$ " + command, 12, INK, mono=True)
    text(ax, 80, 1050, "Heatmap uses a per-image display scale. The score is not a defect probability.", 11.5, MUTED)
    text(ax, 80, 1078, "Data: VisA / PCB1 · Yang Zou et al. · CC BY 4.0 · Development example", 11, MUTED)
    save(fig, DOCS / "images" / "inspection-demo")
    record = {"input_sha256": row["input_sha256"], "prediction_csv_sha256": digest(DOCS / "example-prediction.csv"),
              "original_png_sha256": digest(original_path), "overlay_png_sha256": digest(overlay_path),
              "raw_map_sha256": digest(DOCS / row["raw_map_path"]),
              "detail_roi_xywh": roi, "roi_rule": "310-pixel square centered on the maximum of the saved raw anomaly map",
              "score": float(row["score"]), "threshold": float(row["threshold"]), "decision": row["decision"],
              "images_modified": "Layout, scale, crop and outline annotations only; model artifacts and predictions unchanged"}
    (DOCS / "images" / "inspection-demo.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
