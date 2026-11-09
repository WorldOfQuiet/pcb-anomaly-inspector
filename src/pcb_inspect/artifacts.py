from __future__ import annotations

import os
from pathlib import Path


def save_artifacts(image, raw_map, directory: Path, output_parent: Path, alpha: float) -> dict:
    import cv2
    import numpy as np
    from PIL import Image
    if not directory.resolve().is_relative_to(directory.parent.resolve()):
        raise ValueError("Artifact subdirectory resolves outside the requested output directory")
    for filename in ("original.png", "raw_map.npy", "heatmap.png", "overlay.png"):
        if not (directory / filename).resolve().is_relative_to(directory.resolve()):
            raise ValueError("Artifact file resolves outside its sample directory")
    directory.mkdir(parents=True, exist_ok=True)
    image.save(directory / "original.png")
    np.save(directory / "raw_map.npy", raw_map, allow_pickle=False)
    normalized = (raw_map - raw_map.min()) / max(float(np.ptp(raw_map)), 1e-8)
    color = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[:, :, ::-1]
    heatmap = Image.fromarray(color).resize(image.size, Image.Resampling.BILINEAR)
    heatmap.save(directory / "heatmap.png")
    Image.blend(image, heatmap, alpha).save(directory / "overlay.png")
    return {f"{key}_path": os.path.relpath(directory / filename, output_parent)
            for key, filename in (("original", "original.png"), ("heatmap", "heatmap.png"),
                                  ("overlay", "overlay.png"), ("raw_map", "raw_map.npy"))}
