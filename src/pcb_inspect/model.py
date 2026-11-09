from __future__ import annotations

import io
import math
import time
import warnings
from pathlib import Path

from .common import object_hash, read_json, sha256

BACKBONE_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
SPEC = {
    "backend": "anomalib.models.image.patchcore.torch_model.PatchcoreModel",
    "backend_version": "2.6.2",
    "backbone": "resnet18",
    "layers": ["layer2", "layer3"],
    "num_neighbors": 9,
    "precision": "float32",
    "preprocessing": {
        "image_size": [256, 256], "resize": "full_frame", "crop": "none",
        "interpolation": "Pillow.BILINEAR", "color": "RGB", "exif_transpose": True,
        "scale": 255, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
    },
    "automatic_postprocessing": False,
}


def fingerprint(weights_hash: str, spec: dict) -> str:
    return object_hash({"weights_sha256": weights_hash, "spec": spec})


def inspect_model(directory: Path, require_threshold: bool = True) -> tuple[dict, dict | None]:
    info = read_json(directory / "model.json")
    if info.get("schema_version") != 1 or info.get("spec") != SPEC:
        raise ValueError("Unsupported model schema or preprocessing; expected pcb-inspect model schema 1")
    if sha256(directory / "model.pt") != info["weights_sha256"]:
        raise ValueError("Model weights checksum mismatch")
    if fingerprint(info["weights_sha256"], info["spec"]) != info["model_sha256"]:
        raise ValueError("Model fingerprint mismatch")
    fit_registry = directory / "fit_inputs.json"
    if sha256(fit_registry) != info["fit_inputs_sha256"]:
        raise ValueError("Fit input registry checksum mismatch")
    policy = None
    if (directory / "threshold.json").exists():
        if not info.get("threshold_sha256") or sha256(directory / "threshold.json") != info["threshold_sha256"]:
            raise ValueError("Threshold checksum mismatch")
        policy = read_json(directory / "threshold.json")
        if (policy.get("model_sha256") != info["model_sha256"]
                or policy.get("weights_sha256") != info["weights_sha256"]
                or policy.get("comparison") != "greater_than"):
            raise ValueError("Threshold and model mismatch")
        threshold = policy.get("threshold")
        if threshold is None or not isinstance(threshold, (float, int)) or not math.isfinite(threshold):
            raise ValueError("Model threshold must be finite and calibrated")
    elif require_threshold:
        raise ValueError("Model is not calibrated; run calibrate into a new model directory")
    elif info.get("threshold_sha256"):
        raise ValueError("Calibrated model is missing threshold.json")
    return info, policy


def decode_image(filename: Path):
    from PIL import Image, ImageOps
    if filename.stat().st_size > 20 * 1024 * 1024:
        raise ValueError("Image exceeds 20 MiB")
    data = filename.read_bytes()
    if len(data) > 20 * 1024 * 1024:
        raise ValueError("Image exceeds 20 MiB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise ValueError("Only JPEG and PNG images are supported")
                if image.width * image.height > 20_000_000 or min(image.size) < 16:
                    raise ValueError("Image must be at least 16 x 16 pixels and at most 20 megapixels")
                image.load()
                return ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("Cannot decode image as JPEG or PNG") from exc


def configure(device: str, threads: int, seed: int = 42):
    import numpy as np
    import torch
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    target = torch.device(device)
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; select --device cpu")
        torch.cuda.set_device(target)
    return target


def make_model():
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel
    return PatchcoreModel(backbone=SPEC["backbone"], layers=SPEC["layers"],
                          pre_trained=False, num_neighbors=SPEC["num_neighbors"])


def preprocess(image):
    import numpy as np
    import torch
    from PIL import Image
    size = SPEC["preprocessing"]["image_size"]
    image = image.resize((size[1], size[0]), Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (tensor - mean) / std


def synchronize(device):
    if device.type == "cuda":
        import torch
        torch.cuda.synchronize(device)


class Inspector:
    def __init__(self, directory: Path, device: str = "cpu", threads: int = 4,
                 require_threshold: bool = True):
        import torch
        self.info, self.policy = inspect_model(directory, require_threshold=require_threshold)
        self.device = configure(device, threads)
        self.threads = threads
        self.model = make_model()
        state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.to(self.device).eval()
        self.threshold = self.policy["threshold"] if self.policy is not None else None

    def forward(self, image):
        import numpy as np
        import torch
        tensor = preprocess(image)[None].to(self.device)
        synchronize(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            prediction = self.model(tensor)
        synchronize(self.device)
        elapsed = time.perf_counter() - start
        score = float(prediction.pred_score[0].cpu())
        raw_map = prediction.anomaly_map[0, 0].cpu().numpy()
        if not math.isfinite(score) or not np.isfinite(raw_map).all():
            raise ValueError("Model returned a non-finite score or anomaly map")
        return score, raw_map, elapsed

    def decision(self, score: float) -> str:
        if self.threshold is None:
            return "needs_review"
        return "suspect" if score > self.threshold else "normal"
