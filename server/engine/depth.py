"""Monocular depth: a dense depth map from a single image, plus the
back-projection that lifts pixels to 3D points.

Backend-agnostic with a DPT / MiDaS default via Hugging Face. The
predicted depth is relative (affine-invariant); for metric depth use a
metric model such as ZoeDepth or Metric3D. See Chapter 24.
"""
from __future__ import annotations
import numpy as np

from .capability import Capability
from .world_model import WorldModel
from .config import EngineConfig


def backproject(u: float, v: float, z: float,
                fx: float, fy: float, cx: float, cy: float):
    """Pixel (u, v) at depth z -> 3D point (X, Y, Z) under a pinhole."""
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return (x, y, z)


class DepthCapability(Capability):
    name = "depth"

    def __init__(self, config: EngineConfig | None = None,
                 model_name: str = "Intel/dpt-hybrid-midas"):
        self.cfg = config or EngineConfig()
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.last_depth = None        # cache the most recent map

    def setup(self) -> None:
        from transformers import (AutoImageProcessor,
                                  AutoModelForDepthEstimation)
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(
            self.model_name)
        self.model.eval()
        if str(self.cfg.device).startswith("cuda"):
            self.model.to("cuda")

    def infer_depth(self, frame) -> np.ndarray:
        import cv2
        import torch
        from PIL import Image
        img = Image.fromarray(frame[:, :, ::-1])      # BGR -> RGB
        inputs = self.processor(images=img, return_tensors="pt")
        if str(self.cfg.device).startswith("cuda"):
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            pred = self.model(**inputs).predicted_depth
        depth = pred.squeeze().cpu().numpy()
        return cv2.resize(depth, (frame.shape[1], frame.shape[0]))

    def apply(self, world: WorldModel, frame) -> None:
        if self.model is None:
            return
        depth = self.infer_depth(frame)
        self.last_depth = depth
        h, w = depth.shape
        for obj in world.objects:
            x1, y1 = max(0, int(obj.box.x1)), max(0, int(obj.box.y1))
            x2, y2 = min(w, int(obj.box.x2)), min(h, int(obj.box.y2))
            patch = depth[y1:y2, x1:x2]
            if patch.size:
                obj.properties["depth"] = round(float(np.median(patch)), 3)
