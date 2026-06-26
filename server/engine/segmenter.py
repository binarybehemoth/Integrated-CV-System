"""Instance segmentation: attach pixel-accurate masks to the objects
the detector already found.

We run a YOLO26-seg model and match each returned instance to an
existing object by box IoU, storing a thinned contour polygon on
object.mask. See Chapter 10.
"""
from __future__ import annotations
import numpy as np
from ultralytics import YOLO

from .capability import Capability
from .world_model import WorldModel
from .config import EngineConfig
from .geometry import iou


def _downsample(poly: np.ndarray, step: int = 2) -> np.ndarray:
    """Thin a contour polygon so the mask is cheap to ship and draw,
    while keeping enough points for a smooth silhouette."""
    if len(poly) <= 12:
        return poly
    return poly[::step]


class SegmenterCapability(Capability):
    name = "segmenter"

    def __init__(self, config: EngineConfig | None = None,
                 match_iou: float = 0.5):
        self.cfg = config or EngineConfig()
        self.match_iou = match_iou
        self.model: YOLO | None = None

    def setup(self) -> None:
        weights = f"yolo26{self.cfg.model_size}-seg.pt"
        self.model = YOLO(weights)
        self.model.to(self.cfg.device)

    def apply(self, world: WorldModel, frame: np.ndarray) -> None:
        if self.model is None or not world.objects:
            return
        res = self.model.predict(frame, imgsz=self.cfg.img_size,
                                 conf=self.cfg.conf_threshold,
                                 device=self.cfg.device,
                                 half=self.cfg.half, verbose=False)[0]
        if res.masks is None:
            return
        polys = res.masks.xy                    # list of (N,2) arrays
        boxes = res.boxes.xyxy.cpu().numpy()    # (M,4) xyxy

        for poly, box in zip(polys, boxes):
            best, best_iou = None, self.match_iou
            for obj in world.objects:
                ob = (obj.box.x1, obj.box.y1, obj.box.x2, obj.box.y2)
                score = iou(box, ob)
                if score > best_iou:
                    best, best_iou = obj, score
            if best is not None and best.mask is None:
                best.mask = _downsample(poly).astype(float).tolist()
