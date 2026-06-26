"""Multi-phase hierarchical detector capability.

Runs a coarse parent model over the frame, then a per-parent child model
on each parent crop, so many fine classes are handled by several small
models instead of one big one. Emits Objects carrying both the refined
leaf class and the parent_class. Drop-in alternative to DetectorCapability.
"""
from __future__ import annotations
import json
import os

import numpy as np

from .capability import Capability
from .world_model import WorldModel, Object, BBox


class HierarchicalDetector(Capability):
    name = "hier_detector"

    def __init__(self, parent_weights: str,
                 child_weights: dict | None = None,
                 device: str = "cpu", imgsz: int = 640,
                 conf: float = 0.25):
        self.parent_weights = parent_weights
        self.child_weights = child_weights or {}    # parent_name -> path
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.parent = None
        self.children: dict = {}

    @classmethod
    def from_manifest(cls, manifest_path: str, **kw) -> "HierarchicalDetector":
        """Build from the hierarchy.json produced by train_hierarchical."""
        with open(manifest_path) as fh:
            m = json.load(fh)
        return cls(m["parent_model"], m.get("child_models", {}), **kw)

    def setup(self) -> None:
        from ultralytics import YOLO
        self.parent = YOLO(self.parent_weights)
        self.parent.to(self.device)
        for p, w in self.child_weights.items():
            model = YOLO(w)
            model.to(self.device)
            self.children[p] = model

    def _predict(self, model, image):
        return model.predict(image, imgsz=self.imgsz, conf=self.conf,
                             device=self.device, verbose=False)[0]

    def apply(self, world: WorldModel, frame: np.ndarray) -> None:
        assert self.parent is not None, "call setup() first"
        H, W = frame.shape[:2]
        pres = self._predict(self.parent, frame)
        if pres.boxes is None:
            return

        nid = len(world.objects)
        for b in pres.boxes:
            pcls = self.parent.names.get(int(b.cls), str(int(b.cls)))
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            child = self.children.get(pcls)

            # Single-leaf parent (no child model): emit the parent box.
            if child is None:
                world.objects.append(Object(
                    id=nid, cls=pcls, box=BBox(x1, y1, x2, y2),
                    confidence=float(b.conf)))
                nid += 1
                continue

            # Phase 2: crop to the parent box and run the child model.
            cx1, cy1 = max(0, int(x1)), max(0, int(y1))
            cx2, cy2 = min(W, int(x2)), min(H, int(y2))
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            cres = self._predict(child, crop)
            if cres.boxes is None or len(cres.boxes) == 0:
                # No refinement: keep the parent label as the class.
                world.objects.append(Object(
                    id=nid, cls=pcls, box=BBox(x1, y1, x2, y2),
                    confidence=float(b.conf)))
                nid += 1
                continue

            # Take the most confident child box, map back to full frame.
            best = max(cres.boxes, key=lambda bb: float(bb.conf))
            lx1, ly1, lx2, ly2 = (float(v) for v in best.xyxy[0])
            leaf = child.names.get(int(best.cls), str(int(best.cls)))
            world.objects.append(Object(
                id=nid, cls=leaf, parent_class=pcls,
                box=BBox(cx1 + lx1, cy1 + ly1, cx1 + lx2, cy1 + ly2),
                confidence=float(best.conf)))
            nid += 1
