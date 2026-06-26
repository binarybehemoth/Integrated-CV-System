"""DetectorCapability: the first capability and the only one that
creates Objects. Loads YOLO26, runs it on a frame, and populates the
world model with one Object per detection.
"""
from __future__ import annotations
import numpy as np
from ultralytics import YOLO

from .capability import Capability
from .world_model import WorldModel, Object, BBox
from .config import EngineConfig


class DetectorCapability(Capability):
    name = "detector"

    def __init__(self, config: EngineConfig | None = None,
                 classes: list[str] | None = None):
        self.cfg = config or EngineConfig()
        self.only = set(classes) if classes else None
        self.model: YOLO | None = None
        self.names: dict[int, str] = {}
        self._counter = 0
        self._cache: list[tuple] = []     # last detections (label, box, conf)

    def setup(self) -> None:
        self.model = YOLO(self.cfg.detect_weights)
        self.model.to(self.cfg.device)
        self.names = self.model.names

        # Warm up so the first real frame is not slow.
        dummy = np.zeros((self.cfg.img_size, self.cfg.img_size, 3),
                         dtype=np.uint8)
        self.model.predict(dummy, imgsz=self.cfg.img_size,
                           device=self.cfg.device,
                           half=self.cfg.half, verbose=False)

    def _run_model(self, frame: np.ndarray) -> list[tuple]:
        results = self.model.predict(
            frame,
            imgsz=self.cfg.img_size,
            conf=self.cfg.conf_threshold,
            iou=self.cfg.iou_threshold,
            max_det=self.cfg.max_detections,
            device=self.cfg.device,
            half=self.cfg.half,
            verbose=False,
        )
        r = results[0]
        dets: list[tuple] = []
        if r.boxes is None:
            return dets
        for b in r.boxes:
            cls_idx = int(b.cls)
            label = self.names.get(cls_idx, str(cls_idx))
            if self.only is not None and label not in self.only:
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            dets.append((label, (x1, y1, x2, y2), float(b.conf)))
        return dets

    def apply(self, world: WorldModel, frame: np.ndarray) -> None:
        assert self.model is not None, "call setup() first"

        # Run the detector only every detect_interval frames; on the
        # skipped frames reuse the last detections so the tracker and the
        # rest of the chain keep working with far fewer model calls.
        interval = max(1, getattr(self.cfg, "detect_interval", 1))
        if self._counter % interval == 0:
            self._cache = self._run_model(frame)
        self._counter += 1

        next_id = len(world.objects)
        for (label, (x1, y1, x2, y2), conf) in self._cache:
            world.objects.append(Object(
                id=next_id,
                cls=label,
                box=BBox(x1, y1, x2, y2),
                confidence=conf,
            ))
            next_id += 1

    @staticmethod
    def detections_summary(world: WorldModel) -> str:
        counts: dict[str, int] = {}
        for o in world.objects:
            counts[o.cls] = counts.get(o.cls, 0) + 1
        parts = [f"{n}x {c}" for c, n in sorted(counts.items())]
        return ", ".join(parts) if parts else "(nothing detected)"


if __name__ == "__main__":
    import sys, cv2
    from .engine import Engine

    path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    frame = cv2.imread(path)
    if frame is None:
        raise SystemExit(f"cannot read image: {path}")

    engine = Engine([DetectorCapability()])
    world = engine.process(frame)
    print("Detected:", DetectorCapability.detections_summary(world))
    for o in world.objects:
        cx, cy = o.box.center
        print(f"  #{o.id} {o.cls:12s} conf={o.confidence:.2f} "
              f"center=({cx:.0f},{cy:.0f})")
