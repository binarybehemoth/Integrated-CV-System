"""Keypoints and pose: attach a skeleton of named landmarks to the
person objects the detector found.

We run a YOLO26-pose model and match each pose instance to an existing
object by box IoU, appending COCO-17 Keypoints. See Chapter 11.
"""
from __future__ import annotations
import numpy as np
from ultralytics import YOLO

from .capability import Capability
from .world_model import WorldModel, Keypoint
from .config import EngineConfig
from .geometry import iou

# COCO-17 keypoint names, in the order YOLO26-pose emits them.
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


class PoseCapability(Capability):
    name = "pose"

    def __init__(self, config: EngineConfig | None = None,
                 match_iou: float = 0.5, kp_threshold: float = 0.3):
        self.cfg = config or EngineConfig()
        self.match_iou = match_iou
        self.kp_threshold = kp_threshold
        self.model: YOLO | None = None

    def setup(self) -> None:
        weights = f"yolo26{self.cfg.model_size}-pose.pt"
        self.model = YOLO(weights)
        self.model.to(self.cfg.device)

    def apply(self, world: WorldModel, frame: np.ndarray) -> None:
        if self.model is None or not world.objects:
            return
        res = self.model.predict(frame, imgsz=self.cfg.img_size,
                                 conf=self.cfg.conf_threshold,
                                 device=self.cfg.device,
                                 half=self.cfg.half, verbose=False)[0]
        if res.keypoints is None:
            return
        kps = res.keypoints.data.cpu().numpy()   # (M, 17, 3) x,y,score
        boxes = res.boxes.xyxy.cpu().numpy()     # (M, 4) xyxy

        for inst, box in zip(kps, boxes):
            best, best_iou = None, self.match_iou
            for obj in world.objects:
                ob = (obj.box.x1, obj.box.y1, obj.box.x2, obj.box.y2)
                score = iou(box, ob)
                if score > best_iou:
                    best, best_iou = obj, score
            if best is None or best.keypoints:
                continue
            for name, (x, y, s) in zip(COCO_KEYPOINTS, inst):
                best.keypoints.append(Keypoint(
                    name=name, x=float(x), y=float(y),
                    visible=bool(s >= self.kp_threshold),
                    score=float(s)))
