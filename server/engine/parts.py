"""Part inheritance: decompose an object into nested sub-parts, each a
first-class Object stored in the parent's parts list.

For people we derive part boxes from pose keypoints (Chapter 11). The
same pattern extends to learned part detectors or studio-drawn parts.
See Chapter 19.
"""
from __future__ import annotations

from .capability import Capability
from .world_model import WorldModel, Object, BBox
from .config import EngineConfig

# Part name -> the keypoints whose extent forms that part's box.
PERSON_PARTS = {
    "head": ["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
    "torso": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    "left_arm": ["left_shoulder", "left_elbow", "left_wrist"],
    "right_arm": ["right_shoulder", "right_elbow", "right_wrist"],
    "left_leg": ["left_hip", "left_knee", "left_ankle"],
    "right_leg": ["right_hip", "right_knee", "right_ankle"],
}


def _box_of(points, pad: float = 6.0) -> BBox:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return BBox(min(xs) - pad, min(ys) - pad,
                max(xs) + pad, max(ys) + pad)


class PartsCapability(Capability):
    name = "parts"

    def __init__(self, config: EngineConfig | None = None,
                 min_points: int = 2):
        self.cfg = config or EngineConfig()
        self.min_points = min_points

    def setup(self) -> None:
        pass

    def apply(self, world: WorldModel, frame) -> None:
        next_id = max((o.id for o in world.objects), default=-1) + 1
        for obj in world.objects:
            if obj.cls != "person" or not obj.keypoints or obj.parts:
                continue
            kp = {k.name: (k.x, k.y) for k in obj.keypoints if k.visible}
            for pname, members in PERSON_PARTS.items():
                pts = [kp[m] for m in members if m in kp]
                if len(pts) < self.min_points:
                    continue
                part = Object(id=next_id, cls=pname, box=_box_of(pts),
                              confidence=obj.confidence)
                part.parent_class = "person"
                obj.parts.append(part)
                next_id += 1
