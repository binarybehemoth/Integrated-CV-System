"""Posture, movement, and gait from pose over time.

Posture is read from the current skeleton's geometry; movement comes
from track speed (Chapter 13); action and gait recognition consume a
buffer of recent skeletons -- the interface is here, the model is an
ST-GCN (action) or OpenGait (identity) you train or plug in. See
Chapter 17.
"""
from __future__ import annotations
from collections import deque, defaultdict
import math

from .capability import Capability
from .world_model import WorldModel
from .config import EngineConfig


def keypoint_map(obj) -> dict:
    return {k.name: (k.x, k.y) for k in obj.keypoints if k.visible}


def _mid(p, q):
    return ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)


def _angle(a, b, c) -> float:
    """Interior angle at b of the path a-b-c, in degrees."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    na, nc = math.hypot(*ba), math.hypot(*bc)
    if na * nc == 0:
        return 180.0
    cosv = max(-1.0, min(1.0, (ba[0] * bc[0] + ba[1] * bc[1]) / (na * nc)))
    return math.degrees(math.acos(cosv))


def classify_posture(kp: dict) -> str | None:
    """Standing / sitting / bending / lying from skeleton geometry."""
    need = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if not all(n in kp for n in need):
        return None
    sh = _mid(kp["left_shoulder"], kp["right_shoulder"])
    hp = _mid(kp["left_hip"], kp["right_hip"])
    dx, dy = hp[0] - sh[0], hp[1] - sh[1]
    torso = abs(math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6)))
    if torso > 55:
        return "lying"
    knee = None
    if all(n in kp for n in ("left_hip", "left_knee", "left_ankle")):
        knee = _angle(kp["left_hip"], kp["left_knee"], kp["left_ankle"])
    elif all(n in kp for n in ("right_hip", "right_knee", "right_ankle")):
        knee = _angle(kp["right_hip"], kp["right_knee"], kp["right_ankle"])
    if knee is not None and knee < 120:
        return "sitting"
    if torso > 30:
        return "bending"
    return "standing"


class PostureGaitCapability(Capability):
    name = "posture"

    def __init__(self, config: EngineConfig | None = None,
                 history: int = 30):
        self.cfg = config or EngineConfig()
        self.history = history
        self.buffers: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=history))

    def setup(self) -> None:
        self.buffers.clear()

    def apply(self, world: WorldModel, frame) -> None:
        for obj in world.objects:
            if obj.cls != "person" or not obj.keypoints:
                continue
            kp = keypoint_map(obj)
            posture = classify_posture(kp)
            if posture is not None:
                obj.properties["posture"] = posture
            tid = obj.track_id
            if tid is not None:
                self.buffers[tid].append(kp)
                obj.properties["pose_frames"] = len(self.buffers[tid])
                # A trained ST-GCN / OpenGait would consume
                # list(self.buffers[tid]) here and write an action
                # label or a gait identity onto obj.properties.
