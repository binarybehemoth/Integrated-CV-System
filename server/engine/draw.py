"""A small, reusable debug visualiser for the world model.

In production the browser draws overlays; this is for development.
Later chapters extend draw_world to render masks, keypoints, and
scene-graph relations.
"""
from __future__ import annotations
import cv2
import numpy as np

from .world_model import WorldModel


def _color(label: str) -> tuple[int, int, int]:
    h = abs(hash(label))
    return (37 + h % 180, 60 + (h >> 8) % 180, 90 + (h >> 16) % 160)


def draw_world(frame: np.ndarray, world: WorldModel) -> np.ndarray:
    out = frame.copy()
    for o in world.objects:
        c = _color(o.cls)
        x1, y1, x2, y2 = (int(v) for v in
                          (o.box.x1, o.box.y1, o.box.x2, o.box.y2))
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        tag = f"{o.cls} {o.confidence:.2f}"
        if o.track_id is not None:
            tag = f"#{o.track_id} " + tag
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX,
                                      0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), c, -1)
        cv2.putText(out, tag, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out
