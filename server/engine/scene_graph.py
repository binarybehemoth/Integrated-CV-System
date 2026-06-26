"""Scene-graph relations: connect objects with directed, labelled
relations computed from geometry and pose, populating world.graph.

Relations produced here: contains, on (support), holding, near. For
richer learned relations (sitting-on, wearing, looking-at) see the
discussion of relationship detection in Chapter 20.
"""
from __future__ import annotations

from .capability import Capability
from .world_model import WorldModel
from .config import EngineConfig


def _box(o):
    return (o.box.x1, o.box.y1, o.box.x2, o.box.y2)


def _contains_frac(a, b) -> float:
    """Fraction of b's area that lies inside a."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    barea = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / barea


def _gap(a, b) -> float:
    """Smallest distance between two boxes (0 if they overlap)."""
    dx = max(0.0, b[0] - a[2], a[0] - b[2])
    dy = max(0.0, b[1] - a[3], a[1] - b[3])
    return (dx * dx + dy * dy) ** 0.5


def _h_overlap(a, b) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0]))


class SceneGraphCapability(Capability):
    name = "scene_graph"

    def __init__(self, config: EngineConfig | None = None,
                 near_frac: float = 0.15):
        self.cfg = config or EngineConfig()
        self.near_frac = near_frac

    def setup(self) -> None:
        pass

    def apply(self, world: WorldModel, frame) -> None:
        objs = [o for o in world.objects if o.cls != "text"]
        h, w = frame.shape[:2]
        diag = (h * h + w * w) ** 0.5
        near_thresh = self.near_frac * diag

        for a in objs:
            ba = _box(a)
            aw = ba[2] - ba[0]
            for b in objs:
                if a.id == b.id:
                    continue
                bb = _box(b)
                if _contains_frac(ba, bb) > 0.85:
                    world.graph.add(a.id, "contains", b.id)
                    continue
                # support: a's base rests on b's top, with x-overlap
                if (_h_overlap(ba, bb) > 0.3 * aw
                        and abs(ba[3] - bb[1]) < 0.1 * diag
                        and ba[3] <= bb[3]):
                    world.graph.add(a.id, "on", b.id)
                    continue
                if a.id < b.id and _gap(ba, bb) < near_thresh:
                    world.graph.add(a.id, "near", b.id)

        self._holding(world)

    def _holding(self, world: WorldModel) -> None:
        people = [o for o in world.objects
                  if o.cls == "person" and o.keypoints]
        others = [o for o in world.objects if o.cls != "person"]
        for p in people:
            wrists = [(k.x, k.y) for k in p.keypoints
                      if k.name in ("left_wrist", "right_wrist")
                      and k.visible]
            if not wrists:
                continue
            for o in others:
                ox1, oy1, ox2, oy2 = _box(o)
                pad = 0.1 * ((ox2 - ox1) + (oy2 - oy1))
                for (wx, wy) in wrists:
                    if (ox1 - pad <= wx <= ox2 + pad
                            and oy1 - pad <= wy <= oy2 + pad):
                        world.graph.add(p.id, "holding", o.id)
                        break
