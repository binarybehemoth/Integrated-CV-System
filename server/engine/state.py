"""Properties and states: describe each object with derived attributes
and drive small finite-state machines from them.

Properties (colour, size, aspect, motion, zone) are computed cheaply
from data already on the object. States are discrete labels produced
by per-object FSMs that read those properties over time. The tracker
must run first so motion can be measured per track id. See Chapter 13.
"""
from __future__ import annotations
import numpy as np

from .capability import Capability
from .world_model import WorldModel
from .config import EngineConfig


# ---- property extractors --------------------------------------------------

def dominant_color(frame, box, k: int = 3) -> dict | None:
    """Most common colour inside the box, via a tiny k-means."""
    x1, y1, x2, y2 = (int(v) for v in box)
    crop = frame[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return None
    pixels = crop.reshape(-1, 3).astype(np.float32)
    if len(pixels) > 1000:                       # subsample for speed
        idx = np.random.choice(len(pixels), 1000, replace=False)
        pixels = pixels[idx]
    centers = pixels[np.random.choice(len(pixels), k, replace=False)]
    lab = np.zeros(len(pixels), dtype=int)
    for _ in range(5):                           # a few Lloyd iterations
        d = ((pixels[:, None, :] - centers[None]) ** 2).sum(2)
        lab = d.argmin(1)
        for j in range(k):
            sel = pixels[lab == j]
            if len(sel):
                centers[j] = sel.mean(0)
    b, g, r = (int(v) for v in centers[np.bincount(lab, minlength=k).argmax()])
    return {"r": r, "g": g, "b": b, "hex": f"#{r:02x}{g:02x}{b:02x}"}


def color_name(rgb: dict) -> str:
    """A coarse human colour name from an RGB dict."""
    r, g, b = rgb["r"], rgb["g"], rgb["b"]
    if max(r, g, b) < 50:
        return "black"
    if min(r, g, b) > 200:
        return "white"
    if r > 150 and g > 150 and b < 110:
        return "yellow"
    if r >= g and r >= b:
        return "red"
    if g >= r and g >= b:
        return "green"
    if b >= r and b >= g:
        return "blue"
    return "grey"


def zone(cx: float, cy: float, w: int, h: int) -> str:
    col = "left" if cx < w / 3 else "right" if cx > 2 * w / 3 else "centre"
    row = "top" if cy < h / 3 else "bottom" if cy > 2 * h / 3 else "middle"
    return f"{row}-{col}"


# ---- finite-state machine -------------------------------------------------

class FSM:
    """A tiny finite-state machine: {state: [(condition, next_state)]}."""

    def __init__(self, start: str, transitions: dict):
        self.state = start
        self.transitions = transitions

    def step(self, props: dict) -> str:
        for cond, nxt in self.transitions.get(self.state, []):
            if cond(props):
                self.state = nxt
                break
        return self.state


def motion_fsm() -> FSM:
    return FSM("still", {
        "still":  [(lambda p: p.get("speed", 0.0) > 5.0, "moving")],
        "moving": [(lambda p: p.get("speed", 0.0) <= 5.0, "still")],
    })


def door_fsm() -> FSM:
    return FSM("closed", {
        "closed": [(lambda p: p.get("aspect", 0.0) > 0.6, "open")],
        "open":   [(lambda p: p.get("aspect", 0.0) <= 0.6, "closed")],
    })


# ---- the capability -------------------------------------------------------

class StateCapability(Capability):
    name = "state"

    # Map a class name to an FSM factory; classes with no entry use motion.
    FSM_FACTORIES = {"door": door_fsm}

    def __init__(self, config: EngineConfig | None = None):
        self.cfg = config or EngineConfig()
        self._prev: dict[int, tuple] = {}     # track_id -> centroid
        self._fsms: dict[int, FSM] = {}       # track_id -> FSM

    def setup(self) -> None:
        self._prev.clear()
        self._fsms.clear()

    def apply(self, world: WorldModel, frame) -> None:
        h, w = frame.shape[:2]
        for obj in world.objects:
            bw = max(1.0, obj.box.x2 - obj.box.x1)
            bh = max(1.0, obj.box.y2 - obj.box.y1)
            cx = (obj.box.x1 + obj.box.x2) / 2.0
            cy = (obj.box.y1 + obj.box.y2) / 2.0

            p = obj.properties
            p["area"] = round(bw * bh, 1)
            p["aspect"] = round(bw / bh, 2)
            p["zone"] = zone(cx, cy, w, h)

            col = dominant_color(frame, (obj.box.x1, obj.box.y1,
                                         obj.box.x2, obj.box.y2))
            if col is not None:
                p["color"] = col["hex"]
                p["color_name"] = color_name(col)

            tid = obj.track_id
            if tid is not None:
                if tid in self._prev:
                    px, py = self._prev[tid]
                    p["speed"] = round(((cx - px) ** 2 +
                                        (cy - py) ** 2) ** 0.5, 1)
                self._prev[tid] = (cx, cy)

                fsm = self._fsms.get(tid)
                if fsm is None:
                    fsm = self.FSM_FACTORIES.get(obj.cls, motion_fsm)()
                    self._fsms[tid] = fsm
                obj.state = fsm.step(p)
