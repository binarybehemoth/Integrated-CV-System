"""Multi-object tracking: give every object a stable id across frames.

A compact, dependency-free tracking-by-detection tracker: match this
frame's detections to existing tracks by IoU (greedy, high score
first), spawn ids for the unmatched, and age out tracks that go
unseen. This is the ByteTrack idea in miniature; swap in a Kalman
filter or the ultralytics trackers when you need more. See Chapter 12.
"""
from __future__ import annotations
from dataclasses import dataclass

from .capability import Capability
from .world_model import WorldModel
from .config import EngineConfig
from .geometry import iou


@dataclass
class Track:
    track_id: int
    box: tuple            # (x1, y1, x2, y2)
    cls: str
    age: int = 0          # frames since last matched
    hits: int = 1         # total matches


class TrackerCapability(Capability):
    name = "tracker"

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: list[Track] = []
        self._next_id = 0

    def setup(self) -> None:
        self.tracks = []
        self._next_id = 0

    def apply(self, world: WorldModel, frame) -> None:
        # Detections, highest confidence first (the ByteTrack order).
        dets = sorted(world.objects, key=lambda o: o.confidence,
                      reverse=True)
        unmatched = set(range(len(self.tracks)))

        for obj in dets:
            box = (obj.box.x1, obj.box.y1, obj.box.x2, obj.box.y2)
            best, best_iou = None, self.iou_threshold
            for ti in unmatched:
                tr = self.tracks[ti]
                if tr.cls != obj.cls:
                    continue
                score = iou(box, tr.box)
                if score > best_iou:
                    best, best_iou = ti, score
            if best is not None:
                tr = self.tracks[best]
                tr.box, tr.age = box, 0
                tr.hits += 1
                obj.track_id = tr.track_id
                unmatched.discard(best)
            else:
                tr = Track(self._next_id, box, obj.cls)
                self._next_id += 1
                self.tracks.append(tr)
                obj.track_id = tr.track_id

        # Age the tracks we did not match; drop the stale ones.
        for ti in unmatched:
            self.tracks[ti].age += 1
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
