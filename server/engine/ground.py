"""Ground plane and horizon: locate where each object meets the ground
and estimate relative (or, if calibrated, metric) distance.

Uncalibrated, it orders objects by depth from their foot position.
Given camera height, focal length, and the horizon line, it estimates
metric ground distance by perspective. See Chapter 21.
"""
from __future__ import annotations

from .capability import Capability
from .world_model import WorldModel
from .config import EngineConfig


class GroundPlaneCapability(Capability):
    name = "ground"

    def __init__(self, config: EngineConfig | None = None,
                 horizon_frac: float = 0.5,
                 camera_height_m: float | None = None,
                 focal_px: float | None = None):
        self.cfg = config or EngineConfig()
        self.horizon_frac = horizon_frac
        self.camera_height_m = camera_height_m
        self.focal_px = focal_px

    def setup(self) -> None:
        pass

    def apply(self, world: WorldModel, frame) -> None:
        h, w = frame.shape[:2]
        horizon_y = self.horizon_frac * h
        on_ground = []
        for obj in world.objects:
            fx = (obj.box.x1 + obj.box.x2) / 2.0
            fy = obj.box.y2                       # box bottom = contact
            obj.properties["foot"] = [round(fx, 1), round(fy, 1)]
            d = fy - horizon_y
            if d <= 1.0:                          # at/above horizon
                continue
            if self.camera_height_m and self.focal_px:
                dist = self.camera_height_m * self.focal_px / d
                obj.properties["ground_distance_m"] = round(dist, 2)
            obj.properties["depth_proxy"] = round(1000.0 / d, 1)
            on_ground.append(obj)

        # Closer objects have a larger foot_y (lower in the image).
        on_ground.sort(key=lambda o: o.box.y2, reverse=True)
        for rank, obj in enumerate(on_ground):
            obj.properties["depth_rank"] = rank
