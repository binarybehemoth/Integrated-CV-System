"""Geometric primitives: approximate each object with a simple 3D
shape -- box, cylinder, or sphere -- parameterised from its class and
box. A compact, manipulable proxy for the object's volume.

This is the class-prior route (route A): instant, no depth required.
Route B (back-project a mask to a point cloud and RANSAC-fit a shape)
needs depth from Chapter 24. See Chapter 22.
"""
from __future__ import annotations

from .capability import Capability
from .world_model import WorldModel
from .config import EngineConfig

# Class -> primitive shape prior; anything unlisted defaults to a box.
SHAPE_PRIORS = {
    "sphere": {"sports ball", "orange", "apple", "ball", "donut",
               "frisbee"},
    "cylinder": {"bottle", "cup", "wine glass", "can", "vase", "person",
                 "potted plant", "fire hydrant"},
}


def primitive_for(cls: str) -> str:
    for shape, members in SHAPE_PRIORS.items():
        if cls in members:
            return shape
    return "box"


# Part name -> primitive, so an object can be reconstructed as a
# combination of primitives (head sphere, torso box, limbs cylinders).
PART_PRIORS = {
    "head": "sphere",
    "torso": "box",
    "left_arm": "cylinder", "right_arm": "cylinder",
    "left_leg": "cylinder", "right_leg": "cylinder",
}


def primitive_for_part(name: str) -> str:
    return PART_PRIORS.get(name, "box")


def _params_for(shape: str, w: float, h: float) -> dict:
    if shape == "sphere":
        return {"radius": round((w + h) / 4.0, 1)}
    if shape == "cylinder":
        return {"radius": round(w / 2.0, 1), "height": round(h, 1)}
    return {"width": round(w, 1), "height": round(h, 1), "depth": round(w, 1)}


class GeometryCapability(Capability):
    name = "geometry3d"

    def __init__(self, config: EngineConfig | None = None):
        self.cfg = config or EngineConfig()

    def setup(self) -> None:
        pass

    def apply(self, world: WorldModel, frame) -> None:
        for obj in world.objects:
            w = obj.box.x2 - obj.box.x1
            h = obj.box.y2 - obj.box.y1
            shape = primitive_for(obj.cls)
            obj.properties["primitive"] = {"shape": shape,
                                           **_params_for(shape, w, h)}
            # Give each detected part its own primitive too, so the 3D
            # view can rebuild the object as a set of primitives.
            for part in obj.parts:
                pw = part.box.x2 - part.box.x1
                ph = part.box.y2 - part.box.y1
                pshape = primitive_for_part(part.cls)
                part.properties["primitive"] = {"shape": pshape,
                                                **_params_for(pshape, pw, ph)}
