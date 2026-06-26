"""Class inheritance: a taxonomy over object classes so the system can
reason about categories, not just leaf labels (a car is-a vehicle).

The taxonomy is a simple child -> parent map; the capability stamps
each object's parent_class and ancestor chain. See Chapter 18.
"""
from __future__ import annotations

from .capability import Capability
from .world_model import WorldModel
from .config import EngineConfig

# A small default taxonomy: child -> parent. Extend freely.
DEFAULT_PARENTS = {
    "vehicle": "object", "animal": "object", "person": "object",
    "furniture": "object", "text": "object",
    "face": "person",
    "car": "vehicle", "truck": "vehicle", "bus": "vehicle",
    "motorcycle": "vehicle", "bicycle": "vehicle", "train": "vehicle",
    "dog": "animal", "cat": "animal", "horse": "animal",
    "bird": "animal", "cow": "animal", "sheep": "animal",
    "chair": "furniture", "couch": "furniture", "bed": "furniture",
    "dining table": "furniture",
}


class ClassHierarchy:
    def __init__(self, parents: dict | None = None):
        self.parents = dict(parents or DEFAULT_PARENTS)

    def parent(self, cls: str) -> str | None:
        return self.parents.get(cls)

    def ancestors(self, cls: str) -> list[str]:
        out, seen = [], set()
        cur = self.parents.get(cls)
        while cur is not None and cur not in seen:
            out.append(cur)
            seen.add(cur)
            cur = self.parents.get(cur)
        return out

    def is_descendant(self, cls: str, ancestor: str) -> bool:
        return ancestor in self.ancestors(cls)

    def add(self, cls: str, parent: str) -> None:
        self.parents[cls] = parent


class HierarchyCapability(Capability):
    name = "hierarchy"

    def __init__(self, config: EngineConfig | None = None,
                 hierarchy: ClassHierarchy | None = None):
        self.cfg = config or EngineConfig()
        self.hierarchy = hierarchy or ClassHierarchy()

    def setup(self) -> None:
        pass

    def apply(self, world: WorldModel, frame) -> None:
        for obj in world.objects:
            obj.parent_class = self.hierarchy.parent(obj.cls)
            anc = self.hierarchy.ancestors(obj.cls)
            if anc:
                obj.properties["ancestors"] = anc
