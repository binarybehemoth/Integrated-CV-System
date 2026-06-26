"""World-model data structures shared by every capability.

One WorldModel describes a single frame: a list of Objects and a
SceneGraph relating them. Capabilities enrich these structures in
place as the engine threads one WorldModel through them.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class BBox:
    """An axis-aligned bounding box in pixel coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2))

    def iou(self, other: "BBox") -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass
class Keypoint:
    """A named, located landmark with a visibility flag and score."""
    name: str
    x: float
    y: float
    visible: bool = True
    score: float = 0.0


@dataclass
class Object:
    """One detected thing and everything we learn about it."""
    id: int
    cls: str
    box: BBox
    confidence: float = 0.0

    # Filled in by later capabilities; all optional.
    mask: Optional[list] = None
    keypoints: list[Keypoint] = field(default_factory=list)
    parts: list["Object"] = field(default_factory=list)
    properties: dict = field(default_factory=dict)
    state: Optional[str] = None
    track_id: Optional[int] = None
    parent_class: Optional[str] = None

    def is_a(self, category: str, hierarchy=None) -> bool:
        """True if this object's class is or descends from category."""
        if self.cls == category or self.parent_class == category:
            return True
        if hierarchy is not None:
            return hierarchy.is_descendant(self.cls, category)
        return False


@dataclass
class Edge:
    """A directed, labelled relation: subject --label--> object."""
    subject_id: int
    label: str
    object_id: int
    confidence: float = 1.0


@dataclass
class SceneGraph:
    edges: list[Edge] = field(default_factory=list)

    def add(self, subject_id: int, label: str, object_id: int,
            confidence: float = 1.0) -> None:
        self.edges.append(Edge(subject_id, label, object_id, confidence))

    def with_label(self, label: str) -> list[Edge]:
        return [e for e in self.edges if e.label == label]


@dataclass
class WorldModel:
    """Everything known about a single frame."""
    frame_id: int = 0
    timestamp: float = 0.0
    objects: list[Object] = field(default_factory=list)
    graph: SceneGraph = field(default_factory=SceneGraph)

    def by_id(self, oid: int) -> Optional[Object]:
        return next((o for o in self.objects if o.id == oid), None)

    def of_class(self, cls: str) -> list[Object]:
        return [o for o in self.objects if o.cls == cls]

    def to_dict(self) -> dict:
        """Lossless conversion to plain dicts for JSON / storage."""
        return asdict(self)
