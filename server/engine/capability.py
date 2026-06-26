"""The plug-in interface every perception step implements."""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from .world_model import WorldModel


class Capability(ABC):
    """A single perception step that enriches the world model."""

    name: str = "capability"

    def setup(self) -> None:
        """Load models or resources once, before the first frame.
        Override if needed; the default does nothing."""
        return None

    @abstractmethod
    def apply(self, world: WorldModel, frame: np.ndarray) -> None:
        """Read 'frame' and the current 'world', then mutate 'world'
        in place to add this capability's contribution."""
        raise NotImplementedError
