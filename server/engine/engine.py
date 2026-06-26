"""The Engine threads one WorldModel through an ordered capability list."""
from __future__ import annotations
import time
import numpy as np

from .world_model import WorldModel
from .capability import Capability


class Engine:
    def __init__(self, capabilities: list[Capability]):
        self.capabilities = capabilities
        self._frame_id = 0
        self._ready = False

    def setup(self) -> None:
        for cap in self.capabilities:
            cap.setup()
        self._ready = True

    def process(self, frame: np.ndarray) -> WorldModel:
        if not self._ready:
            self.setup()
        world = WorldModel(frame_id=self._frame_id, timestamp=time.time())
        for cap in self.capabilities:
            cap.apply(world, frame)
        self._frame_id += 1
        return world
