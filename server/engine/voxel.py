"""Voxel reconstruction: represent shape as a 3D grid of occupied
cells. Build it from multiple silhouettes (visual hull / space
carving) or by fusing depth maps (TSDF, described in Chapter 23).

This module provides the grid and a space-carving step; it is a
utility (reconstruction is heavier than per-frame perception) rather
than an engine capability. See Chapter 23.
"""
from __future__ import annotations

import numpy as np


class VoxelGrid:
    def __init__(self, dims=(64, 64, 64), origin=(0.0, 0.0, 0.0),
                 voxel_size: float = 1.0):
        self.occ = np.zeros(dims, dtype=bool)
        self.origin = np.asarray(origin, dtype=float)
        self.voxel_size = voxel_size

    @property
    def dims(self):
        return self.occ.shape

    def fill(self) -> None:
        self.occ[:] = True

    def count(self) -> int:
        return int(self.occ.sum())

    def centers(self) -> np.ndarray:
        """World-space centres of all occupied voxels."""
        idx = np.argwhere(self.occ)
        return self.origin + (idx + 0.5) * self.voxel_size

    def to_sparse(self) -> list:
        return np.argwhere(self.occ).tolist()


def carve(grid: VoxelGrid, silhouette: np.ndarray, project) -> int:
    """Space carving: drop occupied voxels that project outside the
    silhouette in one view.

    silhouette: boolean (H, W) foreground mask for this view.
    project:    callable mapping a world point -> (u, v) pixel.
    Call once per view; the visual hull is what survives all views.
    """
    idx = np.argwhere(grid.occ)
    if len(idx) == 0:
        return 0
    centers = grid.origin + (idx + 0.5) * grid.voxel_size
    h, w = silhouette.shape
    keep = np.zeros(len(centers), dtype=bool)
    for i, c in enumerate(centers):
        u, v = project(c)
        ui, vi = int(u), int(v)
        if 0 <= ui < w and 0 <= vi < h and silhouette[vi, ui]:
            keep[i] = True
    grid.occ[tuple(idx[~keep].T)] = False
    return grid.count()
