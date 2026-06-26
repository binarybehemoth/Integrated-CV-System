"""Fingerprint and thumbprint recognition by minutiae.

A readable, classical pipeline: enhance, binarise, thin to a 1-pixel
ridge skeleton, extract minutiae by the crossing-number method, and
match two minutiae sets. For production accuracy use SourceAFIS or the
NIST NBIS tools (MINDTCT + BOZORTH3); this module shows the mechanism.
See Chapter 16.
"""
from __future__ import annotations
from dataclasses import dataclass

import cv2
import numpy as np
# scikit-image is an optional extra used only for ridge skeletonisation here.
# It is not installed by default (this module is not part of the live chain);
# install it with `pip install scikit-image` if you use the fingerprint code.


@dataclass
class Minutia:
    x: int
    y: int
    angle: float
    kind: str          # "ending" or "bifurcation"


def orientation_field(gray: np.ndarray, block: int = 16):
    """Block-wise ridge orientation from gradient structure tensors."""
    g = gray.astype(np.float64)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    h, w = g.shape
    ori = np.zeros((h // block, w // block))
    for by in range(h // block):
        for bx in range(w // block):
            ys = slice(by * block, (by + 1) * block)
            xs = slice(bx * block, (bx + 1) * block)
            gxx, gyy = gx[ys, xs], gy[ys, xs]
            vx = float((2 * gxx * gyy).sum())
            vy = float((gxx ** 2 - gyy ** 2).sum())
            ori[by, bx] = 0.5 * np.arctan2(vx, vy)
    return ori, block


def enhance(gray: np.ndarray) -> np.ndarray:
    """Normalise, equalise, and adaptively binarise to ridge pixels."""
    g = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    g = cv2.equalizeHist(g)
    binar = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 25, 10)
    return (binar // 255).astype(np.uint8)


def _crossing_number(skel: np.ndarray, ori, block: int) -> list[Minutia]:
    out: list[Minutia] = []
    h, w = skel.shape
    oh, ow = ori.shape
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skel[y, x] != 1:
                continue
            nb = [skel[y - 1, x - 1], skel[y - 1, x], skel[y - 1, x + 1],
                  skel[y, x + 1], skel[y + 1, x + 1], skel[y + 1, x],
                  skel[y + 1, x - 1], skel[y, x - 1], skel[y - 1, x - 1]]
            cn = 0.5 * sum(abs(int(nb[i]) - int(nb[i + 1])) for i in range(8))
            kind = "ending" if cn == 1 else "bifurcation" if cn == 3 else None
            if kind is None:
                continue
            angle = float(ori[min(y // block, oh - 1), min(x // block, ow - 1)])
            out.append(Minutia(x, y, angle, kind))
    return out


def extract(gray: np.ndarray) -> list[Minutia]:
    """Full pipeline: grayscale print -> list of minutiae."""
    binar = enhance(gray)
    from skimage.morphology import skeletonize  # optional extra; see top of file
    skel = skeletonize(binar > 0).astype(np.uint8)
    ori, block = orientation_field(gray)
    return _crossing_number(skel, ori, block)


def match_score(a: list[Minutia], b: list[Minutia],
                dist_tol: float = 15.0) -> float:
    """Greedy nearest-neighbour minutiae match (illustrative).

    Assumes the two prints are roughly aligned. Real matchers
    (BOZORTH3, SourceAFIS) estimate the alignment first.
    """
    used: set[int] = set()
    matched = 0
    for ma in a:
        best, best_d = None, dist_tol
        for j, mb in enumerate(b):
            if j in used or mb.kind != ma.kind:
                continue
            d = ((ma.x - mb.x) ** 2 + (ma.y - mb.y) ** 2) ** 0.5
            if d < best_d:
                best, best_d = j, d
        if best is not None:
            used.add(best)
            matched += 1
    return matched / max(1, min(len(a), len(b)))


class FingerprintMatcher:
    """Enroll/identify against a gallery of minutiae sets."""

    def __init__(self, threshold: float = 0.3):
        self.gallery: dict[str, list[Minutia]] = {}
        self.threshold = threshold

    def enroll(self, name: str, gray: np.ndarray) -> int:
        m = extract(gray)
        self.gallery[name] = m
        return len(m)

    def identify(self, gray: np.ndarray) -> tuple[str, float]:
        m = extract(gray)
        best, best_s = "unknown", self.threshold
        for name, ref in self.gallery.items():
            s = match_score(m, ref)
            if s > best_s:
                best, best_s = name, s
        return best, best_s
