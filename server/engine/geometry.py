"""Geometric helpers reused by NMS, tracking, evaluation, scene graph."""
from __future__ import annotations


def iou(a, b) -> float:
    """Intersection over Union of two (x1, y1, x2, y2) boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def nms(boxes, scores, iou_thr: float = 0.45):
    """A readable, scalar non-maximum suppression for one class."""
    order = sorted(range(len(boxes)), key=lambda i: scores[i],
                   reverse=True)
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if iou(boxes[i], boxes[j]) < iou_thr]
    return keep
