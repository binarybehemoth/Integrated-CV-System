"""Active learning: choose the most informative images to label next.

Labelling is the real bottleneck, so rather than labelling at random we
score unlabelled images by how uncertain the current model is on them
and label the most uncertain first. These functions score and rank
detections. See Chapter 29.
"""
from __future__ import annotations
import math
from typing import Iterable


def least_confidence(confidences: list[float]) -> float:
    """Uncertainty = 1 - highest detection confidence. High when the
    model's best guess on an image is weak."""
    if not confidences:
        return 1.0                     # no detections at all is informative
    return 1.0 - max(confidences)


def mean_entropy(per_box_class_probs: list[list[float]]) -> float:
    """Average normalised entropy of the class distribution per box.
    High when the model is torn between classes."""
    if not per_box_class_probs:
        return 1.0
    total = 0.0
    for probs in per_box_class_probs:
        probs = [p for p in probs if p > 0.0]
        if not probs:
            continue
        h = -sum(p * math.log(p) for p in probs)
        total += h / math.log(len(probs)) if len(probs) > 1 else 0.0
    return total / len(per_box_class_probs)


def image_score(confidences: list[float],
                per_box_class_probs: list[list[float]] | None = None,
                w_conf: float = 0.6, w_entropy: float = 0.4) -> float:
    """Blend least-confidence and entropy into one informativeness score."""
    score = w_conf * least_confidence(confidences)
    if per_box_class_probs:
        score += w_entropy * mean_entropy(per_box_class_probs)
    return score


def rank_pool(pool: Iterable[dict], k: int = 20) -> list[dict]:
    """Rank pool items (each: {id, confidences, class_probs?}) by score,
    most informative first, and return the top k to send for labelling."""
    scored = []
    for item in pool:
        s = image_score(item.get("confidences", []),
                        item.get("class_probs"))
        scored.append({**item, "score": round(s, 4)})
    scored.sort(key=lambda d: d["score"], reverse=True)
    return scored[:k]
