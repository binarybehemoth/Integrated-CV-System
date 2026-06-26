"""A small, trainable orientation ("pose") head for geometric primitives.

The detector gives a 2-D box (the translation part of a 6-DoF pose); this head
supplies the *rotation* part. For every annotated part the studio stores a
rotation angle and a 3-D rotation axis. This module learns to regress those
from the part's image crop, so at inference the rotation is *predicted from
pixels* rather than copied from a fixed annotation.

Output parametrisation (5 numbers, all continuous and differentiable):

    [cos(theta), sin(theta), ax, ay, az]

theta is recovered with atan2 (no wrap-around discontinuity) and the axis is
L2-normalised. Together with the detection box + depth, this yields a full
6-DoF pose (3-DoF translation from the box, 3-DoF rotation from this head).

Everything degrades gracefully if torch / cv2 are unavailable: training and
prediction simply become no-ops and the caller falls back to the static
primitive manifest.
"""
from __future__ import annotations

import json
import math
import os
from typing import List, Optional, Tuple

try:
    import numpy as np
except Exception:                       # pragma: no cover
    np = None

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except Exception:                       # pragma: no cover
    torch = None
    nn = object
    _TORCH = False

try:
    import cv2
except Exception:                       # pragma: no cover
    cv2 = None

CROP = 64                               # network input resolution


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
if _TORCH:
    class OrientationNet(nn.Module):
        """Lightweight conv regressor: 64x64x3 -> [cos, sin, ax, ay, az]."""

        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(3, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),   # 32
                nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),  # 16
                nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),  # 8
                nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Sequential(
                nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 5))

        def forward(self, x):
            z = self.body(x).flatten(1)
            out = self.head(z)
            ang = out[:, :2]
            axis = out[:, 2:]
            ang = ang / (ang.norm(dim=1, keepdim=True) + 1e-8)   # unit (cos,sin)
            axis = axis / (axis.norm(dim=1, keepdim=True) + 1e-8)
            return torch.cat([ang, axis], dim=1)


def _prep(crop_bgr) -> Optional["np.ndarray"]:
    """BGR uint8 crop -> 3x64x64 float32 in [0,1] (RGB)."""
    if cv2 is None or np is None or crop_bgr is None or crop_bgr.size == 0:
        return None
    img = cv2.resize(crop_bgr, (CROP, CROP))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    return img.transpose(2, 0, 1)


# --------------------------------------------------------------------------- #
# Training-sample collection (called from dataset.build_from_payload)
# --------------------------------------------------------------------------- #
def save_pose_samples(payload: dict, out_dir: str) -> int:
    """Crop every annotated part that carries a rotation and save it with its
    [angle, axis] label under ``out_dir/pose``. Returns the sample count."""
    if cv2 is None or np is None:
        return 0
    pose_dir = os.path.join(out_dir, "pose")
    labels: List[dict] = []
    idx = 0
    for im in payload.get("images", []):
        data = im.get("data") or im.get("dataUrl")
        if not data:
            continue
        b64 = data.split(",", 1)[1] if "," in data else data
        try:
            import base64
            arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            continue
        if frame is None:
            continue
        h, w = frame.shape[:2]
        for o in im.get("objects", []):
            for p in o.get("parts", []):
                prim = p.get("primitive") or {}
                if "rotation" not in prim:
                    continue
                box = p.get("box") or {}
                x1 = int(max(0, box.get("x1", 0))); y1 = int(max(0, box.get("y1", 0)))
                x2 = int(min(w, box.get("x2", 0))); y2 = int(min(h, box.get("y2", 0)))
                if x2 - x1 < 6 or y2 - y1 < 6:
                    continue
                crop = frame[y1:y2, x1:x2]
                if idx == 0:
                    os.makedirs(pose_dir, exist_ok=True)
                fn = f"{idx:05d}.jpg"
                cv2.imwrite(os.path.join(pose_dir, fn), crop)
                ax = float(prim.get("axisX", 0) or 0)
                ay = float(prim.get("axisY", 1) or 0)
                az = float(prim.get("axisZ", 0) or 0)
                labels.append({"file": fn, "angle": float(prim.get("rotation", 0) or 0),
                               "axis": [ax, ay, az]})
                idx += 1
    if labels:
        with open(os.path.join(pose_dir, "labels.json"), "w", encoding="utf-8") as fh:
            json.dump(labels, fh)
    return idx


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_pose_head(pose_dir: str, out_path: str, epochs: int = 80) -> Optional[str]:
    """Train OrientationNet on the crops saved by ``save_pose_samples``.
    Returns the weights path, or None if training could not run."""
    if not _TORCH or cv2 is None or np is None:
        return None
    labels_path = os.path.join(pose_dir, "labels.json")
    if not os.path.exists(labels_path):
        return None
    with open(labels_path) as fh:
        labels = json.load(fh)
    xs, ys = [], []
    for rec in labels:
        crop = cv2.imread(os.path.join(pose_dir, rec["file"]), cv2.IMREAD_COLOR)
        t = _prep(crop)
        if t is None:
            continue
        th = math.radians(rec.get("angle", 0.0))
        ax = np.array(rec.get("axis", [0, 1, 0]), dtype="float32")
        n = float(np.linalg.norm(ax)) or 1.0
        ax = ax / n
        xs.append(t)
        ys.append([math.cos(th), math.sin(th), ax[0], ax[1], ax[2]])
    if len(xs) < 2:
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.tensor(np.stack(xs), dtype=torch.float32, device=device)
    Y = torch.tensor(np.array(ys), dtype=torch.float32, device=device)
    net = OrientationNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    net.train()
    for _ in range(max(1, epochs)):
        opt.zero_grad()
        pred = net(X)
        # angle term + axis term (both already unit-normalised inside forward)
        loss = lossf(pred[:, :2], Y[:, :2]) + lossf(pred[:, 2:], Y[:, 2:])
        loss.backward()
        opt.step()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(net.state_dict(), out_path)
    return out_path


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
class PosePredictor:
    """Loads a trained OrientationNet and predicts (angle_deg, axis) per crop."""

    def __init__(self, weights: str) -> None:
        self.ok = False
        self.net = None
        if not _TORCH or not weights or not os.path.exists(weights):
            return
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.net = OrientationNet().to(self.device)
            self.net.load_state_dict(torch.load(weights, map_location=self.device))
            self.net.eval()
            self.ok = True
        except Exception:
            self.ok = False

    def predict(self, crop_bgr) -> Optional[Tuple[float, list]]:
        if not self.ok:
            return None
        t = _prep(crop_bgr)
        if t is None:
            return None
        with torch.no_grad():
            x = torch.tensor(t[None], dtype=torch.float32, device=self.device)
            out = self.net(x)[0].cpu().numpy()
        angle = math.degrees(math.atan2(float(out[1]), float(out[0])))
        axis = [float(out[2]), float(out[3]), float(out[4])]
        n = math.sqrt(sum(a * a for a in axis)) or 1.0
        axis = [a / n for a in axis]
        return angle, axis
