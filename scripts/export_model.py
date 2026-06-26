#!/usr/bin/env python3
"""Export a YOLO26 model to a faster inference format.

Ultralytics can export to ONNX, TensorRT, OpenVINO, CoreML and more.
ONNX is the portable baseline; TensorRT gives the largest GPU speedup.
Half precision (FP16) roughly halves memory and boosts throughput with
negligible accuracy loss. See Chapter 30.

Usage:
    python scripts/export_model.py best.pt --format onnx --half
    python scripts/export_model.py best.pt --format engine --half --imgsz 640
"""
from __future__ import annotations
import argparse


def export(weights: str, fmt: str = "onnx", half: bool = False,
           imgsz: int = 640, dynamic: bool = False) -> str:
    """Export ``weights`` to ``fmt`` and return the exported path."""
    from ultralytics import YOLO

    model = YOLO(weights)
    path = model.export(
        format=fmt,        # onnx | engine (TensorRT) | openvino | coreml
        half=half,         # FP16
        imgsz=imgsz,
        dynamic=dynamic,   # allow variable input size (ONNX)
    )
    return str(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a YOLO26 model.")
    ap.add_argument("weights", help="Path to .pt weights")
    ap.add_argument("--format", default="onnx",
                    choices=["onnx", "engine", "openvino", "coreml",
                             "torchscript"])
    ap.add_argument("--half", action="store_true", help="FP16 export")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--dynamic", action="store_true")
    args = ap.parse_args()

    out = export(args.weights, args.format, args.half, args.imgsz,
                 args.dynamic)
    print(f"Exported to: {out}")


if __name__ == "__main__":
    main()
