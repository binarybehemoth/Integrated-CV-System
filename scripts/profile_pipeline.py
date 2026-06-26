#!/usr/bin/env python3
"""Profile the perception pipeline, capability by capability.

The slowest capability, not the one you suspect, is where to optimise.
This wraps each capability's apply() with a timer and reports the mean
milliseconds per stage over a number of frames. See Chapter 32.

Usage:
    python scripts/profile_pipeline.py --frames 50
"""
from __future__ import annotations
import argparse
import statistics
import time
from collections import defaultdict


def profile_engine(engine, frames: list, warmup: int = 3) -> dict:
    """Run frames through the engine, timing each capability's apply().

    Returns {capability_name: {"mean_ms", "max_ms", "calls"}}.
    Assumes engine.capabilities is the ordered list of capabilities and
    each has a .name and .apply(world, frame) method.
    """
    timings: dict[str, list[float]] = defaultdict(list)

    for i, frame in enumerate(frames):
        world = engine.new_world(frame) if hasattr(engine, "new_world") else None
        for cap in engine.capabilities:
            t0 = time.perf_counter()
            world = cap.apply(world, frame)
            dt = (time.perf_counter() - t0) * 1000.0
            if i >= warmup:                # skip warmup frames
                timings[cap.name].append(dt)

    report = {}
    for name, xs in timings.items():
        report[name] = {
            "mean_ms": round(statistics.mean(xs), 2) if xs else 0.0,
            "max_ms": round(max(xs), 2) if xs else 0.0,
            "calls": len(xs),
        }
    return report


def print_report(report: dict) -> None:
    total = sum(v["mean_ms"] for v in report.values())
    width = max((len(k) for k in report), default=10)
    print(f"{'capability':<{width}}  {'mean ms':>8}  {'max ms':>8}  {'% total':>8}")
    for name, v in sorted(report.items(), key=lambda kv: -kv[1]["mean_ms"]):
        pct = 100.0 * v["mean_ms"] / total if total else 0.0
        print(f"{name:<{width}}  {v['mean_ms']:>8.2f}  {v['max_ms']:>8.2f}"
              f"  {pct:>7.1f}%")
    print(f"{'TOTAL':<{width}}  {total:>8.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Profile the pipeline.")
    ap.add_argument("--frames", type=int, default=50)
    ap.parse_args()
    print("Import this module and call profile_engine(engine, frames).")
