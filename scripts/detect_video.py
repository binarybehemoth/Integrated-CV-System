"""Run the engine over a video file and write an annotated copy.
Run: python scripts/detect_video.py <input.mp4> [out.mp4]
"""
import sys, time, cv2

from server.engine.engine import Engine
from server.engine.detector import DetectorCapability
from server.engine.draw import draw_world


def main(src: str, dst: str = "out.mp4") -> None:
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {src}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))

    engine = Engine([DetectorCapability()])
    engine.setup()

    n, t0 = 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        world = engine.process(frame)
        writer.write(draw_world(frame, world))
        n += 1

    cap.release(); writer.release()
    dt = time.time() - t0
    print(f"{n} frames in {dt:.1f}s  ->  {n / dt:.1f} FPS, wrote {dst}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "input.mp4")
