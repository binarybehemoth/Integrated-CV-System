"""End-to-end environment smoke test.
Run: python scripts/smoke_detect.py <image.jpg>
"""
import sys
import cv2
from ultralytics import YOLO


def main(image_path: str, out_path: str = "smoke_out.jpg") -> None:
    model = YOLO("yolo26n.pt")
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(image_path)

    results = model(frame, verbose=False)
    r = results[0]

    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = model.names[int(box.cls)]
        conf = float(box.conf)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(frame, f"{cls} {conf:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

    cv2.imwrite(out_path, frame)
    print(f"Wrote {out_path} with {len(r.boxes)} detections.")


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    main(img)
