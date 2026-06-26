"""Confirm YOLO26 loads and runs. Run: python scripts/verify_yolo.py"""
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
print("Task   :", model.task)
print("Classes:", len(model.names))

results = model("https://ultralytics.com/images/bus.jpg", verbose=False)
r = results[0]
print("Detections:", len(r.boxes))
for b in r.boxes:
    cls = model.names[int(b.cls)]
    conf = float(b.conf)
    print(f"  {cls:12s} {conf:.2f}")
