"""Post an image to the running server and validate the reply.

Usage:  python scripts/test_detect.py [path/to/image.jpg]
Requires the server to be running (see Chapter 6).
"""
import sys
import httpx

url = "http://localhost:8000/detect"
path = sys.argv[1] if len(sys.argv) > 1 else "street.jpg"

with open(path, "rb") as f:
    files = {"file": (path, f, "image/jpeg")}
    r = httpx.post(url, files=files, timeout=30.0)

r.raise_for_status()
data = r.json()
assert "objects" in data and "image" in data, "bad response shape"
print(f"{len(data['objects'])} objects in "
      f"{data['timing_ms']} ms on a "
      f"{data['image']['width']}x{data['image']['height']} image")
for o in data["objects"][:5]:
    print(f"  - {o['cls']:<12} {o['confidence']:.2f}")
