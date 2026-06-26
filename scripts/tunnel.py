#!/usr/bin/env python3
"""Expose the local server over HTTPS with ngrok, for phone testing.

The browser only grants camera + WebRTC on a secure origin. ngrok gives
you a public https:// URL that tunnels to the local server, so you can
open the live demo on a phone. Combine with the FPS slider on the page
to keep the tunnelled bandwidth low.

Usage:
    # 1) start the server in one terminal:
    python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
    # 2) start the tunnel in another:
    python scripts/tunnel.py            # prints an https URL for your phone

Requires: pip install pyngrok   (and a free ngrok auth token)
Set the token once:  ngrok config add-authtoken <YOUR_TOKEN>
"""
from __future__ import annotations
import argparse
import time


def main() -> None:
    ap = argparse.ArgumentParser(description="ngrok HTTPS tunnel.")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    try:
        from pyngrok import ngrok
    except ImportError:
        raise SystemExit(
            "pyngrok is not installed. Run: pip install pyngrok\n"
            "Then set your token once: ngrok config add-authtoken <TOKEN>")

    tunnel = ngrok.connect(args.port, "http")
    url = tunnel.public_url.replace("http://", "https://")
    print("\n" + "=" * 60)
    print("  Open this on your phone (camera needs https):")
    print("   ", url)
    print("    studio:", url + "/studio.html")
    print("=" * 60)
    print("  Lower the FPS slider on the page to reduce bandwidth.")
    print("  Press Ctrl+C to close the tunnel.\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        ngrok.disconnect(tunnel.public_url)
        ngrok.kill()
        print("tunnel closed.")


if __name__ == "__main__":
    main()
