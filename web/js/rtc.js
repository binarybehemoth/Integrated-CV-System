// rtc.js — capture the camera and POST frames to /detect over HTTPS,
// then draw the returned world model. HTTP (not WebRTC) so it works
// through an ngrok tunnel on a phone.
(function () {
  "use strict";

  const video = document.getElementById("video");
  const canvas = document.getElementById("overlay");
  const startBtn = document.getElementById("start");
  const stopBtn = document.getElementById("stop");
  const fpsEl = document.getElementById("fps");

  const fpsSlider = document.getElementById("fpsSlider");
  const fpsValEl = document.getElementById("fpsval");
  function targetFps() {
    return fpsSlider ? parseInt(fpsSlider.value, 10) || 12 : 12;
  }

  // Resolution slider: the fraction of the native frame size that is encoded
  // and POSTed to the server (20/40/60/80/100%). Lower = less bandwidth/compute.
  const frame = document.getElementById("frame");
  const resSlider = document.getElementById("resSlider");
  const resValEl = document.getElementById("resval");
  function captureScale() {
    const v = resSlider ? parseInt(resSlider.value, 10) || 100 : 100;
    return Math.min(1, Math.max(0.05, v / 100));
  }

  let stream = null, raf = null;
  let lastFps = performance.now();

  function sizeCanvas() {
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
    // (#three is sized by the Three.js renderer in viz.js)
  }
  window.addEventListener("resize", sizeCanvas);

  // Lower the camera frame rate to cut transmitted bandwidth (helps a lot
  // over an ngrok tunnel on mobile). Adjustable live via the slider.
  function applyFrameRate() {
    if (fpsValEl) fpsValEl.textContent = targetFps() + " fps";
    if (!stream) return;
    const track = stream.getVideoTracks()[0];
    if (track && track.applyConstraints) {
      track.applyConstraints({ frameRate: { ideal: targetFps(), max: targetFps() } })
        .catch(() => {});
    }
  }
  if (fpsSlider) {
    fpsSlider.addEventListener("input", applyFrameRate);
    if (fpsValEl) fpsValEl.textContent = targetFps() + " fps";
  }
  if (resSlider) {
    const showRes = () => { if (resValEl) resValEl.textContent = resSlider.value + "%"; };
    resSlider.addEventListener("input", showRes);
    showRes();
  }

  // The overlay redraws every animation frame for smoothness; the fps
  // readout shows DETECTIONS per second (the rate frames round-trip to
  // the server), which is what actually matters over a slow link.
  let detCount = 0;
  function renderLoop() {
    // Paint the exact downscaled frame the server receives, so the live view
    // visibly reflects the resolution slider (blocky at 20%, sharp at 100%).
    if (frame && stream && video.videoWidth) {
      const s = captureScale();
      const dw = Math.max(1, Math.round(video.videoWidth * s));
      const dh = Math.max(1, Math.round(video.videoHeight * s));
      if (frame.width !== dw) frame.width = dw;
      if (frame.height !== dh) frame.height = dh;
      frame.getContext("2d").drawImage(video, 0, 0, dw, dh);
    }
    Overlay.draw(canvas);
    const now = performance.now();
    if (now - lastFps >= 1000) {
      fpsEl.textContent = detCount + " det/s";
      detCount = 0;
      lastFps = now;
    }
    raf = requestAnimationFrame(renderLoop);
  }

  // Capture the current video frame and POST it to /detect. This uses
  // plain HTTPS, so it works through an ngrok tunnel on a phone, unlike
  // WebRTC media (whose peer-to-peer UDP cannot traverse the tunnel).
  const capCanvas = document.createElement("canvas");
  let detecting = false, detectTimer = null;

  async function captureAndDetect() {
    if (!stream || !video.videoWidth) return;
    const s = captureScale();                  // 0.2 .. 1.0 from the slider
    capCanvas.width = Math.max(1, Math.round(video.videoWidth * s));
    capCanvas.height = Math.max(1, Math.round(video.videoHeight * s));
    capCanvas.getContext("2d").drawImage(video, 0, 0, capCanvas.width, capCanvas.height);
    const blob = await new Promise((res) => capCanvas.toBlob(res, "image/jpeg", 0.7));
    if (!blob) return;
    const fd = new FormData();
    fd.append("file", blob, "frame.jpg");      // FastAPI expects "file"
    // In YOLOE (open-vocabulary) mode, post the text prompt to /detect_open.
    // In two-phase cascade mode, post the (optional) narrowing prompt to
    // /detect_cascade. Otherwise run the full engine chain via /detect.
    let url = "/detect";
    const m = Overlay.getMode && Overlay.getMode();
    if (m === "yoloe") {
      const inp = document.getElementById("yoloePrompt");
      fd.append("prompt", (inp && inp.value) || "");
      url = "/detect_open";
    } else if (m === "cascade") {
      const inp = document.getElementById("yoloePrompt");
      fd.append("prompt", (inp && inp.value) || "");
      url = "/detect_cascade";
    }
    const r = await fetch(url, { method: "POST", body: fd });
    if (r.ok) { Overlay.setWorld(await r.json()); detCount++; }
  }

  // Self-scheduling loop, rate-limited to the FPS slider, never letting
  // requests overlap (so a slow link just lowers the rate, not piles up).
  async function detectLoop() {
    if (!stream) return;
    const t0 = performance.now();
    if (!detecting) {
      detecting = true;
      try { await captureAndDetect(); } catch (e) { /* transient */ }
      detecting = false;
    }
    const interval = 1000 / targetFps();
    detectTimer = setTimeout(detectLoop, Math.max(20, interval - (performance.now() - t0)));
  }

  async function start() {
    startBtn.disabled = true;
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "environment",
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: targetFps(), max: targetFps() },
      },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    sizeCanvas();
    applyFrameRate();

    stopBtn.disabled = false;
    renderLoop();                              // draw overlay continuously
    detectLoop();                              // POST frames to /detect
  }

  function stop() {
    if (raf) cancelAnimationFrame(raf);
    if (detectTimer) clearTimeout(detectTimer);
    detectTimer = null;
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = null;
    Overlay.setWorld(null);
    Overlay.draw(canvas);
    startBtn.disabled = false;
    stopBtn.disabled = true;
    fpsEl.textContent = "0 det/s";
  }

  startBtn.addEventListener("click", () =>
    start().catch((e) => {
      console.error(e);
      alert("Camera/connection error: " + e.message);
      startBtn.disabled = false;
    })
  );
  stopBtn.addEventListener("click", stop);
})();
