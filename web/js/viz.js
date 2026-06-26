// viz.js — drive the capability "tour": tap the stage to cycle through
// every capability display, keep a textual log, and render the 3D
// (WebGL) reconstruction of geometric primitives + voxels via Three.js.
(function (global) {
  "use strict";

  let modeIdx = 0;
  let flipZ = false;                          // some depth models invert near/far
  const modes = (global.Overlay && global.Overlay.MODES) || [];
  const stage = document.getElementById("stage");
  const three = document.getElementById("three");
  const modeLabel = document.getElementById("modeLabel");
  const logEl = document.getElementById("log");
  const flipBtn = document.getElementById("flipDepth");

  function applyMode() {
    const m = modes[modeIdx];
    if (!m) return;
    global.Overlay.setMode(m.id);
    if (modeLabel) modeLabel.textContent =
      `(${modeIdx + 1}/${modes.length}) ${m.label} \u2014 tap to change`;
    const is3d = m.id === "3d";
    if (three) three.style.display = is3d ? "block" : "none";
    const overlay = document.getElementById("overlay");
    if (overlay) overlay.style.display = is3d ? "none" : "block";
    if (flipBtn) flipBtn.style.display = is3d ? "block" : "none";
    const yp = document.getElementById("yoloePrompt");
    if (yp) {
      const wants = (m.id === "yoloe" || m.id === "cascade");
      yp.style.display = wants ? "block" : "none";
      if (m.id === "cascade")
        yp.placeholder = "optional: narrow phase-1 classes (blank = all 4585)";
      else if (m.id === "yoloe")
        yp.placeholder = "type classes e.g. zebra, traffic cone, skateboard";
    }
    if (is3d) ensureThree();
  }
  function next() { modeIdx = (modeIdx + 1) % modes.length; applyMode(); }

  if (stage) stage.addEventListener("click", next);
  // Typing in / tapping the YOLOE prompt must not advance the capability tour.
  const ypEl = document.getElementById("yoloePrompt");
  if (ypEl) {
    ["click", "mousedown", "touchstart", "keydown"].forEach((ev) =>
      ypEl.addEventListener(ev, (e) => e.stopPropagation()));
  }
  if (flipBtn) flipBtn.addEventListener("click", (e) => {
    e.stopPropagation();                      // don't advance the mode
    flipZ = !flipZ;
    flipBtn.textContent = flipZ ? "Depth: flipped" : "Flip depth";
  });
  applyMode();

  // ---- textual log, refreshed a few times a second --------------------
  setInterval(() => {
    if (!logEl || !global.Overlay) return;
    const lines = global.Overlay.describe();
    const UNIT = 12;                       // px per indent level (2 spaces = 1 level)
    logEl.innerHTML = "";
    for (const ln of lines) {
      const lead = (ln.match(/^ */)[0] || "").length;
      const depth = Math.floor(lead / 2);
      const div = document.createElement("div");
      div.textContent = ln.slice(lead) || "\u00a0";   // nbsp keeps blank lines
      // padding sets the line's level; the negative text-indent makes wrapped
      // continuation hang one step further right, never back to the margin.
      div.style.paddingLeft = (depth * UNIT + 18) + "px";
      div.style.textIndent = "-18px";
      div.style.whiteSpace = "normal";
      logEl.appendChild(div);
    }
  }, 400);

  // ---- 3D reconstruction (Three.js), static, matching the scene -------
  let scene = null, camera = null, renderer = null, group = null;

  function ensureThree() {
    if (renderer || !three || typeof THREE === "undefined") return;
    const w = three.clientWidth || 360, h = three.clientHeight || 480;
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0e1116);
    camera = new THREE.PerspectiveCamera(55, w / h, 0.1, 100);
    camera.position.set(0, 0, 6);
    camera.lookAt(0, 0, -2);                 // face the scene head-on
    renderer = new THREE.WebGLRenderer({ canvas: three, antialias: true });
    renderer.setSize(w, h, false);
    scene.add(new THREE.AmbientLight(0xffffff, 0.75));
    const dir = new THREE.DirectionalLight(0xffffff, 0.6);
    dir.position.set(2, 4, 5); scene.add(dir);
    group = new THREE.Group(); scene.add(group);
    const grid = new THREE.GridHelper(8, 16, 0x335577, 0x223344);
    grid.position.y = -1.6; scene.add(grid);
    animate();
  }

  function hashColor(name) {
    let h = 0;
    for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
    return new THREE.Color(`hsl(${h % 360}, 70%, 60%)`);
  }

  function rebuild() {
    if (!group) return;
    while (group.children.length) group.remove(group.children[0]);
    const world = global.Overlay.getWorld();
    if (!world || !world.image) return;
    const iw = world.image.width, ih = world.image.height;
    const objs = world.objects || [];

    // Depth from MiDaS/DPT is disparity-like: LARGER = NEARER. Normalize
    // across the visible objects so the nearest sits closest to the
    // camera. If depth is off (it is opt-in), fall back to box size as a
    // proxy -- a bigger box usually means a nearer object.
    let dmin = Infinity, dmax = -Infinity, hasDepth = false;
    let amin = Infinity, amax = -Infinity;
    for (const o of objs) {
      const d = o.properties && o.properties.depth;
      if (d != null) { hasDepth = true; dmin = Math.min(dmin, d); dmax = Math.max(dmax, d); }
      const a = ((o.box.x2 - o.box.x1) / iw) * ((o.box.y2 - o.box.y1) / ih);
      amin = Math.min(amin, a); amax = Math.max(amax, a);
    }

    const K = 4;                            // scene units across the frame

    // Build one primitive mesh (+ edges) for a primitive-bearing object
    // or part, placed by its box at depth z.
    function addPrimitive(node, z, fallbackColor) {
      const pp = node.properties || {};
      const wN = (node.box.x2 - node.box.x1) / iw;
      const hN = (node.box.y2 - node.box.y1) / ih;
      const cx = (node.box.x1 + node.box.x2) / 2 / iw - 0.5;
      const cy = -((node.box.y1 + node.box.y2) / 2 / ih - 0.5);
      const shape = (pp.primitive && pp.primitive.shape) || "box";
      let geo;
      if (shape === "sphere") {
        geo = new THREE.SphereGeometry(Math.max(0.03, (wN + hN) / 4 * K), 20, 14);
      } else if (shape === "cylinder") {
        const r = Math.max(0.03, (wN / 2) * K);
        geo = new THREE.CylinderGeometry(r, r, Math.max(0.06, hN * K), 18);
      } else {
        geo = new THREE.BoxGeometry(Math.max(0.04, wN * K), Math.max(0.04, hN * K),
          Math.max(0.04, Math.min(wN, hN) * K));
      }
      const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
        color: hashColor(node.cls || fallbackColor), transparent: true,
        opacity: 0.82, roughness: 0.6, metalness: 0.1,
      }));
      mesh.position.set(cx * K, cy * K, z);
      // Apply the annotated rotation: `rotation` degrees about the axis vector
      // (axisX, axisY, axisZ). Defaults to no rotation about a Y-up axis.
      const prim = pp.primitive || {};
      const deg = parseFloat(prim.rotation) || 0;
      if (deg) {
        const ax = new THREE.Vector3(parseFloat(prim.axisX) || 0,
          prim.axisY == null ? 1 : (parseFloat(prim.axisY) || 0),
          parseFloat(prim.axisZ) || 0);
        if (ax.lengthSq() > 1e-9) {
          ax.normalize();
          mesh.setRotationFromAxisAngle(ax, deg * Math.PI / 180);
        }
      }
      group.add(mesh);
      const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo),
        new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.28 }));
      edges.position.copy(mesh.position);
      edges.quaternion.copy(mesh.quaternion);    // match the mesh rotation
      group.add(edges);
    }

    for (const o of objs) {
      const p = o.properties || {};
      const wN = (o.box.x2 - o.box.x1) / iw;
      const hN = (o.box.y2 - o.box.y1) / ih;
      // t = 1 for the nearest object, 0 for the farthest (flippable, since
      // some depth models invert the convention).
      let t = 0.5;
      if (hasDepth && p.depth != null && dmax > dmin) {
        t = (p.depth - dmin) / (dmax - dmin);
      } else if (amax > amin) {
        t = (wN * hN - amin) / (amax - amin);   // bigger box = nearer
      }
      if (flipZ) t = 1 - t;
      const z = -3 + t * 4;                 // near -> z=1 (front), far -> z=-3

      // If the object has parts (with primitives), reconstruct it as a
      // combination of part primitives; otherwise draw one primitive.
      const parts = (o.parts || []).filter((pt) => pt.properties && pt.properties.primitive);
      if (parts.length) {
        for (const part of parts) addPrimitive(part, z, o.cls);
      } else {
        addPrimitive(o, z, o.cls);
      }
    }

    // Voxels, if the world model carries an occupancy list.
    const voxels = world.voxels || (world.voxel && world.voxel.cells);
    if (Array.isArray(voxels)) {
      const g = new THREE.BoxGeometry(0.12, 0.12, 0.12);
      const m = new THREE.MeshStandardMaterial({ color: 0x66ccff,
        transparent: true, opacity: 0.5 });
      for (const v of voxels.slice(0, 4000)) {
        const cube = new THREE.Mesh(g, m);
        cube.position.set((v[0] - 8) * 0.14, (v[1] - 8) * 0.14, (v[2] - 8) * 0.14);
        group.add(cube);
      }
    }
  }

  function animate() {
    if (!renderer) return;
    requestAnimationFrame(animate);
    if (modes[modeIdx] && modes[modeIdx].id === "3d") {
      rebuild();                             // mirror current scene; no rotation
      renderer.render(scene, camera);
    }
  }

  global.Viz = { next, applyMode };
})(window);
