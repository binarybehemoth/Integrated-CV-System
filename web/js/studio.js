// studio.js — the annotation studio: draw boxes, sub-parts, and
// keypoints; manage a class hierarchy and per-object properties;
// persist to IndexedDB; export YOLO labels; trigger training.
(function () {
  "use strict";

  // ---- State -------------------------------------------------------
  const state = {
    images: [],            // {id, name, dataUrl, w, h}
    current: null,         // current image id
    classes: {},           // name -> parent (parent "" means root)
    annos: {},             // imageId -> [object]
    tool: "box",
    activeClass: null,
    selected: null,        // selected object id
    selectedPart: null,    // selected part index within the selected object
    kpIndex: 0,
  };
  const COCO_KP = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
  ];

  // Seed a small default hierarchy so the tool is usable immediately.
  Object.assign(state.classes, {
    object: "", vehicle: "object", animal: "object", person: "object",
    car: "vehicle", truck: "vehicle", dog: "animal", cat: "animal",
  });
  state.activeClass = null;

  // ---- Elements ----------------------------------------------------
  const $ = (id) => document.getElementById(id);
  const canvas = $("canvas");
  const ctx = canvas.getContext("2d");
  let img = new Image();

  // ---- Persistence (IndexedDB via db.js Store) ---------------------
  const store = new window.Store("cv-studio", ["images", "annos", "meta"]);

  async function save() {
    await store.put("meta", "classes", state.classes);
    for (const im of state.images) await store.put("images", im.id, im);
    for (const id in state.annos) await store.put("annos", id, state.annos[id]);
    $("saved").textContent = "IndexedDB: saved " + new Date().toLocaleTimeString();
  }

  async function load() {
    const cls = await store.get("meta", "classes");
    if (cls) state.classes = cls;
    state.fixed = new Set((await store.get("meta", "fixed")) || []);
    state.trained = new Set((await store.get("meta", "trained")) || []);
    state.kpSchemas = (await store.get("meta", "kpschemas")) || {};
    state.kpFor = (await store.get("meta", "kpfor")) || {};
    state.importedSrc = (await store.get("meta", "importedSrc")) || {};

    // Prepopulate (first run) or migrate (on a version bump) the FIXED
    // level-1 vocabulary: a three-level tree (supercategory -> subcategory ->
    // the 4585 classes) plus the "Named Persons" category and keypoint
    // skeletons. On a version bump we re-seed the fixed tree but KEEP any
    // custom level-2 classes the user has added.
    const ver = (await store.get("meta", "vocab_version")) || 0;
    try {
      const r = await fetch("/data/yoloe_vocab.json");
      if (r.ok) {
        const v = await r.json();
        const fileVer = v.version || 1;
        const newHier = v.hierarchy || {};
        // Preserve user-added custom classes (anything not in the fixed tree).
        const custom = {};
        for (const [name, parent] of Object.entries(state.classes || {})) {
          if (!(name in newHier)) custom[name] = parent;
        }
        // ALWAYS restore the full fixed tree (+ keep custom), so the hierarchy
        // self-heals even if a previous state was incomplete (e.g. only the
        // tiny default seed survived an earlier offline load).
        state.classes = Object.assign({}, newHier, custom);
        state.fixed = new Set(v.fixed_classes || []);
        state.kpSchemas = v.keypoint_schemas || {};
        state.kpFor = v.keypoints || {};
        state.namedPersons = v.named_persons_category || "Named Persons";
        if (fileVer > ver) state.activeClass = null;
        await store.put("meta", "classes", state.classes);
        await store.put("meta", "fixed", [...state.fixed]);
        await store.put("meta", "kpschemas", state.kpSchemas);
        await store.put("meta", "kpfor", state.kpFor);
        await store.put("meta", "vocab_version", fileVer);
        if ($("saved")) $("saved").textContent =
          "Loaded " + state.fixed.size + " classes in " +
          Object.keys(newHier).length + " nodes";
      }
    } catch (e) { /* offline: keep what we have */ }

    const ims = await store.all("images");
    if (ims && ims.length) {
      state.images = ims;
      for (const im of ims) {
        const a = await store.get("annos", im.id);
        state.annos[im.id] = a || [];
      }
    }
  }

  // ---- Geometry helpers (zoom + pan aware) -------------------------
  // view.zoom multiplies the fit scale; view.panX/Y offset the image in
  // canvas pixels. All screen<->image conversions go through these, so
  // boxes are always stored in image pixels regardless of zoom/pan.
  const view = { zoom: 1, panX: 0, panY: 0 };
  function baseScale() {
    return { sx: canvas.width / (img.naturalWidth || canvas.width),
             sy: canvas.height / (img.naturalHeight || canvas.height) };
  }
  function dispScale() {
    const b = baseScale();
    return { sx: b.sx * view.zoom, sy: b.sy * view.zoom };
  }
  function toImg(px, py) {                 // canvas px -> image px
    const s = dispScale();
    return [(px - view.panX) / s.sx, (py - view.panY) / s.sy];
  }
  function toCanvas(ix, iy) {              // image px -> canvas px
    const s = dispScale();
    return [ix * s.sx + view.panX, iy * s.sy + view.panY];
  }
  function resetView() { view.zoom = 1; view.panX = 0; view.panY = 0; }
  function zoomAt(px, py, factor) {        // zoom keeping (px,py) fixed
    const [ix, iy] = toImg(px, py);
    view.zoom = Math.max(0.2, Math.min(12, view.zoom * factor));
    const s = dispScale();
    view.panX = px - ix * s.sx;
    view.panY = py - iy * s.sy;
    render();
    updateZoomLabel();
  }
  function updateZoomLabel() {
    const el = document.getElementById("zoomLabel");
    if (el) el.textContent = Math.round(view.zoom * 100) + "%";
  }
  function curAnnos() { return state.annos[state.current] || []; }
  function nextId() {
    const a = curAnnos();
    return a.reduce((m, o) => Math.max(m, o.id), -1) + 1;
  }
  function selectedObj() {
    return curAnnos().find((o) => o.id === state.selected) || null;
  }

  // ---- Drawing -----------------------------------------------------
  function colorFor(name) {
    let h = 0;
    for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
    return `hsl(${h % 360}, 70%, 55%)`;
  }
  function render() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const s = dispScale();
    if (img.src) {
      ctx.drawImage(img, view.panX, view.panY,
        (img.naturalWidth || canvas.width) * s.sx,
        (img.naturalHeight || canvas.height) * s.sy);
    }
    for (const o of curAnnos()) {
      const col = colorFor(o.cls);
      const [x, y] = toCanvas(o.box[0], o.box[1]);
      const [x2, y2] = toCanvas(o.box[2], o.box[3]);
      const w = x2 - x, h = y2 - y;
      ctx.lineWidth = o.id === state.selected ? 3 : 2;
      ctx.strokeStyle = col;
      ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = col;
      ctx.font = "600 13px system-ui";
      ctx.fillText(o.cls, x + 3, Math.max(12, y - 5));
      // sub-parts
      ctx.setLineDash([4, 3]); ctx.lineWidth = 1;
      for (const p of o.parts || []) {
        const [px0, py0] = toCanvas(p.box[0], p.box[1]);
        const [px1, py1] = toCanvas(p.box[2], p.box[3]);
        ctx.strokeRect(px0, py0, px1 - px0, py1 - py0);
      }
      ctx.setLineDash([]);
      // keypoints
      ctx.fillStyle = "#ffd400";
      for (const k of o.keypoints || []) {
        const [kx, ky] = toCanvas(k.x, k.y);
        ctx.beginPath();
        ctx.arc(kx, ky, 4, 0, 6.283); ctx.fill();
      }
    }
    if (drag) {
      ctx.strokeStyle = "#fff"; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.5;
      ctx.strokeRect(drag.x0, drag.y0, drag.x1 - drag.x0, drag.y1 - drag.y0);
      ctx.setLineDash([]);
    }
    if (typeof renderPrimOverlay === "function") renderPrimOverlay();
  }

  // ---- Mouse: draw boxes / place keypoints / select / pan / zoom ----
  let drag = null;
  let pan = null;
  canvas.addEventListener("mousedown", (e) => {
    if (!state.current) return;
    const r = canvas.getBoundingClientRect();
    const px = e.clientX - r.left, py = e.clientY - r.top;
    // Pan with the middle button, Shift+drag, or the Pan tool.
    if (e.button === 1 || e.shiftKey || state.tool === "pan") {
      e.preventDefault();
      pan = { x: px, y: py, panX: view.panX, panY: view.panY };
      return;
    }
    if (state.tool === "keypoint") { placeKeypoint(px, py); return; }
    if (state.tool === "select") { pickObject(px, py); return; }
    drag = { x0: px, y0: py, x1: px, y1: py };
  });
  canvas.addEventListener("mousemove", (e) => {
    const r = canvas.getBoundingClientRect();
    const px = e.clientX - r.left, py = e.clientY - r.top;
    const co = $("coords");
    if (co) {
      const im = state.images.find((i) => i.id === state.current);
      if (im) {
        const [ix, iy] = toImg(px, py);
        co.textContent = "x: " + Math.round(ix) + "  y: " + Math.round(iy);
      } else { co.textContent = ""; }
    }
    if (pan) {
      view.panX = pan.panX + (px - pan.x);
      view.panY = pan.panY + (py - pan.y);
      render();
      return;
    }
    if (!drag) return;
    drag.x1 = px; drag.y1 = py;
    render();
  });
  canvas.addEventListener("mouseup", () => {
    if (pan) { pan = null; return; }
    if (!drag) return;
    const [ix0, iy0] = toImg(Math.min(drag.x0, drag.x1), Math.min(drag.y0, drag.y1));
    const [ix1, iy1] = toImg(Math.max(drag.x0, drag.x1), Math.max(drag.y0, drag.y1));
    const box = [ix0, iy0, ix1, iy1];
    if (ix1 - ix0 > 3 && iy1 - iy0 > 3) {
      if (state.tool === "box") {
        curAnnos().push({ id: nextId(), cls: state.activeClass, box: box,
          parts: [], keypoints: [], properties: {} });
        state.annos[state.current] = curAnnos();
      } else if (state.tool === "part") {
        const o = selectedObj() || lastObject();
        if (o) {
          o.parts.push({ cls: "part" + (o.parts.length + 1), box: box });
          state.selectedPart = o.parts.length - 1;
        }
      }
    }
    drag = null; refresh();
  });
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("wheel", (e) => {
    if (!state.current) return;
    e.preventDefault();
    const r = canvas.getBoundingClientRect();
    zoomAt(e.clientX - r.left, e.clientY - r.top,
           e.deltaY < 0 ? 1.12 : 1 / 1.12);
  }, { passive: false });

  function placeKeypoint(px, py) {
    const o = selectedObj() || lastObject();
    if (!o) return;
    const [ix, iy] = toImg(px, py);
    const name = COCO_KP[state.kpIndex % COCO_KP.length];
    o.keypoints.push({ name: name, x: ix, y: iy, visible: true });
    state.kpIndex++; refresh();
  }
  function pickObject(px, py) {
    const [ix, iy] = toImg(px, py);
    const hit = curAnnos().filter((o) =>
      ix >= o.box[0] && ix <= o.box[2] && iy >= o.box[1] && iy <= o.box[3]);
    state.selected = hit.length ? hit[hit.length - 1].id : null;
    refresh();
  }
  function lastObject() {
    const a = curAnnos();
    return a.length ? a[a.length - 1] : null;
  }

  // ---- Class hierarchy UI (clickable, expand/collapse) -------------
  function renderClasses() {
    const tree = $("classTree"); if (!tree) return;
    const lab = $("activeClassLabel");
    if (lab) lab.textContent = state.activeClass || "\u2014";
    const top = $("activeClassName");
    if (top) top.textContent = state.activeClass || "none selected";
    if (!state.expanded) state.expanded = new Set();

    // Build parent -> children ONCE (O(n)). The old code filtered all 4,679
    // classes per node (O(n^2)), which is what made expand/collapse lag.
    const byParent = new Map();
    for (const [name, parent] of Object.entries(state.classes)) {
      const k = parent || "";
      let arr = byParent.get(k);
      if (!arr) byParent.set(k, (arr = []));
      arr.push(name);
    }
    for (const arr of byParent.values()) arr.sort();

    // Render only roots + expanded branches into one fragment (one reflow).
    const frag = document.createDocumentFragment();
    const build = (name, depth) => {
      const kids = byParent.get(name) || [];
      const li = document.createElement("li");
      li.className = "treeitem" + (name === state.activeClass ? " active" : "");
      li.style.paddingLeft = (depth * 14 + 4) + "px";
      const open = state.expanded.has(name);
      const tog = document.createElement("span");
      tog.className = "tog";
      tog.textContent = kids.length ? (open ? "\u25BC" : "\u25B6") : "\u2022";
      tog.style.cursor = kids.length ? "pointer" : "default";
      if (kids.length) tog.onclick = (e) => {
        e.stopPropagation();
        if (open) state.expanded.delete(name); else state.expanded.add(name);
        renderClasses();
      };
      const lbl = document.createElement("span");
      lbl.className = "lbl";
      lbl.textContent = " " + name + (kids.length ? "  (" + kids.length + ")" : "");
      const fixed = state.fixed && state.fixed.has(name);
      const isLeaf = kids.length === 0;
      const custom = isLeaf && !fixed;          // user-added leaf
      if (fixed) {
        lbl.style.fontWeight = "bold";          // YOLOE level-1 vocabulary
      } else if (custom) {
        const trained = state.trained && state.trained.has(name);
        if (trained) lbl.style.textDecoration = "underline";  // has a model
        else lbl.style.fontStyle = "italic";                  // untrained
      }
      lbl.onclick = () => {
        state.activeClass = name;
        // selecting ANY class (fixed or custom) clears the list and loads
        // only the images annotated with that class (or its descendants).
        if (typeof filterImagesFor === "function") filterImagesFor(name);
        renderClasses();
      };
      li.appendChild(tog); li.appendChild(lbl);
      if (custom) {
        const del = document.createElement("span");
        del.textContent = " \u00d7";
        del.title = "remove this custom class";
        del.style.cssText = "cursor:pointer;color:#c0653a;margin-left:6px";
        del.onclick = (e) => {
          e.stopPropagation();
          if (!confirm('Remove custom class "' + name + '"?')) return;
          const parent = state.classes[name];
          delete state.classes[name];
          if (state.trained) state.trained.delete(name);
          if (state.activeClass === name) state.activeClass = parent || null;
          renderClasses(); save();
        };
        li.appendChild(del);
      }
      frag.appendChild(li);
      if (kids.length && open) for (const c of kids) build(c, depth + 1);
    };
    for (const r of (byParent.get("") || [])) build(r, 0);
    tree.innerHTML = "";
    tree.appendChild(frag);
  }

  // Expand a node's ancestor chain so a (newly added / selected) class is
  // visible without forcing the whole 4,679-node tree open.
  function expandTo(name) {
    state.expanded = state.expanded || new Set();
    let cur = state.classes[name];
    let hops = 0;
    while (cur && hops < 8) { state.expanded.add(cur); cur = state.classes[cur]; hops++; }
  }

  // ---- Class search: quick find + select among all 4,679 classes --------
  if ($("classSearch")) {
    const box = $("classSearch"), results = $("searchResults");
    const pick = (nm) => {
      state.activeClass = nm;
      expandTo(nm);
      box.value = ""; results.innerHTML = "";
      renderClasses();
      const act = document.querySelector("#classTree .treeitem.active");
      if (act && act.scrollIntoView) act.scrollIntoView({ block: "center" });
    };
    const runSearch = () => {
      const q = box.value.trim().toLowerCase();
      results.innerHTML = "";
      if (!q) return;
      const starts = [], has = [];
      for (const nm of Object.keys(state.classes)) {
        const low = nm.toLowerCase();
        if (low === q || low.startsWith(q)) starts.push(nm);
        else if (low.includes(q)) has.push(nm);
      }
      starts.sort(); has.sort();
      const hits = starts.concat(has).slice(0, 40);
      for (const nm of hits) {
        const li = document.createElement("li");
        li.className = "treeitem";
        const parent = state.classes[nm];
        const lbl = document.createElement("span");
        lbl.className = "lbl"; lbl.style.cursor = "pointer";
        lbl.textContent = nm + (parent ? "  \u2014 " + parent : "");
        lbl.onclick = () => pick(nm);
        li.appendChild(lbl); results.appendChild(li);
      }
      if (!hits.length) {
        const li = document.createElement("li");
        li.className = "hint"; li.textContent = "no match";
        results.appendChild(li);
      }
    };
    box.addEventListener("input", runSearch);
    box.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        const first = results.querySelector(".lbl");
        if (first) first.click();
      }
    });
  }

  // ---- Object list + inspector -------------------------------------
  function renderObjects() {
    const ul = $("objList"); ul.innerHTML = "";
    curAnnos().forEach((o) => {
      const li = document.createElement("li");
      li.className = o.id === state.selected ? "sel" : "";
      li.textContent = `#${o.id} ${o.cls}` +
        (o.parts.length ? ` \u00b7 ${o.parts.length}p` : "") +
        (o.keypoints.length ? ` \u00b7 ${o.keypoints.length}kp` : "");
      li.onclick = () => { state.selected = o.id; state.selectedPart = null; refresh(); };
      ul.appendChild(li);
    });
    const o = selectedObj();
    $("inspector").style.display = o ? "block" : "none";
    if (!o) return;
    $("selInfo").textContent =
      `#${o.id} ${o.cls} \u2014 ancestors: ${ancestors(o.cls).join(" \u2192 ") || "(root)"}`;
    renderPartList(o);
    renderKpList(o);
    renderPropList(o);
    renderPartInspector(o);
  }

  // Replace a label span with an input for in-place renaming.
  function renameInline(li, span, current, onSave) {
    const inp = document.createElement("input");
    inp.className = "rename"; inp.value = current;
    li.replaceChild(inp, span); inp.focus(); inp.select();
    let done = false;
    const commit = (ok) => {
      if (done) return; done = true;
      const v = inp.value.trim();
      if (ok && v) onSave(v); else renderObjects();
    };
    inp.onkeydown = (e) => {
      if (e.key === "Enter") commit(true);
      else if (e.key === "Escape") commit(false);
    };
    inp.onblur = () => commit(true);
  }

  function editRow(name, onPick, onRename, onDelete, selected) {
    const li = document.createElement("li");
    if (selected) li.className = "sel";
    const nm = document.createElement("span");
    nm.className = "nm"; nm.textContent = name;
    if (onPick) nm.onclick = onPick; else nm.style.cursor = "default";
    const ren = document.createElement("button");
    ren.textContent = "\u270e"; ren.title = "Rename";
    ren.onclick = (e) => { e.stopPropagation(); renameInline(li, nm, name.split(" ")[0], onRename); };
    const del = document.createElement("button");
    del.className = "del"; del.textContent = "\u00d7"; del.title = "Delete";
    del.onclick = (e) => { e.stopPropagation(); onDelete(); };
    li.appendChild(nm); li.appendChild(ren); li.appendChild(del);
    return li;
  }

  function renderPartList(o) {
    $("partCount").textContent = `(${o.parts.length})`;
    const ul = $("partList"); ul.innerHTML = "";
    o.parts.forEach((part, i) => {
      const label = part.cls + (part.primitive ? ` [${part.primitive.shape}]` : "");
      ul.appendChild(editRow(label,
        () => { state.selectedPart = i; renderObjects(); },
        (v) => { part.cls = v; save(); renderObjects(); },
        () => { o.parts.splice(i, 1); if (state.selectedPart === i) state.selectedPart = null; save(); refresh(); },
        i === state.selectedPart));
    });
  }

  function renderKpList(o) {
    $("kpCount").textContent = `(${o.keypoints.length})`;
    const ul = $("kpList"); ul.innerHTML = "";
    o.keypoints.forEach((kp, i) => {
      const label = `${kp.name} (${Math.round(kp.x)},${Math.round(kp.y)})`;
      ul.appendChild(editRow(label, null,
        (v) => { kp.name = v; save(); renderObjects(); },
        () => { o.keypoints.splice(i, 1); save(); refresh(); }, false));
    });
  }

  function renderPropList(o) {
    const ul = $("propList"); ul.innerHTML = "";
    Object.entries(o.properties).forEach(([k, v]) => {
      ul.appendChild(editRow(`${k}: ${v}`, null,
        (nv) => { o.properties[k] = nv; save(); renderObjects(); },
        () => { delete o.properties[k]; save(); renderObjects(); }, false));
    });
  }

  let overlayPart = null;                  // part shown in the WebGL overlay
  let primRenderer, primScene, primCam, primMesh, primEdges;

  function ensurePrimThree() {
    if (primRenderer || !window.THREE) return;
    const cv = $("primOverlay"); if (!cv) return;
    primRenderer = new THREE.WebGLRenderer({ canvas: cv, alpha: true, antialias: true });
    primRenderer.setPixelRatio(1);             // keep buffer == logical px
    primRenderer.setClearColor(0x000000, 0);   // fully transparent background
    primScene = new THREE.Scene();
    primCam = new THREE.OrthographicCamera(0, cv.width, 0, cv.height, -4000, 4000);
    primCam.position.z = 1000;
    primScene.add(new THREE.AmbientLight(0xffffff, 0.9));
    const d = new THREE.DirectionalLight(0xffffff, 0.5);
    d.position.set(0.5, 0.7, 1); primScene.add(d);
  }

  // Render the active part's primitive (with its rotation) as a translucent
  // 3D widget pinned over its box on the image, so the rotation is visible.
  function renderPrimOverlay() {
    const cv = $("primOverlay"), main = $("canvas");
    if (!cv || !main) return;
    const part = overlayPart;
    if (!part || !part.primitive || !window.THREE) { cv.style.display = "none"; return; }
    ensurePrimThree(); if (!primRenderer) return;
    // Buffer must equal the main canvas buffer; the ortho camera spans it 1:1.
    if (cv.width !== main.width || cv.height !== main.height) {
      cv.width = main.width; cv.height = main.height;
    }
    if (primCam) {
      primCam.right = cv.width; primCam.bottom = cv.height;
      primCam.updateProjectionMatrix();
    }
    primRenderer.setSize(cv.width, cv.height, false);   // viewport == buffer
    // Display size + position must match the main canvas exactly so the
    // pixel-space mesh lands on the box. Mirror the main canvas's rendered box.
    const rect = main.getBoundingClientRect();
    cv.style.width = rect.width + "px";
    cv.style.height = rect.height + "px";
    cv.style.display = "block";
    if (primMesh) { primScene.remove(primMesh); primMesh.geometry.dispose(); primMesh = null; }
    if (primEdges) { primScene.remove(primEdges); primEdges.geometry.dispose(); primEdges = null; }
    const [bx1, by1, bx2, by2] = part.box;
    const [cx1, cy1] = toCanvas(bx1, by1), [cx2, cy2] = toCanvas(bx2, by2);
    const w = Math.abs(cx2 - cx1), h = Math.abs(cy2 - cy1);
    const cx = (cx1 + cx2) / 2, cy = (cy1 + cy2) / 2;
    const prim = part.primitive, shape = prim.shape || "box";
    let geo;
    if (shape === "sphere") geo = new THREE.SphereGeometry(Math.max(4, (w + h) / 4), 24, 16);
    else if (shape === "cylinder") geo = new THREE.CylinderGeometry(Math.max(4, w / 2), Math.max(4, w / 2), Math.max(8, h), 24);
    else geo = new THREE.BoxGeometry(Math.max(6, w), Math.max(6, h), Math.max(6, Math.min(w, h)));
    primMesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
      color: 0x37c4ff, transparent: true, opacity: 0.28, roughness: 0.5,
      metalness: 0.1, depthWrite: false }));
    primMesh.position.set(cx, cy, 0);
    const deg = parseFloat(prim.rotation) || 0;
    if (deg) {
      const ax = new THREE.Vector3(parseFloat(prim.axisX) || 0,
        prim.axisY == null ? 1 : (parseFloat(prim.axisY) || 0), parseFloat(prim.axisZ) || 0);
      if (ax.lengthSq() > 1e-9) { ax.normalize(); primMesh.setRotationFromAxisAngle(ax, deg * Math.PI / 180); }
    }
    primScene.add(primMesh);
    primEdges = new THREE.LineSegments(new THREE.EdgesGeometry(geo),
      new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.55 }));
    primEdges.position.copy(primMesh.position);
    primEdges.quaternion.copy(primMesh.quaternion);
    primScene.add(primEdges);
    primRenderer.render(primScene, primCam);
  }

  function renderPartInspector(o) {
    const insp = $("partInspector");
    const part = (state.selectedPart != null) ? o.parts[state.selectedPart] : null;
    overlayPart = part;                    // drive the WebGL primitive overlay
    if (!part) { insp.style.display = "none"; renderPrimOverlay(); return; }
    insp.style.display = "block";
    $("partInspName").textContent = part.cls;
    $("partPrim").value = part.primitive ? part.primitive.shape : "";
    renderPrimDims(part);
    renderPrimOverlay();
  }

  function defaultDims(shape, part) {
    const bw = Math.round(part.box[2] - part.box[0]);
    const bh = Math.round(part.box[3] - part.box[1]);
    if (shape === "sphere") return { radius: Math.round((bw + bh) / 4) };
    if (shape === "cylinder") return { radius: Math.round(bw / 2), height: bh };
    return { width: bw, height: bh, depth: bw };
  }

  function renderPrimDims(part) {
    const wrap = $("primDims"); wrap.innerHTML = "";
    if (!part.primitive) return;
    const rotKeys = ["rotation", "axisX", "axisY", "axisZ"];
    const addInput = (key, val, hint) => {
      const row = document.createElement("label");
      row.className = "kv"; row.style.alignItems = "center";
      row.appendChild(document.createTextNode(key + " "));
      const inp = document.createElement("input");
      inp.type = "number"; inp.step = "any"; inp.style.flex = "1"; inp.value = val;
      if (hint) inp.title = hint;
      inp.onchange = () => { part.primitive[key] = parseFloat(inp.value) || 0; save(); renderPrimOverlay(); };
      row.appendChild(inp); wrap.appendChild(row);
    };
    // shape dimensions
    Object.keys(part.primitive)
      .filter((k) => k !== "shape" && rotKeys.indexOf(k) === -1)
      .forEach((key) => addInput(key, part.primitive[key]));
    // rotation: degrees about a 3D axis. Defaults: no rotation, Y-up axis.
    if (part.primitive.rotation == null) part.primitive.rotation = 0;
    if (part.primitive.axisX == null) part.primitive.axisX = 0;
    if (part.primitive.axisY == null) part.primitive.axisY = 1;
    if (part.primitive.axisZ == null) part.primitive.axisZ = 0;
    const sep = document.createElement("div");
    sep.className = "hint";
    sep.style.margin = "4px 0 2px";
    sep.textContent = "rotation \u2014 degrees about axis (x, y, z):";
    wrap.appendChild(sep);
    addInput("rotation", part.primitive.rotation, "degrees of rotation");
    addInput("axisX", part.primitive.axisX, "axis vector x");
    addInput("axisY", part.primitive.axisY, "axis vector y");
    addInput("axisZ", part.primitive.axisZ, "axis vector z");
  }
  function ancestors(cls) {
    const out = []; let cur = state.classes[cls];
    while (cur) { out.push(cur); cur = state.classes[cur]; }
    return out;
  }

  // ---- Image list --------------------------------------------------
  function filterImagesFor(cls) {
    if (!cls) { state.imageFilter = null; state.imageFilterName = null; renderImages(); return; }
    // collect the class plus every descendant leaf, so selecting a branch
    // (e.g. a supercategory) shows all images under it.
    const kids = new Map();
    for (const c in state.classes) {
      const p = state.classes[c];
      if (!kids.has(p)) kids.set(p, []);
      kids.get(p).push(c);
    }
    const wanted = new Set([cls]);
    const stack = [cls];
    while (stack.length) {
      const cur = stack.pop();
      for (const ch of (kids.get(cur) || [])) {
        if (!wanted.has(ch)) { wanted.add(ch); stack.push(ch); }
      }
    }
    state.imageFilter = wanted;          // Set of class names
    state.imageFilterName = cls;         // label for the header
    renderImages();
  }

  function renderImages() {
    const ul = $("imageList"); ul.innerHTML = "";
    const flt = state.imageFilter;       // Set, or null for "all"
    const shown = state.images.filter((im) => !flt ||
      (state.annos[im.id] || []).some((a) => flt.has(a.cls)));
    if (flt) {
      const hdr = document.createElement("li");
      hdr.className = "hint";
      hdr.textContent = state.imageFilterName
        ? shown.length + ' image(s) for "' + state.imageFilterName + '" \u00b7 '
        : "Select a class to list its images \u00b7 ";
      const clr = document.createElement("a");
      clr.textContent = "show all"; clr.style.cssText = "cursor:pointer;color:#2E75B6";
      clr.onclick = () => filterImagesFor(null);
      hdr.appendChild(clr); ul.appendChild(hdr);
    }
    shown.forEach((im) => {
      const li = document.createElement("li");
      li.className = im.id === state.current ? "sel" : "";
      const label = document.createElement("span");
      label.className = "imglabel";
      label.textContent = im.name + ` (${(state.annos[im.id] || []).length})`;
      label.onclick = () => selectImage(im.id);
      const del = document.createElement("button");
      del.className = "imgdel"; del.textContent = "\u00d7";
      del.title = "Delete this image";
      del.onclick = (e) => { e.stopPropagation(); deleteImage(im.id); };
      li.appendChild(label); li.appendChild(del);
      ul.appendChild(li);
    });
  }

  async function deleteImage(id) {
    const im = state.images.find((i) => i.id === id);
    if (!im) return;
    if (!confirm(`Delete "${im.name}"? Its annotations will be removed.`)) return;
    const wasCurrent = state.current === id;
    state.images = state.images.filter((i) => i.id !== id);
    delete state.annos[id];
    try { await store.del("images", id); await store.del("annos", id); }
    catch (e) { /* persistence best-effort */ }
    if (wasCurrent) {
      state.current = null; state.selected = null;
      if (state.images.length) { selectImage(state.images[0].id); return; }
      img = new Image();                       // nothing left: clear the canvas
      if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
      $("stat").textContent = "No image \u2014 add candidates to begin";
    }
    refresh();
  }
  function fitCanvas() {
    if (!img || !img.naturalWidth) return;
    const main = document.querySelector("main");   // the real workspace, not the wrap
    const availW = Math.max(160, main.clientWidth * 0.97 - 16);
    const availH = Math.max(160, main.clientHeight * 0.97 - 84);  // minus toolbar+hints
    const ar = img.naturalWidth / img.naturalHeight;
    let w = availW, h = w / ar;
    if (h > availH) { h = availH; w = h * ar; }     // preserve aspect ratio
    canvas.width = Math.round(w);
    canvas.height = Math.round(h);
  }
  function selectImage(id) {
    state.current = id; state.selected = null; state.kpIndex = 0;
    const im = state.images.find((i) => i.id === id);
    img = new Image();
    img.onload = () => {
      fitCanvas();                            // fill the viewport, keep AR
      $("stat").textContent = `${im.name} — ${img.naturalWidth}×${img.naturalHeight}`;
      resetView();
      updateZoomLabel();
      refresh();
    };
    img.src = im.dataUrl;
  }
  window.addEventListener("resize", () => {
    if (img && img.src) { fitCanvas(); render(); }
  });

  // ---- Export (YOLO) -----------------------------------------------
  function leafClasses() {
    // Classes used by any annotation, in stable order.
    const used = new Set();
    for (const id in state.annos)
      for (const o of state.annos[id]) used.add(o.cls);
    return [...used];
  }
  function exportYolo() {
    const classList = leafClasses();
    const idOf = (c) => classList.indexOf(c);
    const files = {};
    for (const im of state.images) {
      const lines = [];
      for (const o of state.annos[im.id] || []) {
        const cx = (o.box[0] + o.box[2]) / 2 / im.w;
        const cy = (o.box[1] + o.box[3]) / 2 / im.h;
        const w = (o.box[2] - o.box[0]) / im.w;
        const h = (o.box[3] - o.box[1]) / im.h;
        const line = idOf(o.cls) + " " + cx.toFixed(6) + " " +
          cy.toFixed(6) + " " + w.toFixed(6) + " " + h.toFixed(6);
        lines.push(line);
      }
      files[im.name.replace(/\.[^.]+$/, "") + ".txt"] = lines.join("\n");
    }
    const hierarchy = {};
    classList.forEach((c) => { if (state.classes[c]) hierarchy[c] = state.classes[c]; });
    downloadText("classes.txt", classList.join("\n"));
    downloadText("hierarchy.json", JSON.stringify(hierarchy, null, 2));
    for (const fn in files) downloadText(fn, files[fn]);
  }
  function downloadText(name, text) {
    const a = document.createElement("a");
    a.href = "data:text/plain;charset=utf-8," + encodeURIComponent(text);
    a.download = name; a.click();
  }

  // ---- Training trigger --------------------------------------------
  // Resolve a class up to its level-1 leaf (the nearest fixed vocabulary
  // ancestor; or the class itself if it is already a level-1 leaf).
  async function startTraining() {
    const sel = state.activeClass;
    if (!sel) { alert("Select a class in the hierarchy first."); return; }
    const fixed = state.fixed || new Set();
    // One level-2 model per level-1 leaf, trained on ALL of its custom classes
    // at once regardless of how deeply they are nested; data + weights are saved
    // to a folder named after that level-1 leaf class.
    const level1LeafOf = (cls) => {
      let cur = cls, hops = 0;
      while (cur && !fixed.has(cur) && hops < 16) { cur = state.classes[cur]; hops++; }
      return cur || state.classes[cls] || cls;
    };
    const parent = level1LeafOf(sel);
    const custom = [];
    for (const c in state.classes) {
      if (fixed.has(c)) continue;                 // custom classes only
      if (level1LeafOf(c) === parent) custom.push(c);
    }
    if (!custom.length) {
      alert('No custom classes under "' + parent + '". Add a class beneath it ' +
            "(and annotate some images) before training.");
      return;
    }
    const want = new Set(custom);
    // Keep only this level-1 leaf's level-2 annotations; each box carries its
    // own parts / keypoints / primitives+rotations / properties.
    const images = state.images.map((im) => {
      const objs = (state.annos[im.id] || []).filter((o) => want.has(o.cls));
      return {
        name: im.name, width: im.w, height: im.h, data: im.dataUrl,
        objects: objs.map((o) => ({
          cls: o.cls, box: o.box, parts: o.parts,
          keypoints: o.keypoints, properties: o.properties,
        })),
      };
    }).filter((im) => im.objects.length);
    if (!images.length) {
      alert('No annotations yet for the custom classes under "' + parent +
            '". Draw at least one box for a custom class first.');
      return;
    }
    const payload = {
      classes: custom,
      hierarchy: state.classes,
      images: images,
      config: {
        epochs: 50, imgsz: 640,
        parent_class: parent,
        name: parent,                 // data + weights folder = level-1 leaf class
      },
    };
    try {
      const res = await fetch("/train", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (data.job_id) pollTraining(data.job_id, parent);
      else $("saved").textContent = "Training: " + (data.status || "submitted");
    } catch (err) {
      $("saved").textContent = "Training endpoint offline (see Chapter 27)";
    }
  }

  async function pollTraining(jobId, leaf) {
    const tick = async () => {
      try {
        const r = await fetch("/train/" + jobId);
        const s = await r.json();
        const pct = s.progress != null ? " " + Math.round(s.progress * 100) + "%" : "";
        const ep = s.epoch ? ` (epoch ${s.epoch}/${s.epochs || "?"})` : "";
        $("saved").textContent = `Training [${jobId}]: ${s.status}${pct}${ep}`;
        if (s.status === "done") {
          $("saved").textContent += " \u2014 weights: " + (s.weights || "saved");
          if (leaf) {
            state.trained = state.trained || new Set();
            state.trained.add(leaf);
            try { await store.put("meta", "trained", [...state.trained]); } catch (e) {}
            renderClasses();
          }
          return;
        }
        if (s.status === "failed") {
          $("saved").textContent = "Training failed: " + (s.error || "see logs");
          return;
        }
        setTimeout(tick, 2000);           // keep polling until done/failed
      } catch (err) {
        $("saved").textContent = "Lost contact with training job " + jobId;
      }
    };
    tick();
  }

  // ---- Wire up -----------------------------------------------------
  function refresh() { render(); renderObjects(); renderImages(); }

  $("file").addEventListener("change", (e) => {
    for (const f of e.target.files) {
      const reader = new FileReader();
      reader.onload = () => {
        const probe = new Image();
        probe.onload = () => {
          const id = "img_" + Date.now() + "_" + Math.random().toString(36).slice(2, 6);
          const rec = { id, name: f.name, dataUrl: reader.result,
            w: probe.naturalWidth, h: probe.naturalHeight };
          state.images.push(rec); state.annos[id] = [];
          if (!state.current) selectImage(id); else renderImages();
        };
        probe.src = reader.result;
      };
      reader.readAsDataURL(f);
    }
  });

  // ---- Import annotated images for the active class (two sources) --------
  async function downloadImages(url, label, btn) {
    const leaf = state.activeClass;
    if (!leaf) {
      alert("Click a class in the hierarchy first \u2014 its name is the search " +
            "term, and imported boxes are labelled with it.");
      return;
    }
    state.importedSrc = state.importedSrc || {};
    const n = parseInt(($("dlN") && $("dlN").value) || "10", 10) || 10;
    const orig = btn.textContent;
    btn.disabled = true; btn.textContent = "Fetching\u2026";
    try {
      const fd = new FormData();
      fd.append("class_name", leaf); fd.append("n", String(n));
      // skip images already imported for THIS class so we get new ones
      const exclude = state.importedSrc[leaf] || [];
      fd.append("exclude", JSON.stringify(exclude));
      const r = await fetch(url, { method: "POST", body: fd });
      if (!r.ok) {
        const j = await r.json().catch(() => ({}));
        alert(label + " fetch failed: " + (j.detail || j.error || r.status));
      } else {
        const data = await r.json();
        let added = 0, firstId = null;
        for (const im of data.images || []) {
          const id = "dl_" + Date.now() + "_" +
            Math.random().toString(36).slice(2, 6);
          const rec = { id, name: im.name, dataUrl: im.data,
            w: im.width, h: im.height };
          const annos = (im.boxes || []).map((b) => ({
            id: nextId(), cls: leaf, box: b.box,
            parts: [], keypoints: [], properties: {} }));
          state.images.push(rec); state.annos[id] = annos;
          await store.put("images", id, rec);
          await store.put("annos", id, annos);
          if (im.source_id) {
            (state.importedSrc[leaf] = state.importedSrc[leaf] || [])
              .push(im.source_id);
          }
          if (!firstId) firstId = id;
          added++;
        }
        if (added) await store.put("meta", "importedSrc", state.importedSrc);
        if (added && !state.current && firstId) selectImage(firstId);
        else renderImages();
        if ($("saved")) $("saved").textContent = added
          ? "Imported " + added + " " + label + " images for " + leaf
          : "No new " + label + " images for " + leaf +
            " (all available already imported)";
        if (data.license) console.log(label + " license:", data.license);
      }
    } catch (e) { alert(label + " fetch failed: " + e.message); }
    btn.disabled = false; btn.textContent = orig;
  }
  if ($("fetchOI")) $("fetchOI").addEventListener("click", () =>
    downloadImages("/fetch_open_images", "Open Images", $("fetchOI")));

  document.querySelectorAll(".tool").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tool").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.tool = btn.dataset.tool;
      canvas.style.cursor = state.tool === "select" ? "pointer"
        : state.tool === "pan" ? "grab" : "crosshair";
    });
  });
  // Zoom buttons (zoom around the canvas centre).
  const zc = () => [canvas.width / 2, canvas.height / 2];
  if ($("zoomIn")) $("zoomIn").addEventListener("click", () => zoomAt(...zc(), 1.25));
  if ($("zoomOut")) $("zoomOut").addEventListener("click", () => zoomAt(...zc(), 1 / 1.25));
  if ($("zoomReset")) $("zoomReset").addEventListener("click", () => {
    resetView(); updateZoomLabel(); render();
  });
  // Designate a geometric primitive for the selected part.
  $("partPrim").addEventListener("change", (e) => {
    const o = selectedObj(); if (!o || state.selectedPart == null) return;
    const part = o.parts[state.selectedPart]; if (!part) return;
    const shape = e.target.value;
    if (!shape) delete part.primitive;
    else part.primitive = { shape, ...defaultDims(shape, part) };
    save(); renderObjects();
  });
  $("addClass").addEventListener("click", () => {
    const c = $("newClass").value.trim();
    const p = state.activeClass;            // attach as a child of the active class
    if (!c) return;
    if (state.fixed && state.fixed.has(c)) {
      alert('"' + c + '" is a fixed level-1 YOLOE class and cannot be redefined.');
      return;
    }
    if (!p || !(p in state.classes)) {
      alert("Select a parent class first (click one in the tree or use search). " +
            "The new class is added as its child.");
      return;
    }
    state.classes[c] = p; state.activeClass = c;
    expandTo(c);
    $("newClass").value = "";
    renderClasses(); save();
  });

  // ---- Named Persons: face enrollment ------------------------------------
  async function renderFaces() {
    const ul = $("faceList"); if (!ul) return;
    let names = [];
    try {
      const r = await fetch("/faces");
      if (r.ok) names = (await r.json()).identities || [];
    } catch (e) { /* server offline */ }
    ul.innerHTML = "";
    for (const nm of names) {
      const li = document.createElement("li");
      li.className = "treeitem";
      const lbl = document.createElement("span");
      lbl.textContent = nm;
      lbl.style.cursor = "pointer";
      lbl.onclick = () => { state.activeClass = nm; renderClasses(); };
      const del = document.createElement("button");
      del.textContent = "\u00d7"; del.title = "remove identity";
      del.style.marginLeft = "6px";
      del.onclick = async () => {
        const fd = new FormData(); fd.append("name", nm);
        try { await fetch("/faces/delete", { method: "POST", body: fd }); }
        catch (e) { /* ignore */ }
        delete state.classes[nm]; renderClasses(); renderFaces();
      };
      li.appendChild(lbl); li.appendChild(del); ul.appendChild(li);
    }
    if (names.length) { renderClasses(); save(); }
  }
  if ($("enrollFace")) {
    $("enrollFace").addEventListener("click", async () => {
      const nm = ($("faceName").value || "").trim();
      const file = $("faceFile").files[0];
      if (!nm || !file) { alert("Enter a name and choose a face photo."); return; }
      const fd = new FormData();
      fd.append("name", nm); fd.append("file", file);
      $("enrollFace").disabled = true;
      $("enrollFace").textContent = "Enrolling\u2026";
      try {
        const r = await fetch("/enroll_face", { method: "POST", body: fd });
        if (r.ok) {
          $("faceName").value = ""; $("faceFile").value = "";
          await renderFaces();
        } else {
          const j = await r.json().catch(() => ({}));
          alert("Enrollment failed: " + (j.error || r.status));
        }
      } catch (e) { alert("Enrollment failed: " + e.message); }
      $("enrollFace").disabled = false;
      $("enrollFace").textContent = "Enroll face";
    });
    renderFaces();
  }
  $("addProp").addEventListener("click", () => {
    const o = selectedObj(); if (!o) return;
    const k = $("propKey").value.trim(); const v = $("propVal").value.trim();
    if (k) o.properties[k] = v;
    $("propKey").value = ""; $("propVal").value = ""; renderObjects();
  });
  $("delObj").addEventListener("click", () => {
    state.annos[state.current] = curAnnos().filter((o) => o.id !== state.selected);
    state.selected = null; refresh();
  });
  $("save").addEventListener("click", save);
  $("export").addEventListener("click", exportYolo);
  $("train").addEventListener("click", startTraining);

  // ---- Boot --------------------------------------------------------
  (async function () {
    await load();
    state.activeClass = null;            // no class selected on init
    state.current = null;               // no image loaded on the canvas
    state.imageFilter = new Set();      // empty filter -> image list starts cleared
    state.imageFilterName = null;
    renderClasses();
    renderImages();
    renderObjects();
  })();
})();
