// overlay.js — draw world-model JSON onto a canvas, one capability at a
// time. The display MODE selects which capability layers are drawn.
// Coordinates use the SAME object-fit: cover transform as the <video>,
// so boxes line up tightly with the displayed frame.
(function (global) {
  "use strict";

  // Class hierarchy (supercategory -> subcategory -> class), fetched once so
  // the "hierarchy" layer can show a detection's FULL ancestry, not just its
  // immediate parent.
  let HIER = {};
  fetch("/data/yoloe_vocab.json").then((r) => (r.ok ? r.json() : null))
    .then((v) => { if (v && v.hierarchy) HIER = v.hierarchy; })
    .catch(() => {});

  function ancestryChain(cls, parentClass) {
    const chain = [];
    let cur = cls;
    if (!(cur in HIER) && parentClass && parentClass in HIER) {
      chain.push(cls);              // custom leaf not in vocab: start at parent
      cur = parentClass;
    }
    let hops = 0;
    while (cur && hops < 12) { chain.push(cur); cur = HIER[cur]; hops++; }
    chain.reverse();
    if (chain[0] !== "object") chain.unshift("object");   // ensure a root label
    return chain;
  }

  const SKELETON = [
    ["left_shoulder", "right_shoulder"], ["left_shoulder", "left_elbow"],
    ["left_elbow", "left_wrist"], ["right_shoulder", "right_elbow"],
    ["right_elbow", "right_wrist"], ["left_shoulder", "left_hip"],
    ["right_shoulder", "right_hip"], ["left_hip", "right_hip"],
    ["left_hip", "left_knee"], ["left_knee", "left_ankle"],
    ["right_hip", "right_knee"], ["right_knee", "right_ankle"],
    ["nose", "left_eye"], ["nose", "right_eye"],
    ["left_eye", "left_ear"], ["right_eye", "right_ear"],
  ];

  const cache = {};
  function colorFor(name) {
    if (cache[name]) return cache[name];
    let h = 0;
    for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
    return (cache[name] = `hsl(${h % 360}, 70%, 60%)`);
  }

  const MODES = [
    { id: "all",         label: "All capabilities" },
    { id: "detection",   label: "Detection (boxes)" },
    { id: "segmentation",label: "Segmentation (per-object masks)" },
    { id: "keypoints",   label: "Keypoints / pose (named)" },
    { id: "tracking",    label: "Tracking (persistent IDs + trails)" },
    { id: "hierarchy",   label: "Class hierarchy (ancestor\u2192leaf)" },
    { id: "parts",       label: "Parts (named sub-objects)" },
    { id: "scenegraph",  label: "Scene graph (relations)" },
    { id: "depth",       label: "Depth (near\u2192far)" },
    { id: "3d",          label: "3D reconstruction (WebGL)" },
    { id: "yoloe",       label: "YOLOE (open-vocab \u2014 type any classes)" },
    { id: "cascade",     label: "Two-phase (YOLOE \u2192 custom level-2)" },
  ];
  const LAYERS = {
    all:          new Set(["graph", "mask", "box", "label", "pose", "parts"]),
    detection:    new Set(["box", "label"]),
    segmentation: new Set(["segmask"]),
    keypoints:    new Set(["pose", "kpname"]),
    tracking:     new Set(["box", "track", "trail"]),
    hierarchy:    new Set(["box", "hierarchy"]),
    parts:        new Set(["box", "parts", "partname"]),
    scenegraph:   new Set(["graph", "box"]),
    depth:        new Set(["depthbox", "depthlabel"]),
    "3d":         new Set([]),
    yoloe:        new Set(["segmask", "box", "label"]),
    cascade:      new Set(["segmask", "box", "label", "parts", "partname",
                           "pose", "hierarchy"]),
  };

  let latest = null;
  let mode = "all";
  const trails = {};

  function setWorld(world) {
    latest = world;
    for (const o of (world && world.objects) || []) {
      if (o.track_id == null) continue;
      const cx = (o.box.x1 + o.box.x2) / 2, cy = (o.box.y1 + o.box.y2) / 2;
      (trails[o.track_id] = trails[o.track_id] || []).push([cx, cy]);
      if (trails[o.track_id].length > 40) trails[o.track_id].shift();
    }
  }
  function setMode(m) { if (LAYERS[m]) mode = m; }
  function getMode() { return mode; }
  function getWorld() { return latest; }

  // object-fit: cover transform from image space to canvas pixels.
  function projector(canvas) {
    const iw = latest.image.width, ih = latest.image.height;
    const scale = Math.max(canvas.width / iw, canvas.height / ih);
    const ox = (canvas.width - iw * scale) / 2;
    const oy = (canvas.height - ih * scale) / 2;
    return {
      x: (v) => v * scale + ox,
      y: (v) => v * scale + oy,
      s: scale,
    };
  }

  function draw(canvas) {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!latest || !latest.image) return;
    const on = LAYERS[mode] || LAYERS.all;
    const P = projector(canvas);
    ctx.lineWidth = 2;
    ctx.font = "600 14px system-ui, sans-serif";
    ctx.textBaseline = "top";

    const byId = {};
    for (const o of latest.objects || []) byId[o.id] = o;

    if (on.has("graph")) {
      const edges = (latest.graph && latest.graph.edges) || [];
      ctx.save();
      ctx.lineWidth = 1.5; ctx.strokeStyle = "rgba(255,255,255,0.8)";
      ctx.font = "600 12px system-ui, sans-serif";
      for (const e of edges) {
        const s = byId[e.subject_id], t = byId[e.object_id];
        if (!s || !t) continue;
        const a = [P.x((s.box.x1 + s.box.x2) / 2), P.y((s.box.y1 + s.box.y2) / 2)];
        const b = [P.x((t.box.x1 + t.box.x2) / 2), P.y((t.box.y1 + t.box.y2) / 2)];
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
        const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2;
        const tw = ctx.measureText(e.label).width;
        ctx.fillStyle = "rgba(0,0,0,0.6)"; ctx.fillRect(mx - tw / 2 - 3, my - 8, tw + 6, 16);
        ctx.fillStyle = "#fff"; ctx.fillText(e.label, mx - tw / 2, my - 7);
      }
      ctx.restore();
    }

    if (on.has("trail")) {
      for (const id in trails) {
        const pts = trails[id];
        if (pts.length < 2) continue;
        ctx.strokeStyle = colorFor("track" + id); ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(P.x(pts[0][0]), P.y(pts[0][1]));
        for (let i = 1; i < pts.length; i++) ctx.lineTo(P.x(pts[i][0]), P.y(pts[i][1]));
        ctx.stroke();
      }
    }

    let dMin = Infinity, dMax = -Infinity;
    if (on.has("depthbox")) {
      for (const o of latest.objects || []) {
        const d = o.properties && o.properties.depth;
        if (d != null) { dMin = Math.min(dMin, d); dMax = Math.max(dMax, d); }
      }
    }

    for (const o of latest.objects || []) {
      const b = o.box;
      const x = P.x(b.x1), y = P.y(b.y1), w = P.x(b.x2) - P.x(b.x1), h = P.y(b.y2) - P.y(b.y1);
      const col = colorFor(o.cls);

      // Segmentation: fill each object's mask with a DISTINCT colour,
      // shading its silhouette, plus a clear outline. Falls back to the
      // box only when no polygon is available.
      if (on.has("segmask")) {
        const oc = colorFor(o.cls + "#" + (o.track_id != null ? o.track_id : o.id));
        if (o.mask && o.mask.length > 2) {
          ctx.save();
          ctx.lineJoin = "round";
          ctx.beginPath();
          ctx.moveTo(P.x(o.mask[0][0]), P.y(o.mask[0][1]));
          for (let i = 1; i < o.mask.length; i++) ctx.lineTo(P.x(o.mask[i][0]), P.y(o.mask[i][1]));
          ctx.closePath();
          ctx.globalAlpha = 0.5; ctx.fillStyle = oc; ctx.fill();   // shaded silhouette
          ctx.globalAlpha = 1.0; ctx.lineWidth = 3; ctx.strokeStyle = oc; ctx.stroke();
          // a thin dark inner edge so adjacent objects stay distinct
          ctx.globalAlpha = 0.9; ctx.lineWidth = 1; ctx.strokeStyle = "rgba(0,0,0,0.6)"; ctx.stroke();
          ctx.restore();
        } else {
          ctx.globalAlpha = 0.4; ctx.fillStyle = oc; ctx.fillRect(x, y, w, h);
          ctx.globalAlpha = 1.0; ctx.lineWidth = 2; ctx.strokeStyle = oc; ctx.strokeRect(x, y, w, h);
        }
        continue;                            // segmentation view shows only masks
      }

      if (on.has("mask") && o.mask && o.mask.length > 2) {
        ctx.beginPath(); ctx.moveTo(P.x(o.mask[0][0]), P.y(o.mask[0][1]));
        for (let i = 1; i < o.mask.length; i++) ctx.lineTo(P.x(o.mask[i][0]), P.y(o.mask[i][1]));
        ctx.closePath();
        ctx.globalAlpha = 0.32; ctx.fillStyle = col; ctx.fill(); ctx.globalAlpha = 1.0;
      }

      if (on.has("depthbox")) {
        const d = o.properties && o.properties.depth;
        const t = (d != null && dMax > dMin) ? (d - dMin) / (dMax - dMin) : 0.5;
        const hue = 20 + 200 * t;            // near=warm, far=cool
        ctx.strokeStyle = `hsl(${hue}, 85%, 60%)`; ctx.lineWidth = 2 + 3 * (1 - t);
        ctx.strokeRect(x, y, w, h);
        if (on.has("depthlabel") && d != null) {
          ctx.fillStyle = `hsl(${hue}, 85%, 60%)`; ctx.fillText("d=" + d.toFixed(2), x + 4, y + 2);
        }
      }

      if (on.has("box")) { ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.strokeRect(x, y, w, h); }

      let label = null;
      if (on.has("label")) {
        label = `${o.cls} ${(o.confidence || 0).toFixed(2)}`;
        if (o.properties && o.properties.text) label = '"' + o.properties.text + '"';
      } else if (on.has("track")) {
        label = (o.track_id != null ? "#" + o.track_id + " " : "") + o.cls;
      } else if (on.has("hierarchy")) {
        label = ancestryChain(o.cls, o.parent_class).join(" \u2192 ");
      }
      // Surface the recognised person's name on the box when known.
      if (label && o.properties && o.properties.identity &&
          o.properties.identity !== "unknown") {
        label = o.properties.identity + " \u00b7 " + label;
      }
      if (label) {
        const tw = ctx.measureText(label).width + 8;
        ctx.fillStyle = col; ctx.fillRect(x, y - 18, tw, 18);
        ctx.fillStyle = "#0e1116"; ctx.fillText(label, x + 4, y - 17);
      }

      if (on.has("pose") && o.keypoints && o.keypoints.length) {
        const pt = {};
        for (const k of o.keypoints) if (k.visible) pt[k.name] = [P.x(k.x), P.y(k.y)];
        ctx.strokeStyle = col; ctx.lineWidth = 2;
        for (const [a, c] of SKELETON) {
          if (pt[a] && pt[c]) { ctx.beginPath(); ctx.moveTo(pt[a][0], pt[a][1]); ctx.lineTo(pt[c][0], pt[c][1]); ctx.stroke(); }
        }
        ctx.fillStyle = "#ffd400";
        for (const name in pt) {
          ctx.beginPath(); ctx.arc(pt[name][0], pt[name][1], 3, 0, 6.283); ctx.fill();
          if (on.has("kpname")) {
            ctx.fillStyle = "#fff"; ctx.font = "600 11px system-ui";
            ctx.fillText(name, pt[name][0] + 5, pt[name][1] - 4);
            ctx.fillStyle = "#ffd400"; ctx.font = "600 14px system-ui";
          }
        }
      }

      if (on.has("parts") && o.parts && o.parts.length) {
        ctx.strokeStyle = col; ctx.lineWidth = 1; ctx.setLineDash([4, 3]);
        for (const part of o.parts) {
          const pb = part.box;
          ctx.strokeRect(P.x(pb.x1), P.y(pb.y1), P.x(pb.x2) - P.x(pb.x1), P.y(pb.y2) - P.y(pb.y1));
          if (on.has("partname") && part.cls) {
            ctx.setLineDash([]); ctx.fillStyle = col; ctx.font = "600 11px system-ui";
            ctx.fillText(part.cls, P.x(pb.x1) + 2, P.y(pb.y1) + 2);
            ctx.font = "600 14px system-ui"; ctx.setLineDash([4, 3]);
          }
        }
        ctx.setLineDash([]);
      }
    }
  }

  function describe() {
    if (!latest) return ["(waiting for frames\u2026)"];
    const objs = latest.objects || [];
    if (!objs.length) return ["(nothing detected)"];
    const IND = "  ";
    const lines = [];

    // Build a tree from each object's full ancestry chain (object -> ... -> class);
    // every detected instance hangs at its leaf class node.
    const root = { children: {}, items: [] };
    for (const o of objs) {
      const chain = ancestryChain(o.cls, o.parent_class);
      let node = root;
      for (const seg of chain) {
        node.children[seg] = node.children[seg] || { children: {}, items: [] };
        node = node.children[seg];
      }
      node.items.push(o);
    }

    // one detail per indented line for a single detected instance
    function detailLines(o, depth) {
      const pad = IND.repeat(depth);
      const p = o.properties || {};
      const out = [];
      if (p.identity && p.identity !== "unknown") out.push(pad + "name: " + p.identity);
      if (o.track_id != null) out.push(pad + "track id: #" + o.track_id);
      if (o.keypoints && o.keypoints.length) {
        const vis = o.keypoints.filter((k) => k.visible).map((k) => k.name);
        out.push(pad + "keypoints: " + o.keypoints.length +
          (vis.length ? " (" + vis.slice(0, 6).join(", ") +
            (vis.length > 6 ? ", \u2026" : "") + ")" : ""));
      }
      if (o.parts && o.parts.length) {
        const pn = o.parts.filter((pt) => pt.cls).map((pt) => {
          const prim = pt.properties && pt.properties.primitive;
          const shape = prim ? (prim.shape || prim) : null;
          return pt.cls + (shape ? " [" + shape + "]" : "");
        });
        if (pn.length) out.push(pad + "parts: " + pn.join(", "));
      }
      if (p.text) out.push(pad + "ocr: \"" + p.text + "\"");
      // every remaining property, one per indented line (nothing dropped)
      const PRETTY = { dominant_color: "colour", primitive: "3d",
                       identity_score: "name confidence", face_box: "face box",
                       parent_class: "parent" };
      const skip = { identity: 1, text: 1 };          // already shown above
      for (const key of Object.keys(p)) {
        if (skip[key]) continue;
        let v = p[key];
        if (v == null) continue;
        if (Array.isArray(v)) {
          v = v.map((x) => (typeof x === "number" ? +x.toFixed(1) : x)).join(", ");
        } else if (typeof v === "object") {
          v = v.shape
            ? v.shape + (v.rotation ? " (rot " + v.rotation + "\u00b0)" : "")
            : JSON.stringify(v);
        } else if (typeof v === "number") {
          v = Math.abs(v) < 1000 ? +v.toFixed(2) : v;
        }
        if (v === "") continue;
        out.push(pad + (PRETTY[key] || key) + ": " + v);
      }
      return out;
    }

    // walk the tree, one indent level per generation
    function walk(node, depth) {
      for (const k of Object.keys(node.children).sort()) {
        const child = node.children[k];
        const inst = child.items;
        const pad = IND.repeat(depth);
        if (inst.length === 0) {
          lines.push(pad + k);                       // pure category node
        } else if (inst.length === 1) {
          const o = inst[0];
          lines.push(pad + k + (o.track_id != null ? " #" + o.track_id : ""));
          detailLines(o, depth + 1).forEach((l) => lines.push(l));
        } else {
          lines.push(pad + k + "  (\u00d7" + inst.length + ")");
          inst.forEach((o, i) => {
            lines.push(IND.repeat(depth + 1) + k +
              (o.track_id != null ? " #" + o.track_id : " [" + (i + 1) + "]"));
            detailLines(o, depth + 2).forEach((l) => lines.push(l));
          });
        }
        if (Object.keys(child.children).length) walk(child, depth + 1);
      }
    }
    walk(root, 0);

    // Scene graph, placed below the tree.
    const edges = (latest.graph && latest.graph.edges) || [];
    if (edges.length) {
      lines.push("");
      lines.push("scene graph:");
      edges.forEach((e) => {
        const sub = objs.find((o) => o.id === e.subject_id);
        const obj = objs.find((o) => o.id === e.object_id);
        const sl = sub ? sub.cls + (sub.track_id != null ? " #" + sub.track_id : "") : "?";
        const ol = obj ? obj.cls + (obj.track_id != null ? " #" + obj.track_id : "") : "?";
        lines.push(IND + sl + " \u2014 " + e.label + " \u2014 " + ol);
      });
    }
    return lines;
  }

  global.Overlay = { setWorld, draw, colorFor, setMode, getMode, getWorld, describe, MODES, projector };
})(window);
