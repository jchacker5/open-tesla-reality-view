// otrv studio front end: plain ES modules, no build step, no third-party code, no external requests.
const TOKEN = document.querySelector('meta[name="otrv-token"]').content;
// city pages live on a second local server (another origin: no token, no cookie, their own CSP)
const CITIES = document.querySelector('meta[name="otrv-cities"]').content;
const cityUrl = (name, hash = "") => `${CITIES}/cities/${encodeURIComponent(name)}/${hash}`;
const STAGES = ["frames", "masks", "features", "match", "map", "undistort", "train", "export"];
const STAGE_LABEL = {
  frames: "Frames", masks: "Masks", features: "Features", match: "Match",
  map: "Map", undistort: "Undistort", train: "Train", export: "Export",
};
const STAGE_HINT = {
  frames: "sampling frames from each camera", masks: "masking your car (and movers)",
  features: "COLMAP SIFT features", match: "matching across cameras and time", map: "solving camera poses",
  undistort: "undistorting for training", train: "Brush training the splat", export: "writing the city",
};
const CAM_SHORT = { front: "Front", back: "Back", left_repeater: "Left", right_repeater: "Right", left_pillar: "L pillar", right_pillar: "R pillar" };
const outlet = document.getElementById("view");
let cleanup = [];

// ---- helpers ------------------------------------------------------------------------------------------

async function api(path, opts = {}) {
  const init = { ...opts, headers: { "X-OTRV-Token": TOKEN, ...(opts.headers || {}) } };
  if (opts.json !== undefined) {
    init.body = JSON.stringify(opts.json);
    init.headers["Content-Type"] = "application/json";
    init.method = init.method || "POST";
  }
  const r = await fetch(path, init);
  const ct = r.headers.get("content-type") || "";
  const data = ct.includes("json") ? await r.json() : await r.text();
  if (!r.ok) {
    let msg = data && data.detail ? data.detail : r.statusText;
    if (Array.isArray(msg)) msg = msg.map((d) => `${(d.loc || []).slice(-1)[0]}: ${d.msg}`).join("; ");
    throw new Error(msg);
  }
  return data;
}

// Drive mode is viewer v2 (window.otrvDrive, deep link #cam=drive). /api/viewer says whether the served
// viewer has it; without it the button is disabled with a tooltip instead of opening a plain replay.
let viewerInfo = null;
async function viewerFeatures() {
  if (!viewerInfo) {
    try { viewerInfo = await api("/api/viewer"); } catch { viewerInfo = { drive: false }; }
  }
  return viewerInfo;
}
function driveButton(name, drive) {
  if (drive) return h("a", { class: "btn primary", href: cityUrl(name, "#cam=drive"), target: "_blank", rel: "noopener" }, icon("car"), "Drive");
  return h("span", { class: "btn-wrap", title: "Drive mode needs viewer v2 (the served viewer has no window.otrvDrive yet); Replay works" },
    h("button", { class: "primary", disabled: true, "aria-label": "Drive (needs viewer v2)" }, icon("car"), "Drive"));
}

function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k === "style") for (const [p, val] of Object.entries(v)) el.style.setProperty(p, val);
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, v);
  }
  append(el, kids);
  return el;
}
function append(el, kids) {
  for (const k of kids.flat(Infinity)) {
    if (k === null || k === undefined || k === false) continue;
    el.append(k instanceof Node ? k : document.createTextNode(String(k)));
  }
  return el;
}
const NS = "http://www.w3.org/2000/svg";
function s(tag, attrs, ...kids) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v !== undefined && v !== null) el.setAttribute(k, v);
  for (const k of kids.flat(Infinity)) if (k) el.append(k);
  return el;
}
const ICONS = {
  shield: "M12 3l7 3v6c0 4.4-3 8-7 9-4-1-7-4.6-7-9V6l7-3z M9 12l2 2 4-4",
  usb: "M12 2v14 M8 6l4-4 4 4 M7 11h3v3 M17 9h-3v4 M12 16a3 3 0 1 0 0 6 3 3 0 0 0 0-6z",
  folder: "M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z",
  data: "M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3z M4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6 M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6",
  upload: "M12 16V4 M7 9l5-5 5 5 M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3",
  plus: "M12 5v14 M5 12h14",
  x: "M6 6l12 12 M18 6L6 18",
  back: "M15 18l-6-6 6-6",
  chev: "M9 6l6 6-6 6",
  spark: "M12 3l2.2 5.8L20 11l-5.8 2.2L12 19l-2.2-5.8L4 11l5.8-2.2z",
  play: "M7 5v14l12-7z",
  pause: "M8 5v14 M16 5v14",
  check: "M5 12l5 5 9-10",
  alert: "M12 8v5 M12 16.5v.5 M10.3 3.9L2.5 18a2 2 0 0 0 1.7 3h15.6a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z",
  car: "M5 16h14 M6 16l1.5-5.5A2 2 0 0 1 9.4 9h5.2a2 2 0 0 1 1.9 1.5L18 16 M7 16v2 M17 16v2 M8 13h1 M15 13h1",
  replay: "M4 12a8 8 0 1 0 2.3-5.7 M4 4v4h4",
  trash: "M4 7h16 M9 7V4h6v3 M6 7l1 13h10l1-13",
  refresh: "M20 12a8 8 0 1 1-2.3-5.7 M20 4v4h-4",
  film: "M4 5h16v14H4z M8 5v14 M16 5v14 M4 9h4 M4 15h4 M16 9h4 M16 15h4",
  map: "M9 4L3 6v14l6-2 6 2 6-2V4l-6 2z M9 4v14 M15 6v14",
  build: "M14 6l4 4 M4 20l7-7 M13 4l7 7-3 3-7-7z",
};
function icon(name, cls) {
  const paths = ICONS[name].split(/(?= M)/).map((d) => s("path", { d: d.trim() }));
  return s("svg", { viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", "stroke-width": "1.8", "stroke-linecap": "round", "stroke-linejoin": "round", class: cls }, paths);
}

let toastTimer;
function toast(msg, bad = false) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.className = `toast show${bad ? " bad" : ""}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.className = "toast"), bad ? 6000 : 3000);
}
function confirmDialog(title, body, okLabel = "Delete") {
  return new Promise((resolve) => {
    const d = h("dialog", {},
      h("h2", { text: title }), h("p", { text: body }),
      h("div", { class: "row" },
        h("button", { class: "ghost", onclick: () => { d.close(); resolve(false); } }, "Cancel"),
        h("button", { class: "primary", onclick: () => { d.close(); resolve(true); } }, okLabel)));
    d.addEventListener("close", () => { d.remove(); resolve(false); });
    document.body.append(d);
    d.showModal();
  });
}

const pad2 = (n) => String(n).padStart(2, "0");
function clock(sec) {
  sec = Math.max(0, Math.round(sec || 0));
  const m = Math.floor(sec / 60), ss = sec % 60;
  return m >= 60 ? `${Math.floor(m / 60)}:${pad2(m % 60)}:${pad2(ss)}` : `${m}:${pad2(ss)}`;
}
function human(sec) {
  if (sec === null || sec === undefined || !isFinite(sec)) return "—";
  sec = Math.max(0, sec);
  if (sec < 60) return `${Math.round(sec)} s`;
  if (sec < 3600) return `${Math.round(sec / 60)} min`;
  const hrs = Math.floor(sec / 3600), m = Math.round((sec % 3600) / 60);
  return m ? `${hrs} h ${m} min` : `${hrs} h`;
}
function bytes(n) {
  if (!n && n !== 0) return "—";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1000 && i < u.length - 1) { n /= 1000; i++; }
  return `${n >= 100 || i === 0 ? Math.round(n) : n.toFixed(1)} ${u[i]}`;
}
function parseTs(ts) {
  const [d, t] = ts.split("_");
  const [y, mo, da] = d.split("-").map(Number);
  const [hh, mm, ss] = t.split("-").map(Number);
  return new Date(y, mo - 1, da, hh, mm, ss);
}
const dateFmt = new Intl.DateTimeFormat(undefined, { weekday: "short", month: "short", day: "numeric" });
const timeFmt = new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" });
const timeSecFmt = new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit", second: "2-digit" });

// ---- svg charts ---------------------------------------------------------------------------------------

function sparkline(series, w = 280, hgt = 38, opts = {}) {
  const svg = s("svg", { viewBox: `0 0 ${w} ${hgt}`, preserveAspectRatio: "none", class: opts.cls || "spark" });
  if (!series || series.length < 2) {
    svg.append(s("text", { x: 0, y: hgt - 6, fill: "#5b616c", "font-size": "11" }, document.createTextNode("no telemetry speed")));
    return svg;
  }
  const t0 = series[0][0], t1 = series[series.length - 1][0] || 1;
  const vmax = Math.max(45, ...series.map((p) => p[1]));
  const X = (t) => ((t - t0) / Math.max(1, t1 - t0)) * w;
  const Y = (v) => hgt - 2 - (v / vmax) * (hgt - 6);
  const band = s("rect", { x: 0, width: w, y: Y(40), height: Y(10) - Y(40), fill: "rgba(62,207,142,.07)" });
  const pts = series.map((p) => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`);
  const id = `g${Math.random().toString(36).slice(2, 8)}`;
  const grad = s("linearGradient", { id, x1: 0, x2: 0, y1: 0, y2: 1 },
    s("stop", { offset: "0", "stop-color": "#3d7bff", "stop-opacity": ".35" }),
    s("stop", { offset: "1", "stop-color": "#3d7bff", "stop-opacity": "0" }));
  svg.append(s("defs", {}, grad), band,
    s("polygon", { points: `0,${hgt} ${pts.join(" ")} ${w},${hgt}`, fill: `url(#${id})` }),
    s("polyline", { points: pts.join(" "), fill: "none", stroke: "#3d7bff", "stroke-width": "1.5", "vector-effect": "non-scaling-stroke", "stroke-linejoin": "round" }));
  if (opts.best) {
    const b = opts.best;
    svg.append(s("rect", { x: X(b.offset), width: Math.max(2, X(b.offset + b.dur) - X(b.offset)), y: 0, height: hgt, fill: "rgba(61,123,255,.16)", stroke: "rgba(107,155,255,.6)", "stroke-width": "1", "vector-effect": "non-scaling-stroke" }));
  }
  return svg;
}

function projectRoute(route, w, hgt, padding = 8) {
  if (!route || route.length < 2) return null;
  const lat0 = route.reduce((a, p) => a + p[0], 0) / route.length;
  const k = Math.cos((lat0 * Math.PI) / 180);
  const xs = route.map((p) => p[1] * k), ys = route.map((p) => -p[0]);
  const minx = Math.min(...xs), maxx = Math.max(...xs), miny = Math.min(...ys), maxy = Math.max(...ys);
  const span = Math.max(maxx - minx, maxy - miny) || 1e-9;
  const sc = Math.min(w - 2 * padding, hgt - 2 * padding) / span;
  const ox = (w - (maxx - minx) * sc) / 2, oy = (hgt - (maxy - miny) * sc) / 2;
  return xs.map((x, i) => [ox + (x - minx) * sc, oy + (ys[i] - miny) * sc]);
}
function routeSvg(route, w = 74, hgt = 74, cls = "route") {
  const svg = s("svg", { viewBox: `0 0 ${w} ${hgt}`, class: cls, "aria-label": "Route" });
  const pts = projectRoute(route, w, hgt);
  if (!pts) {
    svg.append(s("circle", { cx: w / 2, cy: hgt / 2, r: 3, fill: "#5b616c" }));
    return svg;
  }
  const str = pts.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
  const a = pts[0], z = pts[pts.length - 1];
  svg.append(
    s("polyline", { points: str, fill: "none", stroke: "rgba(61,123,255,.25)", "stroke-width": "6", "stroke-linecap": "round", "stroke-linejoin": "round" }),
    s("polyline", { points: str, fill: "none", stroke: "#6b9bff", "stroke-width": "2", "stroke-linecap": "round", "stroke-linejoin": "round" }),
    s("circle", { cx: a[0], cy: a[1], r: 3, fill: "#07080a", stroke: "#8a909c", "stroke-width": "1.5" }),
    s("circle", { cx: z[0], cy: z[1], r: 3.5, fill: "#3d7bff", stroke: "#fff", "stroke-width": "1.2" }));
  return svg;
}

// ---- router -------------------------------------------------------------------------------------------

const routes = { sources: pageSources, drives: pageDrives, drive: pageDrive, build: pageBuild, gallery: pageGallery };
async function render() {
  for (const fn of cleanup) try { fn(); } catch { /* ignore */ }
  cleanup = [];
  const [name, ...rest] = (location.hash.replace(/^#\/?/, "") || "sources").split("/");
  const page = routes[name] || pageSources;
  const tab = name === "drive" ? "drives" : name;
  document.querySelectorAll(".tabs a").forEach((a) => a.classList.toggle("on", a.dataset.nav === tab));
  const root = document.createElement("div"); // per render: a slow page from the last route can't draw into this one
  outlet.replaceChildren(root);
  try {
    await page(root, ...rest.map(decodeURIComponent));
  } catch (e) {
    root.replaceChildren(h("div", { class: "card empty" }, icon("alert"), h("h3", { text: "Something went wrong" }), h("p", { text: e.message })));
  }
  outlet.focus({ preventScroll: true });
}
window.addEventListener("hashchange", () => { window.scrollTo(0, 0); render(); });
function every(ms, fn) {
  const id = setInterval(async () => { try { await fn(); } catch { /* offline or navigated away */ } }, ms);
  cleanup.push(() => clearInterval(id));
}

// header badge: builds in flight
async function refreshBadge() {
  try {
    const { jobs } = await api("/api/jobs");
    const n = jobs.filter((j) => j.state === "running" || j.state === "queued").length;
    const b = document.getElementById("buildBadge");
    b.hidden = !n;
    b.textContent = n;
  } catch { /* offline */ }
}
setInterval(refreshBadge, 5000);

// ---- Sources ------------------------------------------------------------------------------------------

async function pageSources(view) {
  const banner = h("section", { class: "banner" },
    h("div", { class: "shield" }, icon("shield")),
    h("div", {},
      h("div", { class: "eyebrow", text: "Private by design" }),
      h("h1", { text: "Everything runs on this Mac" }),
      h("p", { text: "Footage is read where it is: nothing is copied until you build, and then only the clip you pick. COLMAP and Brush run locally, and this app makes no requests to the internet." }),
      h("div", { class: "chips", id: "doctor" }, ["ffmpeg", "colmap", "brush", "disk", "ram"].map((n) => h("span", { class: "chip skeleton", text: `   ${n}   ` }))),
      h("div", { id: "doctorFixes" })));
  const list = h("div", { class: "sources" });
  const pathIn = h("input", { type: "text", class: "mono", placeholder: "/Volumes/TESLADRIVE/TeslaCam  or any folder of clips", "aria-label": "Footage folder path" });
  const addBtn = h("button", { class: "primary" }, icon("plus"), "Add folder");
  const pickBtn = h("button", {}, icon("folder"), "Choose folder…");
  const fileIn = h("input", { type: "file", webkitdirectory: true, multiple: true, hidden: true });
  const upBtn = h("button", { class: "ghost" }, icon("upload"), "Upload clips…");
  const uploadPanel = h("div");
  const addCard = h("div", { class: "card add-source" }, pathIn, addBtn,
    h("div", { class: "alt" }, pickBtn, upBtn, fileIn,
      h("span", { text: "The folder picker uses macOS. Upload copies only the clips you tick into the studio's data folder." })));
  const cta = h("div", { class: "row", style: { "margin-top": "18px" } });
  view.append(banner,
    h("div", { class: "section" }, h("h2", {}, "Footage", h("span", { class: "muted", id: "scanState" })), list),
    h("div", { class: "section" }, h("h2", { text: "Add footage" }), addCard, uploadPanel), cta);

  async function add(path) {
    if (!path.trim()) return;
    addBtn.disabled = true;
    try {
      drawSources(await api("/api/sources", { json: { path: path.trim() } }));
      pathIn.value = "";
      toast("Folder added. Indexing clips…");
    } catch (e) { toast(e.message, true); }
    addBtn.disabled = false;
  }
  addBtn.onclick = () => add(pathIn.value);
  pathIn.onkeydown = (e) => e.key === "Enter" && add(pathIn.value);
  pickBtn.onclick = async () => {
    pickBtn.disabled = true;
    try {
      const r = await api("/api/sources/pick", { method: "POST" });
      if (r.path) { drawSources(r); toast(`Added ${r.path}`); }
    } catch (e) { toast(e.message, true); }
    pickBtn.disabled = false;
  };
  upBtn.onclick = () => fileIn.click();
  fileIn.onchange = () => drawUpload([...fileIn.files], uploadPanel);

  function drawSources(data) {
    const { sources, scan } = data;
    const st = view.querySelector("#scanState");
    st.textContent = scan.running ? ` · indexing ${scan.done}/${scan.total}` : "";
    list.replaceChildren();
    if (!sources.length) {
      list.append(h("div", { class: "card empty" }, icon("usb"),
        h("h3", { text: "No TeslaCam footage found yet" }),
        h("p", { text: "Plug in the car's USB drive (it shows up as /Volumes/<name>/TeslaCam), or add a folder below." })));
    }
    for (const src of sources) {
      const kindLabel = { usb: "USB drive", folder: "Folder", data: "Ingested", uploads: "Uploads" }[src.kind];
      const ic = { usb: "usb", folder: "folder", data: "data", uploads: "upload" }[src.kind];
      const scanning = scan.running && scan.source === src.path;
      const row = h("div", { class: `card source ${src.kind}` },
        h("div", { class: "ico" }, icon(ic)),
        h("div", { class: "grow" },
          h("div", { class: "path", text: src.path }),
          h("div", { class: "sub" }, h("span", { class: `tag ${src.kind === "usb" ? "accent" : ""}`, text: kindLabel }),
            src.exists ? h("span", { text: `${src.clips} clip${src.clips === 1 ? "" : "s"}` }) : h("span", { class: "tag warn", text: "missing" }),
            scanning ? h("span", { class: "muted", text: "indexing…" }) : null)),
        src.removable ? h("button", { class: "sm ghost danger", title: "Remove from the studio (files are not touched)", onclick: async () => {
          drawSources(await api(`/api/sources?path=${encodeURIComponent(src.path)}`, { method: "DELETE" }));
        } }, icon("x"), "Remove") : h("span"));
      list.append(row);
    }
    if (scan.running && scan.total) {
      list.append(h("div", { class: "progress" }, h("i", { style: { width: `${(100 * scan.done) / scan.total}%` } })));
    }
    if (scan.error) list.append(h("div", { class: "fixes", text: scan.error }));
    const total = sources.reduce((a, x) => a + x.clips, 0);
    cta.replaceChildren(total ? h("a", { class: "btn primary big", href: "#/drives" }, `See ${total} clips as drives`, icon("chev")) : "");
  }

  const load = async () => drawSources(await api("/api/sources"));
  await load();
  let wasRunning = true;
  every(1500, async () => {
    const d = await api("/api/sources");
    if (d.scan.running || wasRunning) drawSources(d);
    wasRunning = d.scan.running;
  });

  api("/api/doctor").then(({ checks }) => {
    const box = view.querySelector("#doctor");
    if (!box) return;
    const nice = { ffmpeg: "ffmpeg", ffprobe: "ffprobe", colmap: "COLMAP", brush: "Brush", disk: "Disk", ram: "RAM", platform: "Mac" };
    box.replaceChildren(...checks.filter((c) => c.name !== "ffprobe").map((c) => {
      const cls = c.ok ? "ok" : c.warn_only ? "warn" : "fail";
      let val = "";
      if (c.name === "colmap" || c.name === "brush") val = (c.detail.match(/^\d+\.\d+(\.\d+)?/) || [""])[0];
      if (c.name === "disk") val = (c.detail.match(/^[\d.]+ GB/) || [""])[0] + " free";
      if (c.name === "ram") val = c.detail.split(";")[0];
      if (c.name === "platform") val = c.detail.split(" ").slice(-1)[0];
      return h("span", { class: `chip ${cls}`, title: c.detail + (c.fix ? `\nfix: ${c.fix}` : "") },
        h("span", { class: "st" }), nice[c.name] || c.name, val && !c.ok ? null : val ? h("span", { class: "k", text: val }) : null,
        !c.ok && !c.warn_only ? h("span", { class: "k", text: "missing" }) : null);
    }));
    const bad = checks.filter((c) => !c.ok && c.fix);
    if (bad.length) {
      view.querySelector("#doctorFixes").replaceChildren(h("div", { class: "fixes" },
        bad.map((c) => h("div", { class: c.warn_only ? "warn" : "" }, h("b", { text: `${c.name}: ` }), h("code", { text: c.fix })))));
    }
  }).catch(() => {});
}

function drawUpload(files, panel) {
  const re = /(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})-(front|back|left_repeater|right_repeater|left_pillar|right_pillar)\.mp4$/i;
  const groups = new Map();
  for (const f of files) {
    const m = f.name.match(re);
    if (!m || f.name.startsWith("._")) continue;
    const rel = f.webkitRelativePath || f.name;
    const folder = rel.split("/").slice(0, -1).join("/");
    const k = `${folder}|${m[1]}`;
    if (!groups.has(k)) groups.set(k, { ts: m[1], folder, files: [] });
    groups.get(k).files.push(f);
  }
  const clips = [...groups.values()].sort((a, b) => (a.ts < b.ts ? 1 : -1));
  if (!clips.length) { toast("No TeslaCam clips (YYYY-MM-DD_HH-MM-SS-front.mp4) in that folder", true); return; }
  const boxes = clips.map((c) => h("input", { type: "checkbox", checked: clips.length <= 12 }));
  const btn = h("button", { class: "primary" });
  const bar = h("i", { style: { width: "0%" } });
  const update = () => {
    const sel = clips.filter((_, i) => boxes[i].checked);
    const size = sel.reduce((a, c) => a + c.files.reduce((x, f) => x + f.size, 0), 0);
    btn.textContent = `Upload ${sel.length} clip${sel.length === 1 ? "" : "s"} (${bytes(size)})`;
    btn.disabled = !sel.length;
  };
  boxes.forEach((b) => (b.onchange = update));
  btn.onclick = async () => {
    const sel = clips.filter((_, i) => boxes[i].checked);
    const all = sel.flatMap((c) => c.files.map((f) => ({ f, rel: (f.webkitRelativePath || f.name) })));
    const total = all.reduce((a, x) => a + x.f.size, 0);
    let done = 0;
    btn.disabled = true;
    try {
      for (const { f, rel } of all) {
        await api(`/api/uploads/${rel.split("/").map(encodeURIComponent).join("/")}`, { method: "PUT", body: f, headers: { "Content-Type": "application/octet-stream" } });
        done += f.size;
        bar.style.width = `${(100 * done) / total}%`;
      }
      await api("/api/sources/rescan", { method: "POST" });
      toast(`Uploaded ${sel.length} clips`);
      panel.replaceChildren();
    } catch (e) { toast(e.message, true); btn.disabled = false; }
  };
  update();
  panel.replaceChildren(h("div", { class: "card pad", style: { "margin-top": "12px" } },
    h("div", { class: "row" }, h("div", { class: "grow" }, h("h3", { text: `${clips.length} clips found` }),
      h("div", { class: "muted", text: "Tick the clips to copy into the studio. Nothing leaves this Mac." })),
    h("button", { class: "sm ghost", onclick: () => { boxes.forEach((b) => (b.checked = true)); update(); } }, "Select all"), btn),
    h("div", { class: "progress", style: { "margin-top": "12px" } }, bar),
    h("div", { class: "upload-list" }, clips.map((c, i) => h("label", {}, boxes[i],
      h("span", { class: "mono", text: c.ts }), h("span", { class: "muted", text: `${c.files.length} cams` }),
      h("span", { class: "muted", text: bytes(c.files.reduce((x, f) => x + f.size, 0)) }))))));
}

// ---- Drives grid --------------------------------------------------------------------------------------

let driveFilter = "driving";
async function pageDrives(view) {
  const head = h("div", { class: "page-head" });
  const grid = h("div", { class: "grid" });
  view.append(head, grid);
  let lastKey = "";
  const draw = (data) => {
    const { drives, scan } = data;
    const key = JSON.stringify([drives.map((d) => d.id + d.n_clips), scan.running, driveFilter]);
    if (key === lastKey) return;
    lastKey = key;
    const shown = drives.filter((d) => driveFilter === "all" || (driveFilter === "sentry") === (d.kind === "SentryClips"));
    const clips = drives.reduce((a, d) => a + d.n_clips, 0);
    head.replaceChildren(
      h("div", {}, h("div", { class: "eyebrow", text: "Your footage" }), h("h1", { text: "Drives" }),
        h("p", { text: `${drives.length} drive${drives.length === 1 ? "" : "s"} · ${clips} clips${scan.running ? ` · indexing ${scan.done}/${scan.total}…` : ""}` })),
      h("div", { class: "filters" }, [["driving", "Driving"], ["sentry", "Sentry"], ["all", "All"]].map(([k, l]) =>
        h("button", { class: `chip btnchip ${driveFilter === k ? "on" : ""}`, onclick: () => { driveFilter = k; lastKey = ""; draw(data); } }, l))));
    grid.replaceChildren();
    if (!shown.length) {
      grid.append(h("div", { class: "card empty", style: { "grid-column": "1 / -1" } }, icon("film"),
        h("h3", { text: scan.running ? "Indexing your clips…" : "No drives here" }),
        h("p", {}, scan.running ? "Drives appear as clips are read." : h("a", { href: "#/sources", class: "btn sm" }, "Add footage"))));
    }
    for (const d of shown) grid.append(driveCard(d));
  };
  draw(await api("/api/drives"));
  every(2000, async () => draw(await api("/api/drives")));
}

function driveCard(d) {
  const t = parseTs(d.start);
  const cams = d.thumb_cams.length >= 4 ? d.thumb_cams.slice(0, 4) : d.thumb_cams.slice(0, 1);
  const thumbs = h("div", { class: `thumbs ${cams.length === 1 ? "one" : ""}` },
    cams.map((c) => h("img", { src: `/api/clips/${d.thumb_clip}/thumb.jpg?cam=${c}`, alt: `${CAM_SHORT[c]} camera`, loading: "lazy" })),
    d.best ? h("span", { class: "badge", title: `${Math.round(d.best.share * 100)}% of a 20 s window at 10-40 mph` }, icon("spark"), `Best window · ${Math.round(d.best.mean_mph)} mph`) : null,
    d.kind !== "RecentClips" ? h("span", { class: "kind" }, h("span", { class: "tag", text: d.kind === "SentryClips" ? "Sentry" : d.kind === "SavedClips" ? "Saved" : d.kind })) : null,
    h("span", { class: "dur", text: clock(d.seconds) }));
  const bestMark = d.best ? { offset: d.best.offset, dur: d.best.dur } : null;
  return h("article", { class: "card drive", tabindex: "0", onclick: () => (location.hash = `#/drive/${d.id}`), onkeydown: (e) => e.key === "Enter" && (location.hash = `#/drive/${d.id}`) },
    thumbs,
    h("div", { class: "drive-body" },
      h("div", { class: "drive-title" }, dateFmt.format(t), h("span", { text: timeFmt.format(t) })),
      routeSvg(d.route),
      h("div", { class: "drive-meta" },
        h("span", {}, h("b", { text: human(d.seconds) })),
        h("span", { text: `${d.n_cams} cams` }),
        d.mean_mph !== null ? h("span", {}, h("b", { text: Math.round(d.mean_mph) }), " mph avg") : h("span", { text: "no telemetry" })),
      sparkline(d.speed, 280, 38, { best: bestMark })));
}

// ---- Drive detail -------------------------------------------------------------------------------------

async function pageDrive(view, id) {
  const d = await api(`/api/drives/${id}`);
  const t0 = parseTs(d.start);
  const clips = d.clips.filter((c) => c.cams.length >= 4);
  const firstClip = clips[0] || d.clips[0];
  const state = {
    clip: firstClip, start: 0, dur: Math.min(20, Math.floor(firstClip.seconds || 20)), cam: "front",
    fps: 3, steps: 15000, movers: false, evalOn: false, name: "",
  };
  if (d.best) {
    const c = d.clips.find((x) => x.key === d.best.clip);
    if (c) Object.assign(state, { clip: c, start: d.best.start });
  }

  // player
  const main = h("video", { muted: true, playsinline: true, preload: "auto" });
  main.muted = true;
  const hudMph = h("div", { class: "mph", text: "—" });
  const camRow = h("div", { class: "cams" });
  const playBtn = h("button", { class: "sm" }, icon("pause"));
  const timeRead = h("span", { class: "grow" });
  const player = h("div", { class: "card player" },
    h("div", { class: "main" }, main, h("div", { class: "vid-hud" }, hudMph, h("div", { class: "unit", text: "MPH" })), h("span", { class: "vid-label", id: "mainLabel" })),
    camRow, h("div", { class: "player-bar" }, playBtn, timeRead, h("span", { id: "clipLabel" })));
  let side = [];

  // speed strip
  const W = 1000, H = 150, PADB = 22;
  const strip = s("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none", class: "strip" });
  const total = Math.max(1, d.seconds);
  const X = (t) => (t / total) * W;
  const vmax = Math.max(45, d.max_mph || 0);
  const Y = (v) => H - PADB - (v / vmax) * (H - PADB - 10);
  strip.append(s("rect", { x: 0, y: Y(40), width: W, height: Y(10) - Y(40), fill: "rgba(62,207,142,.06)" }));
  for (const c of d.clips) {
    strip.append(s("line", { x1: X(c.offset), x2: X(c.offset), y1: 0, y2: H - PADB, stroke: "#1f232a", "stroke-width": "1", "vector-effect": "non-scaling-stroke" }));
    const lbl = s("text", { x: X(c.offset) + 4, y: H - 6, fill: "#5b616c", "font-size": "11" });
    lbl.textContent = timeFmt.format(new Date(t0.getTime() + c.offset * 1000));
    strip.append(lbl);
  }
  if (d.speed.length > 1) {
    const pts = d.speed.map((p) => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(" ");
    strip.append(
      s("polygon", { points: `${X(d.speed[0][0])},${H - PADB} ${pts} ${X(d.speed[d.speed.length - 1][0])},${H - PADB}`, fill: "rgba(61,123,255,.10)" }),
      s("polyline", { points: pts, fill: "none", stroke: "#3d7bff", "stroke-width": "1.6", "vector-effect": "non-scaling-stroke", "stroke-linejoin": "round" }));
  } else {
    const tx = s("text", { x: W / 2, y: H / 2, fill: "#5b616c", "font-size": "13", "text-anchor": "middle" });
    tx.textContent = "no telemetry speed in this drive: pick a window by eye";
    strip.append(tx);
  }
  const win = s("rect", { class: "win", y: 2, height: H - PADB - 2, rx: 4, fill: "rgba(61,123,255,.22)", stroke: "#6b9bff", "stroke-width": "1.5", "vector-effect": "non-scaling-stroke" });
  const head = s("line", { y1: 0, y2: H - PADB, stroke: "#fff", "stroke-width": "1.2", "vector-effect": "non-scaling-stroke", opacity: ".8" });
  strip.append(win, head);

  const winRead = h("div", { class: "window-read" });
  const winSub = h("div", { class: "muted" });
  const durIn = h("input", { type: "range", min: 20, max: 60, step: 1, value: state.dur, "aria-label": "Window length" });
  const durVal = h("b");
  const sugBox = h("div", { class: "suggest" });
  const estBox = h("div", { class: "estimate" });
  const fpsIn = h("input", { type: "number", min: 1, max: 6, step: 0.5, value: state.fps });
  const stepsIn = h("select", {}, [7000, 15000, 30000].map((v) => h("option", { value: v, selected: v === state.steps, text: `${v.toLocaleString()} steps${v === 15000 ? " (default)" : v === 30000 ? " (best)" : " (fast)"}` })));
  const nameIn = h("input", { type: "text", class: "mono", placeholder: "auto" });
  const moversIn = h("input", { type: "checkbox" });
  const evalIn = h("input", { type: "checkbox" });
  const buildBtn = h("button", { class: "primary big", style: { width: "100%" } }, icon("build"), "Build city");

  function clipAt(t) {
    return clips.find((c) => t >= c.offset && t < c.offset + (c.seconds || 60)) || null;
  }
  function clampWindow() {
    const len = state.clip.seconds || 60;
    state.dur = Math.min(state.dur, Math.floor(len));
    state.start = Math.max(0, Math.min(state.start, len - state.dur));
  }
  function drawWindow() {
    clampWindow();
    const a = state.clip.offset + state.start;
    win.setAttribute("x", X(a));
    win.setAttribute("width", Math.max(2, X(a + state.dur) - X(a)));
    winRead.replaceChildren(`${clock(a)} – ${clock(a + state.dur)}`, h("small", { text: `${state.dur} s` }));
    return a;
  }
  function updateWindow(seek = true) {
    const a = drawWindow();
    const wall = new Date(t0.getTime() + a * 1000);
    winSub.textContent = `${timeSecFmt.format(wall)} · clip ${state.clip.id}`;
    durVal.textContent = `${state.dur} s`;
    view.querySelector("#clipLabel").textContent = state.clip.id;
    loadVideos(seek);
    estimate();
  }
  let loadedClip = null;
  function loadVideos(seek) {
    if (loadedClip !== state.clip.key) {
      loadedClip = state.clip.key;
      const camList = state.clip.cams;
      if (!camList.includes(state.cam)) state.cam = camList[0];
      main.src = `/api/clips/${state.clip.key}/video/${state.cam}`;
      view.querySelector("#mainLabel").textContent = CAM_SHORT[state.cam];
      side = camList.filter((c) => c !== state.cam).map((c) => {
        const v = h("video", { muted: true, playsinline: true, preload: "auto", src: `/api/clips/${state.clip.key}/video/${c}` });
        v.muted = true;
        const cell = h("div", { class: "cam", title: `Show ${CAM_SHORT[c]}`, onclick: () => { state.cam = c; loadedClip = null; loadVideos(true); } }, v, h("span", { class: "cam-label", text: CAM_SHORT[c] }));
        return { cam: c, v, cell };
      });
      camRow.replaceChildren(...side.map((x) => x.cell));
      camRow.style.setProperty("grid-template-columns", `repeat(${Math.max(3, side.length)}, 1fr)`);
      main.addEventListener("loadedmetadata", () => { main.currentTime = state.start; main.play().catch(() => {}); }, { once: true });
      return;
    }
    if (seek) { main.currentTime = state.start; if (!main.paused) main.play().catch(() => {}); }
  }
  main.addEventListener("timeupdate", () => {
    const tt = main.currentTime;
    if (tt > state.start + state.dur || tt < state.start - 0.5) main.currentTime = state.start;
    const abs = state.clip.offset + main.currentTime;
    head.setAttribute("x1", X(abs)); head.setAttribute("x2", X(abs));
    timeRead.textContent = `${clock(main.currentTime)} / ${clock(state.clip.seconds)}`;
    const sp = d.speed.length ? d.speed.reduce((best, p) => (Math.abs(p[0] - abs) < Math.abs(best[0] - abs) ? p : best)) : null;
    hudMph.textContent = sp ? Math.round(sp[1]) : "—";
    for (const x of side) if (Math.abs(x.v.currentTime - main.currentTime) > 0.25) x.v.currentTime = main.currentTime;
  });
  main.addEventListener("play", () => { playBtn.replaceChildren(icon("pause")); side.forEach((x) => x.v.play().catch(() => {})); });
  main.addEventListener("pause", () => { playBtn.replaceChildren(icon("play")); side.forEach((x) => x.v.pause()); });
  playBtn.onclick = () => (main.paused ? main.play() : main.pause());
  cleanup.push(() => { main.pause(); main.removeAttribute("src"); side.forEach((x) => { x.v.pause(); x.v.removeAttribute("src"); }); });

  // dragging the window
  let drag = null;
  const toT = (ev) => {
    const r = strip.getBoundingClientRect();
    return ((ev.clientX - r.left) / r.width) * total;
  };
  strip.addEventListener("pointerdown", (ev) => {
    const t = toT(ev);
    const a = state.clip.offset + state.start;
    const inside = t >= a && t <= a + state.dur;
    drag = { grab: inside ? t - a : state.dur / 2 };
    strip.setPointerCapture(ev.pointerId);
    strip.classList.add("dragging");
    moveTo(t);
  });
  strip.addEventListener("pointermove", (ev) => drag && moveTo(toT(ev)));
  const end = () => { if (drag) { drag = null; strip.classList.remove("dragging"); updateWindow(true); loadSuggest(); } };
  strip.addEventListener("pointerup", end);
  strip.addEventListener("pointercancel", end);
  function moveTo(t) {
    const center = t - drag.grab + state.dur / 2;
    const c = clipAt(Math.max(0, Math.min(total - 0.01, center))) || state.clip;
    state.clip = c;
    state.start = Math.round(t - drag.grab - c.offset);
    drawWindow(); // video and estimate follow on pointerup
  }
  durIn.oninput = () => { state.dur = Number(durIn.value); updateWindow(false); };
  durIn.onchange = () => { updateWindow(true); loadSuggest(); };

  async function loadSuggest() {
    const { suggestions } = await api(`/api/drives/${id}/suggest?dur=${state.dur}`);
    sugBox.replaceChildren();
    if (!suggestions.length) {
      sugBox.append(h("span", { class: "muted", text: d.speed.length ? `No ${state.dur} s window holds 10–40 mph here.` : "No telemetry: drag the window over a stretch of steady driving." }));
    }
    for (const sg of suggestions) {
      const on = sg.clip === state.clip.key && sg.start === state.start;
      sugBox.append(h("button", { class: `chip btnchip ${on ? "on" : ""}`, title: `${Math.round(sg.share * 100)}% of the window at 10–40 mph`, onclick: () => {
        state.clip = d.clips.find((c) => c.key === sg.clip); state.start = sg.start; updateWindow(true); loadSuggest();
      } }, icon("spark"), `${clock(sg.offset)}`, h("span", { class: "k", text: `${sg.mean_mph} mph · ${Math.round(sg.share * 100)}%` })));
    }
  }
  let estT;
  function estimate() {
    clearTimeout(estT);
    estT = setTimeout(async () => {
      const q = new URLSearchParams({ cams: state.clip.cams.length, dur: state.dur, fps: fpsIn.value, steps: stepsIn.value });
      const e = await api(`/api/estimate?${q}`);
      estBox.replaceChildren(h("span", { text: "Estimated build" }), h("b", { text: `≈ ${human(e.total)}` }),
        h("span", { text: `· ${e.images} images${e.history.length ? " · from your past builds" : ""}` }));
    }, 150);
  }
  fpsIn.oninput = estimate;
  stepsIn.onchange = estimate;

  buildBtn.onclick = async () => {
    buildBtn.disabled = true;
    try {
      const job = await api("/api/jobs", { json: {
        clip: state.clip.key, start: state.start, dur: state.dur, fps: Number(fpsIn.value), steps: Number(stepsIn.value),
        mask_movers: moversIn.checked, eval: evalIn.checked, name: nameIn.value.trim() || null,
      } });
      refreshBadge();
      location.hash = `#/build/${encodeURIComponent(job.name)}`;
    } catch (e) { toast(e.message, true); buildBtn.disabled = false; }
  };

  const tt = parseTs(d.start);
  view.append(
    h("a", { class: "back", href: "#/drives" }, icon("back"), "Drives"),
    h("div", { class: "page-head" },
      h("div", {}, h("div", { class: "eyebrow", text: `${d.kind.replace("Clips", "")} · ${d.n_clips} clip${d.n_clips === 1 ? "" : "s"} · ${d.n_cams} cameras` }),
        h("h1", {}, `${dateFmt.format(tt)}, ${timeFmt.format(tt)}`),
        h("p", { text: `${human(d.seconds)}${d.mean_mph !== null ? ` · ${Math.round(d.mean_mph)} mph average` : ""}${d.max_mph ? ` · ${Math.round(d.max_mph)} mph max` : ""}` }))),
    h("div", { class: "detail" },
      h("div", {}, player,
        h("div", { class: "card strip-card" },
          h("div", { class: "row", style: { "justify-content": "space-between", "margin-bottom": "8px" } },
            h("h3", { text: "Speed over the drive" }), h("span", { class: "muted", style: { "font-size": "12px" }, text: "drag the blue window · it stays inside one clip" })),
          strip,
          h("div", { class: "legend" }, h("span", {}, h("i", { style: { background: "#3d7bff" } }), "speed (mph)"),
            h("span", {}, h("i", { style: { background: "rgba(62,207,142,.35)" } }), "10–40 mph: reconstructs best"),
            h("span", {}, h("i", { style: { background: "rgba(61,123,255,.4)", border: "1px solid #6b9bff" } }), "build window")))),
      h("aside", { class: "side" },
        h("div", { class: "card pad" },
          h("div", { class: "eyebrow", text: "Window" }), winRead, winSub,
          h("div", { class: "slider-row" }, h("span", { text: "Length" }), durIn, durVal),
          h("div", { class: "eyebrow", style: { "margin-top": "18px" }, text: "Suggested" }), sugBox,
          estBox, buildBtn,
          h("details", { class: "adv" }, h("summary", {}, icon("chev"), "Advanced"),
            h("div", { class: "form" },
              h("div", { class: "field" }, h("label", { text: "Frames per second" }), fpsIn),
              h("div", { class: "field" }, h("label", { text: "Training steps" }), stepsIn),
              h("div", { class: "field full" }, h("label", { text: "City name" }), nameIn),
              h("label", { class: "check full" }, moversIn, h("span", {}, "Mask moving cars and people", h("small", { text: "YOLO; needs `uv sync --extra movers` (~1 GB)" }))),
              h("label", { class: "check full" }, evalIn, h("span", {}, "Score with held-out views", h("small", { text: "keeps every 8th frame out of training to report PSNR" })))))),
        h("div", { class: "card pad" }, h("div", { class: "eyebrow", text: "Route" }), routeSvg(d.route, 320, 140, "mini-route")))));
  updateWindow(true);
  loadSuggest();
}

// ---- Build --------------------------------------------------------------------------------------------

async function pageBuild(view, name) {
  const { jobs } = await api("/api/jobs");
  if (!name && jobs.length) name = (jobs.find((j) => j.state === "running") || jobs[0]).name;
  const queue = h("div", { class: "card queue" });
  const detail = h("div");
  view.append(h("div", { class: "page-head" }, h("div", {}, h("div", { class: "eyebrow", text: "One GPU, one build at a time" }), h("h1", { text: "Build" }))),
    h("div", { class: "build" }, queue, detail));
  const drawQueue = (list) => {
    queue.replaceChildren();
    if (!list.length) queue.append(h("div", { class: "empty", style: { padding: "24px 10px" } }, h("p", { text: "No builds yet." })));
    for (const j of list) {
      const pct = overallPct(j);
      queue.append(h("a", { href: `#/build/${encodeURIComponent(j.name)}`, class: j.name === name ? "on" : "" },
        h("div", { class: "qn", text: j.name }),
        h("div", { class: "qs" }, h("span", { class: `state ${j.state}`, text: j.state }),
          h("span", { text: j.state === "running" ? `${STAGE_LABEL[j.stage] || j.stage || ""} · ${human(j.eta)} left` : j.finished ? timeFmt.format(new Date(j.finished * 1000)) : "" })),
        j.state === "running" || j.state === "queued" ? h("div", { class: "progress" }, h("i", { style: { width: `${pct}%` } })) : null));
    }
  };
  drawQueue(jobs);
  every(3000, async () => drawQueue((await api("/api/jobs")).jobs));
  if (!name) {
    detail.append(h("div", { class: "card empty" }, icon("build"), h("h3", { text: "Nothing building" }),
      h("p", {}, "Pick a drive and a window, then press Build. ", h("a", { class: "btn sm", href: "#/drives" }, "Drives"))));
    return;
  }
  await drawJob(name, detail);
}

let weights = null;
async function stageWeights() {
  if (!weights) {
    try { weights = (await api("/api/estimate")).stages; } catch { weights = Object.fromEntries(STAGES.map((st) => [st, 1])); }
  }
  return weights;
}
function overallPct(j, w = weights) {
  w = w || Object.fromEntries(STAGES.map((st) => [st, 1]));
  const tot = STAGES.reduce((a, st) => a + (w[st] || 1), 0);
  const got = STAGES.reduce((a, st) => a + (w[st] || 1) * Math.min(1, (j.frac || {})[st] || 0), 0);
  return Math.round((100 * got) / tot);
}

async function drawJob(name, box) {
  await Promise.all([stageWeights(), viewerFeatures()]);
  let job = await api(`/api/jobs/${encodeURIComponent(name)}`);
  const clip = job.spec.clip;
  const etaV = h("div", { class: "v" });
  const etaL = h("div", { class: "l" });
  const stateTag = h("span", { class: "state" });
  const actions = h("div", { class: "row" });
  const overallBar = h("i");
  const overallTxt = h("span");
  const elapsedTxt = h("span");
  const steps = Object.fromEntries(STAGES.map((st, i) => {
    const n = h("div", { class: "n", text: i + 1 });
    const t = h("div", { class: "t" });
    const bar = h("i", { style: { width: "0%" } });
    const sub = h("div", { class: "subs" });  // pipeline v2 optional steps seen in this stage (substage events)
    const el = h("div", { class: "step", title: STAGE_HINT[st] }, n, h("div", { class: "lbl", text: STAGE_LABEL[st] }), t, sub, h("div", { class: "progress" }, bar));
    return [st, { el, n, t, bar, sub, subs: [], secs: null }];
  }));
  const errBox = h("div", { class: "error-box", hidden: true });
  const doneBox = h("div", { class: "done-box", hidden: true });
  const log = h("pre", { class: "log", "aria-label": "Build log" });
  const follow = h("input", { type: "checkbox", checked: true });
  const ts = parseTs(clip.timestamp);
  box.replaceChildren(
    h("div", { class: "card" },
      h("div", { class: "job-head" },
        h("div", { class: "grow" }, h("h1", { text: job.name }),
          h("div", { class: "sub" }, stateTag,
            h("span", { text: `${dateFmt.format(ts)} ${timeSecFmt.format(new Date(ts.getTime() + job.spec.start * 1000))} · ${job.spec.dur} s · ${Object.keys(clip.cams).length} cams · ${job.spec.fps} fps · ${Number(job.spec.steps).toLocaleString()} steps` })),
          h("div", { style: { "margin-top": "14px" } }, actions)),
        h("div", { class: "eta-big" }, etaV, etaL)),
      h("div", { class: "overall" }, h("div", { class: "row" }, overallTxt, elapsedTxt), h("div", { class: "progress" }, overallBar)),
      errBox, doneBox,
      h("div", { class: "stepper" }, STAGES.map((st) => steps[st].el))),
    h("div", { class: "card log-card" },
      h("div", { class: "log-head" }, h("h3", { text: "Log" }), h("label", { class: "check muted" }, follow, "Follow")), log));

  const lines = [];
  function addLog(line) {
    lines.push(line);
    if (lines.length > 400) lines.shift();
    const cls = line.startsWith("$ ") ? "cmd" : /error|failed|Traceback/i.test(line) ? "err" : null;
    log.append(cls ? h("span", { class: cls, text: line + "\n" }) : document.createTextNode(line + "\n"));
    while (log.childNodes.length > 400) log.firstChild.remove();
    if (follow.checked) log.scrollTop = log.scrollHeight;
  }
  function paint() {
    stateTag.className = `state ${job.state}`;
    stateTag.textContent = job.state;
    const pct = job.state === "done" ? 100 : overallPct(job);
    overallBar.style.width = `${pct}%`;
    overallTxt.textContent = `${pct}% · ${job.state === "running" ? STAGE_HINT[job.stage] || job.stage || "starting" : job.state}`;
    const el = job.started ? ((job.finished || Date.now() / 1000) - job.started) : 0;
    elapsedTxt.textContent = job.started ? `elapsed ${clock(el)}` : "";
    if (job.state === "running" || job.state === "queued") {
      etaV.textContent = job.state === "queued" ? "Queued" : job.eta !== null && job.eta !== undefined ? `~${human(job.eta)}` : "…";
      etaL.textContent = job.state === "queued" ? "waits for the GPU" : "remaining";
    } else {
      etaV.textContent = job.state === "done" ? clock(el) : "—";
      etaL.textContent = job.state === "done" ? "total time" : job.state;
    }
    const doneSet = new Set(job.stages_done || []);
    STAGES.forEach((st, i) => {
      const x = steps[st];
      const f = Math.min(1, (job.frac || {})[st] || 0);
      const isDone = doneSet.has(st) || f >= 1 && job.stage !== st || job.state === "done";
      const active = !isDone && job.stage === st && job.state === "running";
      const failed = !isDone && job.stage === st && job.state === "failed";
      x.el.className = `step${isDone ? " done" : active ? " active" : failed ? " failed" : ""}`;
      x.n.replaceChildren(isDone ? icon("check") : failed ? icon("alert") : String(i + 1));
      x.bar.style.width = `${(isDone ? 1 : f) * 100}%`;
      x.t.textContent = isDone ? (x.secs !== null ? clock(x.secs) : "done") : active ? (job.stage_eta !== null && job.stage_eta !== undefined && f > 0 ? `${Math.round(f * 100)}% · ${human(job.stage_eta)}` : f > 0 ? `${Math.round(f * 100)}%` : "running") : "";
    });
    errBox.hidden = !["failed", "interrupted", "orphaned-running"].includes(job.state) || !job.error;
    errBox.textContent = job.error || "";
    doneBox.hidden = job.state !== "done";
    if (job.state === "done") {
      doneBox.replaceChildren(h("div", {}, h("b", { text: "Your city is ready." }), h("div", { class: "muted", text: "Drive through it, or replay the chase cam." })),
        h("div", { class: "row" },
          driveButton(job.name, viewerInfo.drive),
          h("a", { class: "btn", href: cityUrl(job.name), target: "_blank", rel: "noopener" }, icon("replay"), "Replay")));
    }
    actions.replaceChildren();
    if (job.state === "running" || job.state === "queued") {
      actions.append(h("button", { class: "sm danger", onclick: async () => {
        if (!(await confirmDialog("Cancel this build?", "The current step stops (SIGTERM, then SIGKILL after 5 s). Finished stages are kept, so you can resume.", "Cancel build"))) return;
        job = await api(`/api/jobs/${encodeURIComponent(job.name)}/cancel`, { method: "POST" }); paint();
      } }, icon("x"), "Cancel"));
    }
    if (job.state === "orphaned-running") {
      actions.append(h("button", { class: "sm danger", onclick: async () => {
        if (!(await confirmDialog("Kill the leftover build?", `An earlier studio session left this build running (process group ${job.pgid}). Kill it (SIGTERM, then SIGKILL after 5 s) so it can be resumed here.`, "Kill build"))) return;
        try {
          job = await api(`/api/jobs/${encodeURIComponent(job.name)}/kill`, { method: "POST" });
          for (let i = 0; i < 20 && job.state === "orphaned-running"; i++) {  // SIGKILL comes after 5 s at most
            await new Promise((r) => setTimeout(r, 500));
            job = await api(`/api/jobs/${encodeURIComponent(job.name)}`);
          }
          paint();
        } catch (e) { toast(e.message, true); }
      } }, icon("x"), "Kill"));
    }
    if (["failed", "cancelled", "interrupted"].includes(job.state)) {
      actions.append(h("button", { class: "sm primary", onclick: async () => {
        try { job = await api(`/api/jobs/${encodeURIComponent(job.name)}/resume`, { method: "POST" }); paint(); connect(); } catch (e) { toast(e.message, true); }
      } }, icon("refresh"), "Resume"), h("span", { class: "muted", style: { "font-size": "12px" }, text: (job.stages_done || []).length ? "finished steps are skipped" : "" }));
    }
  }
  paint();

  let es = null;
  function connect() {
    if (es) es.close();
    log.replaceChildren();
    lines.length = 0;
    es = new EventSource(`/api/jobs/${encodeURIComponent(name)}/events`);
    es.onmessage = (m) => {
      const ev = JSON.parse(m.data);
      if (ev.type === "log") return addLog(ev.line);
      if (ev.type === "state") {
        job.state = ev.state;
        if (ev.error) job.error = ev.error;
        if (ev.state === "running" && !job.started) job.started = ev.t;
        if (["done", "failed", "cancelled", "interrupted"].includes(ev.state)) job.finished = ev.t;
        if (ev.state === "queued") { job.finished = null; job.error = null; }
      } else if (ev.type === "stage") {
        job.stage = ev.stage;
        job.frac = job.frac || {};
        if (ev.status === "done") {
          job.frac[ev.stage] = 1;
          job.stages_done = [...new Set([...(job.stages_done || []), ev.stage])];
          if (steps[ev.stage]) steps[ev.stage].secs = ev.seconds;  // unknown stage names are tolerated
        } else {
          job.frac[ev.stage] = job.frac[ev.stage] && job.frac[ev.stage] < 1 ? job.frac[ev.stage] : 0;
          job.stage_eta = null;
        }
        if (ev.eta !== undefined) job.eta = ev.eta;
      } else if (ev.type === "progress") {
        job.stage = ev.stage;
        job.frac = { ...(job.frac || {}), [ev.stage]: ev.frac };
        job.stage_eta = ev.stage_eta;
        job.eta = ev.eta;
      } else if (ev.type === "substage") {
        const x = steps[ev.stage] || steps[job.stage];
        const lbl = ev.label || ev.substage;
        if (!x) addLog(`(${lbl})`);
        else if (!x.subs.includes(lbl)) { x.subs.push(lbl); x.sub.textContent = x.subs.join(" · "); }
      } else if (ev.type === "eta") {
        job.eta = ev.eta;
      }
      paint();
    };
    es.addEventListener("end", () => { es.close(); refreshBadge(); });
  }
  connect();
  const tick = setInterval(paint, 1000);
  cleanup.push(() => { if (es) es.close(); clearInterval(tick); });
}

// ---- Gallery ------------------------------------------------------------------------------------------

async function pageGallery(view) {
  const [{ cities }, { drive }] = await Promise.all([api("/api/cities"), viewerFeatures()]);
  const grid = h("div", { class: "grid" });
  view.append(h("div", { class: "page-head" },
    h("div", {}, h("div", { class: "eyebrow", text: "Built on this Mac" }), h("h1", { text: "Gallery" }),
      h("p", { text: `${cities.length} cit${cities.length === 1 ? "y" : "ies"} · ${bytes(cities.reduce((a, c) => a + c.disk_bytes, 0))} on disk` }))), grid);
  if (!cities.length) {
    grid.append(h("div", { class: "card empty", style: { "grid-column": "1 / -1" } }, icon("map"),
      h("h3", { text: "No cities yet" }), h("p", {}, "Pick a drive, choose a window and build one. ", h("a", { class: "btn sm", href: "#/drives" }, "Drives"))));
  }
  for (const c of cities) {
    const ts = c.timestamp ? parseTs(c.timestamp) : null;
    const when = ts ? `${dateFmt.format(ts)}, ${timeFmt.format(new Date(ts.getTime() + (c.start || 0) * 1000))}` : "—";
    const card = h("article", { class: "card city" },
      h("div", { class: "poster" },
        c.has_poster ? h("img", { src: `/api/cities/${encodeURIComponent(c.name)}/poster.jpg`, alt: "", loading: "lazy" }) : null,
        h("span", { class: "name", text: c.name }),
        h("span", { class: "psnr" }, c.psnr !== null && c.psnr !== undefined
          ? h("span", { class: `psnr-badge ${c.psnr >= 22 ? "good" : ""}`, title: "held-out PSNR from otrv bench", text: `PSNR ${c.psnr.toFixed(1)} dB` })
          : h("span", { class: "psnr-badge", title: "build with “Score with held-out views” to get PSNR", text: "no score" }))),
      h("div", { class: "city-body" },
        h("div", { class: "city-meta" },
          h("div", {}, h("b", { text: when }), "recorded"),
          h("div", {}, h("b", { text: c.dur ? `${c.dur} s · ${c.cams.length} cams` : "—" }), "window"),
          h("div", {}, h("b", { text: bytes(c.splat_bytes) }), "splat"),
          h("div", {}, h("b", { text: bytes(c.disk_bytes) }), "on disk")),
        h("div", { class: "city-actions" },
          driveButton(c.name, drive),
          h("a", { class: "btn", href: cityUrl(c.name), target: "_blank", rel: "noopener" }, icon("replay"), "Replay"),
          h("button", { class: "danger", title: "Delete this city", "aria-label": `Delete ${c.name}`, onclick: async () => {
            if (!(await confirmDialog(`Delete ${c.name}?`, `Removes work/${c.name} (${bytes(c.disk_bytes)}): the splat, frames and models. Your footage is not touched.`))) return;
            try { await api(`/api/cities/${encodeURIComponent(c.name)}`, { method: "DELETE" }); card.remove(); toast(`Deleted ${c.name}`); } catch (e) { toast(e.message, true); }
          } }, icon("trash")))));
    grid.append(card);
  }
}

refreshBadge();
render();
