// Built-in ego car v3: an ORIGINAL white crossover built entirely in code, lofted to the public
// Model Y dimensions (length 4.751 m, width 1.921 m without mirrors, height 1.624 m, wheelbase
// 2.890 m, ground clearance ~0.167 m, overhangs ~0.92 m front / ~0.94 m rear, ~0.73 m wheels on a
// ~1.636 m track). Geometry, lamps and wheels are our own design: no logos, no badges, no copied
// surfaces.
//
//   buildCar(THREE, { paint }) -> { group, body, setBrake(on), setBlink(left, right), setSteer(rad),
//                                   setWheelSpin(rad), setHeadlights(on) }
//
// Local axes: +z forward, +y up, +x = the car's left; metres; origin on the road at the middle of the
// wheelbase. `body` is the sprung part (drive mode pitches/rolls it); wheels stay in `group`.
//
// Body: 18 cross-sections from nose to tail. Each section is a two-part superellipse: a lower half
// (rocker / bumper, exponent NL) below the shoulder line ys, and an upper half (exponent nU) that
// narrows from the shoulder half-width ws to the roof half-width wr (tumblehome). The section
// parameters are Catmull-Rom interpolated along the length and lofted into one smooth indexed
// surface. Wheel wells are pressed into that surface. Paint, black cladding and the dark greenhouse
// are per-vertex colours with smoothstep edges (no stair-stepping), and the greenhouse is pushed a
// few millimetres proud of the body. Lamps are thin masked patches of the same surface.

export const DIMS = { length: 4.751, width: 1.921, height: 1.624, wheelbase: 2.89, track: 1.636, wheelR: 0.365, frontOH: 0.92, rearOH: 0.94 };

//                z      yb     ys     yt     ws      wr     nU
const STATIONS = [
  [2.37, 0.36, 0.56, 0.74, 0.50, 0.40, 3.0],     // nose: rounded, no grille
  [2.355, 0.29, 0.57, 0.795, 0.74, 0.56, 3.2],
  [2.31, 0.25, 0.59, 0.835, 0.86, 0.64, 3.4],
  [2.20, 0.225, 0.62, 0.86, 0.925, 0.69, 3.8],
  [1.95, 0.21, 0.66, 0.885, 0.952, 0.71, 4.2],
  [1.445, 0.20, 0.70, 0.915, 0.9605, 0.72, 4.6],  // front axle: short, gently sloped hood
  [1.08, 0.185, 0.73, 0.975, 0.955, 0.72, 4.8],   // cowl / windshield base
  [0.68, 0.172, 0.75, 1.245, 0.952, 0.70, 5.0],   // long raked windshield
  [0.22, 0.167, 0.765, 1.522, 0.952, 0.665, 5.2],
  [-0.30, 0.167, 0.775, 1.624, 0.952, 0.645, 5.4], // roof crown (continuous glass roof)
  [-0.95, 0.17, 0.785, 1.605, 0.955, 0.64, 5.4],
  [-1.445, 0.20, 0.79, 1.548, 0.9605, 0.625, 5.2], // rear axle: strong shoulder, roof carried back
  [-1.80, 0.25, 0.80, 1.44, 0.948, 0.605, 4.8],
  [-2.02, 0.285, 0.80, 1.30, 0.925, 0.59, 4.5],    // hatch glass, steeper toward the tail
  [-2.14, 0.31, 0.80, 1.185, 0.895, 0.60, 4.2],    // rear glass ends
  [-2.22, 0.33, 0.80, 1.172, 0.865, 0.62, 4.0],    // integrated ducktail lip
  [-2.31, 0.36, 0.79, 1.10, 0.81, 0.62, 3.6],
  [-2.37, 0.40, 0.77, 1.02, 0.68, 0.54, 3.2],
  [-2.385, 0.48, 0.74, 0.93, 0.45, 0.38, 3.0],     // tail
];
const NL = 3.0;
const SEG_PER_STATION = 16, N_LOW = 40, N_UP = 76;
const ARCH_R = 0.405, ARCH_IN = 0.70;  // wheel-well radius and inner wall (half-width)
const COLORS = { trim: "#141619", glass: "#0a0c10" };

export function buildCar(THREE, { paint = "#f2f3f5" } = {}) {
  const lerp = (a, b, t) => a + (b - a) * t;
  const clamp01 = (x) => Math.max(0, Math.min(1, x));
  const ss = (d, w) => { const t = clamp01((d + w) / (2 * w)); return t * t * (3 - 2 * t); }; // soft "d > 0"
  const cr = (p0, p1, p2, p3, t) => { const t2 = t * t, t3 = t2 * t; return 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3); };
  const S = STATIONS, NS = S.length;
  function params(u) {
    const i = Math.min(NS - 2, Math.max(0, Math.floor(u))), t = u - i;
    const g = (k) => cr(S[Math.max(0, i - 1)][k], S[i][k], S[i + 1][k], S[Math.min(NS - 1, i + 2)][k], t);
    return { z: g(0), yb: g(1), ys: g(2), yt: g(3), ws: g(4), wr: g(5), nU: g(6) };
  }
  function uAtZ(z) { // inverse of params(u).z (z decreases with u)
    let lo = 0, hi = NS - 1;
    for (let it = 0; it < 32; it++) { const m = (lo + hi) / 2; params(m).z > z ? (lo = m) : (hi = m); }
    return (lo + hi) / 2;
  }
  // right-hand half of a section; theta in [-pi/2, pi/2] runs from bottom centre over the side to the roof centre
  function sectionPoint(p, th) {
    const c = Math.cos(th), s = Math.sin(th);
    if (th < 0) { const e = 2 / NL; return [p.ws * Math.abs(c) ** e, p.ys - (p.ys - p.yb) * Math.abs(s) ** e]; }
    const e = 2 / p.nU, w = p.ws + (p.wr - p.ws) * s ** 1.35;
    return [w * c ** e, p.ys + (p.yt - p.ys) * s ** e];
  }
  const archDist = (z, y) => Math.min(Math.hypot(z - DIMS.wheelbase / 2, y - DIMS.wheelR), Math.hypot(z + DIMS.wheelbase / 2, y - DIMS.wheelR));
  const beltY = (z) => 0.995 + 0.085 * ss(-1.2 - z, 0.45);            // window line, kicking up at the rear quarter
  const zC = (y) => -1.70 + (y - 1.08) * 0.85;                         // C-pillar: rear edge of the side glass
  // ---- grid ----
  const nu = (NS - 1) * SEG_PER_STATION + 1;
  const ring = [];
  for (let k = 0; k <= N_LOW; k++) ring.push(-Math.PI / 2 + (Math.PI / 2) * (k / N_LOW) ** 0.9);
  for (let k = 1; k <= N_UP; k++) ring.push((Math.PI / 2) * (k / N_UP) ** 1.1);
  const half = ring.length, J = 2 * (half - 1);
  const NV = nu * J;
  const P = new Float32Array((NV + 2) * 3), C = new Float32Array((NV + 2) * 3), Gw = new Float32Array(NV + 2), Tw = new Float32Array(NV + 2);
  const cPaint = new THREE.Color(paint).convertSRGBToLinear(), cTrim = new THREE.Color(COLORS.trim).convertSRGBToLinear(), cGlass = new THREE.Color(COLORS.glass).convertSRGBToLinear();
  const tmp = new THREE.Color();
  for (let i = 0; i < nu; i++) {
    const p = params(i / SEG_PER_STATION);
    for (let j = 0; j < J; j++) {
      const k = j < half ? j : J - j, side = j < half ? -1 : 1;
      let [x, y] = sectionPoint(p, ring[k]);
      x *= side;
      const ad = archDist(p.z, y);
      if (ad < ARCH_R && Math.abs(x) > ARCH_IN) x = Math.sign(x) * lerp(Math.abs(x), ARCH_IN, ss(ARCH_R - ad - 0.025, 0.025));
      const v = i * J + j;
      P.set([x, y, p.z], v * 3);
      // trim: flared arch surrounds + wells, lower cladding all round (diffuser at the rear)
      const flare = ss(ARCH_R + 0.058 - ad, 0.014) * ss(y - 0.03, 0.01);
      const cladY = 0.43 + (0.31 - 0.43) * ss(p.z - 2.0, 0.06) + (0.415 - 0.43) * ss(-2.05 - p.z, 0.06);
      const trim = Math.max(flare, ss(cladY - y, 0.014));
      // greenhouse: windshield + glass roof + rear glass as one band framed by painted roof rails,
      // side glass back to the C-pillar, black pillars
      const ax = Math.abs(x);
      const inZ = ss(1.12 - p.z, 0.025) * ss(p.z + 2.12, 0.025);
      const roofBand = ss(p.wr * 0.84 - ax, 0.018) * ss(y - (p.yt - 0.32), 0.02);
      const sideWin = ss(p.yt - 0.072 - y, 0.014) * ss(p.z - zC(y), 0.025);
      const glass = ss(y - beltY(p.z), 0.014) * inZ * Math.max(roofBand, sideWin);
      Gw[v] = glass; Tw[v] = trim * (1 - glass);
      tmp.copy(cPaint).lerp(cTrim, trim).lerp(cGlass, glass);
      C.set([tmp.r, tmp.g, tmp.b], v * 3);
    }
  }
  const idx = (i, j) => i * J + ((j + J) % J);
  // end caps: fan to each end section's centre
  for (const [n, i] of [[0, 0], [1, nu - 1]]) {
    let cx = 0, cy = 0, cz = 0;
    for (let j = 0; j < J; j++) { cx += P[idx(i, j) * 3]; cy += P[idx(i, j) * 3 + 1]; cz += P[idx(i, j) * 3 + 2]; }
    P.set([cx / J, cy / J, cz / J + (i === 0 ? 0.003 : -0.003)], (NV + n) * 3);
    C.set([cPaint.r, cPaint.g, cPaint.b], (NV + n) * 3);
  }
  const quads = [], ind = [];
  for (let i = 0; i < nu - 1; i++) for (let j = 0; j < J; j++) {
    const a = idx(i, j), b = idx(i + 1, j), c = idx(i + 1, j + 1), d = idx(i, j + 1);
    ind.push(a, c, b, a, d, c);
    let cx = 0, cy = 0, cz = 0;
    for (const v of [a, b, c, d]) { cx += P[v * 3] / 4; cy += P[v * 3 + 1] / 4; cz += P[v * 3 + 2] / 4; }
    quads.push({ a, b, c, d, x: cx, y: cy, z: cz });
  }
  for (let j = 0; j < J; j++) { ind.push(NV, idx(0, j + 1), idx(0, j)); ind.push(NV + 1, idx(nu - 1, j), idx(nu - 1, j + 1)); }
  const body = new THREE.BufferGeometry();
  body.setAttribute("position", new THREE.BufferAttribute(P, 3));
  body.setAttribute("color", new THREE.BufferAttribute(C, 3));
  body.setAttribute("finish", new THREE.BufferAttribute(new Float32Array([...Array(NV + 2)].flatMap((_, v) => [Tw[v], Gw[v]])), 2));
  body.setIndex(ind);
  body.computeVertexNormals();
  // greenhouse sits ~6 mm proud of the body; recompute normals so its edge catches a highlight
  {
    const N0 = body.attributes.normal.array;
    for (let v = 0; v < NV; v++) for (let a = 0; a < 3; a++) P[v * 3 + a] += N0[v * 3 + a] * 0.006 * Gw[v];
    body.attributes.position.needsUpdate = true;
    body.computeVertexNormals();
  }
  const N = body.attributes.normal.array;
  // overlay: a masked patch of the body surface pushed out along the normal
  function overlay(mask, offset) {
    const remap = new Map(), pos = [], nor = [], ix = [];
    const vid = (v) => {
      if (!remap.has(v)) {
        remap.set(v, pos.length / 3);
        pos.push(P[v * 3] + N[v * 3] * offset, P[v * 3 + 1] + N[v * 3 + 1] * offset, P[v * 3 + 2] + N[v * 3 + 2] * offset);
        nor.push(N[v * 3], N[v * 3 + 1], N[v * 3 + 2]);
      }
      return remap.get(v);
    };
    for (const q of quads) if (mask(q)) { const a = vid(q.a), b = vid(q.b), c = vid(q.c), d = vid(q.d); ix.push(a, c, b, a, d, c); }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute("normal", new THREE.Float32BufferAttribute(nor, 3));
    g.setIndex(ix);
    return g;
  }
  // lamps: slim swept headlight strips; rear light bar + C-shaped tail lamps
  const ax = (q) => Math.abs(q.x);
  const headY = (q) => 0.705 + 0.055 * (ax(q) - 0.36);
  const head = (q) => q.z > 2.0 && ax(q) > 0.36 && ax(q) < 0.9 && Math.abs(q.y - headY(q)) < 0.022;
  const drl = (q) => q.z > 2.0 && ax(q) > 0.40 && ax(q) < 0.86 && Math.abs(q.y - headY(q) - 0.008) < 0.0065;
  const amberF = (q) => head(q) && ax(q) > 0.80;
  const rearFace = (q) => q.z < -2.08;
  const bar = (q) => rearFace(q) && ax(q) < 0.62 && Math.abs(q.y - 1.035) < 0.009;
  const cLamp = (q) => rearFace(q) && ax(q) >= 0.56 && (
    Math.abs(q.y - 1.035) < 0.022 ||                                   // upper arm
    (ax(q) > 0.79 && q.y > 0.935 && q.y < 1.055) ||                     // outer spine, wrapping the corner
    (Math.abs(q.y - 0.95) < 0.013 && ax(q) < 0.86));                   // lower arm
  const amberR = (q) => rearFace(q) && ax(q) >= 0.58 && ax(q) < 0.79 && Math.abs(q.y - 0.95) < 0.013;
  // ---- materials ----
  const mats = {
    body: new THREE.MeshPhysicalMaterial({ color: 0xffffff, vertexColors: true, roughness: 0.22, metalness: 0, clearcoat: 1, clearcoatRoughness: 0.06 }),
    paint: new THREE.MeshPhysicalMaterial({ color: paint, roughness: 0.22, metalness: 0, clearcoat: 1, clearcoatRoughness: 0.06 }),
    trim: new THREE.MeshStandardMaterial({ color: COLORS.trim, roughness: 0.62, metalness: 0.05 }),
    seam: new THREE.MeshBasicMaterial({ color: 0x3a3d42, side: THREE.DoubleSide }),
    lens: new THREE.MeshPhysicalMaterial({ color: 0x0c0e12, roughness: 0.1, metalness: 0.3, clearcoat: 1 }),
    drl: new THREE.MeshBasicMaterial({ color: 0xeef4ff, toneMapped: false }),
    tail: new THREE.MeshStandardMaterial({ color: 0x1a0204, emissive: 0xff0a14, emissiveIntensity: 0.55, roughness: 0.5, toneMapped: false }),
    amberL: [], amberR: [],
    tyre: new THREE.MeshStandardMaterial({ color: 0x151515, roughness: 0.9 }),
    rim: new THREE.MeshStandardMaterial({ color: 0x3a3e44, metalness: 0.65, roughness: 0.36 }),
    spoke: new THREE.MeshStandardMaterial({ color: 0x24272b, metalness: 0.55, roughness: 0.45 }),
  };
  const amberMat = () => new THREE.MeshStandardMaterial({ color: 0x141518, emissive: 0xff9a10, emissiveIntensity: 0, roughness: 0.3, toneMapped: false });
  // one body material; a per-vertex "finish" (trim, glass) makes the cladding matte (no clearcoat)
  // and tames the glass reflections, so the boundaries can blend smoothly instead of stair-stepping
  mats.body.onBeforeCompile = (sh) => {
    sh.vertexShader = sh.vertexShader.replace("#include <color_pars_vertex>", "#include <color_pars_vertex>\nattribute vec2 finish;\nvarying vec2 vFinish;")
      .replace("#include <color_vertex>", "#include <color_vertex>\nvFinish = finish;");
    sh.fragmentShader = sh.fragmentShader.replace("#include <color_pars_fragment>", "#include <color_pars_fragment>\nvarying vec2 vFinish;")
      .replace("#include <roughnessmap_fragment>", "#include <roughnessmap_fragment>\nroughnessFactor = mix(roughnessFactor, 0.68, vFinish.x);\nroughnessFactor = mix(roughnessFactor, 0.12, vFinish.y);")
      .replace("#include <lights_physical_fragment>", "#include <lights_physical_fragment>\nmaterial.clearcoat *= (1.0 - vFinish.x) * (1.0 - 0.6 * vFinish.y);")
      .replace("#include <aomap_fragment>", "reflectedLight.indirectSpecular *= mix(1.0, 0.45, vFinish.y);\n#include <aomap_fragment>");
  };
  const group = new THREE.Group();
  const bodyGroup = new THREE.Group();
  group.add(bodyGroup);
  bodyGroup.add(new THREE.Mesh(body, mats.body));
  bodyGroup.add(new THREE.Mesh(overlay(head, 0.004), mats.lens));
  bodyGroup.add(new THREE.Mesh(overlay((q) => drl(q) && !amberF(q), 0.007), mats.drl));
  bodyGroup.add(new THREE.Mesh(overlay((q) => (bar(q) || cLamp(q)) && !amberR(q), 0.005), mats.tail));
  for (const [maskFn, list] of [[amberF, "F"], [amberR, "R"]]) for (const sideSign of [1, -1]) {
    const m = amberMat();
    bodyGroup.add(new THREE.Mesh(overlay((q) => maskFn(q) && Math.sign(q.x) === sideSign, 0.008), m));
    (sideSign > 0 ? mats.amberL : mats.amberR).push(m);
    void list;
  }
  // door-line hints: thin dark seams that follow the side surface
  function seam(z0, y0, y1) {
    const p = params(uAtZ(z0)), pos = [], ix = [];
    for (const side of [1, -1]) {
      const base = pos.length / 3;
      let n = 0;
      for (let k = 0; k <= 400; k++) {
        const th = -Math.PI / 2 + (Math.PI * k) / 400;
        const [x, y] = sectionPoint(p, th);
        if (y < y0 || y > y1) continue;
        const X = side * (x + 0.0015);
        pos.push(X, y, z0 + 0.0035, X, y, z0 - 0.0035);
        if (n) { const a = base + (n - 1) * 2; ix.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
        n++;
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.setIndex(ix);
    return new THREE.Mesh(g, mats.seam);
  }
  bodyGroup.add(seam(0.9, 0.47, beltY(0.9) - 0.004), seam(-0.1, 0.45, beltY(-0.1) - 0.004), seam(-1.0, 0.47, beltY(-1.0) - 0.004));
  // side mirrors: painted pods on black stalks, just aft of the A-pillar base
  for (const s of [1, -1]) {
    const pod = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16), mats.paint);
    pod.scale.set(0.06, 0.07, 0.125);
    pod.position.set(s * 1.005, 1.05, 0.86);
    pod.rotation.y = s * -0.2;
    const stalk = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.035, 0.07), mats.trim);
    stalk.position.set(s * 0.935, 1.03, 0.88);
    bodyGroup.add(pod, stalk);
  }
  // ---- wheels: tyre + aero (turbine) cover with shallow spokes; the front pair steers, all spin ----
  const wheels = [], front = [];
  function makeWheel(side) {
    const r = DIMS.wheelR, w = 0.255, prof = [];
    for (let i = 0; i <= 14; i++) { const t = (i / 14) * Math.PI; prof.push(new THREE.Vector2(r - 0.06 + 0.06 * Math.sin(t), -w / 2 + (w / 2) * (1 - Math.cos(t)))); }
    prof.unshift(new THREE.Vector2(0.27, -w / 2 + 0.01)); prof.push(new THREE.Vector2(0.27, w / 2 - 0.01));
    const tyre = new THREE.Mesh(new THREE.LatheGeometry(prof, 48), mats.tyre);
    tyre.rotation.z = Math.PI / 2;
    const spin = new THREE.Group();
    spin.add(tyre);
    const face = side * (w / 2 - 0.02);
    const disc = new THREE.Mesh(new THREE.CylinderGeometry(0.285, 0.29, 0.02, 48), mats.rim);
    disc.rotation.z = Math.PI / 2; disc.position.x = face;
    spin.add(disc);
    for (let k = 0; k < 6; k++) { // shallow curved spokes, slightly twisted (turbine look)
      const spoke = new THREE.Mesh(new THREE.BoxGeometry(0.012, 0.045, 0.21), mats.spoke);
      const a = (k / 6) * Math.PI * 2;
      spoke.position.set(face + side * 0.008, Math.sin(a) * 0.155, Math.cos(a) * 0.155);
      spoke.rotation.set(-a + 0.38, 0, 0);
      spin.add(spoke);
    }
    const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 0.03, 24), mats.spoke);
    hub.rotation.z = Math.PI / 2; hub.position.x = face + side * 0.01;
    spin.add(hub);
    const pivot = new THREE.Group();
    pivot.add(spin);
    pivot.userData.spin = spin;
    return pivot;
  }
  for (const z of [DIMS.wheelbase / 2, -DIMS.wheelbase / 2]) for (const s of [1, -1]) {
    const wheel = makeWheel(s);
    wheel.position.set(s * (DIMS.track / 2 - 0.012), DIMS.wheelR, z);
    group.add(wheel);
    wheels.push(wheel);
    if (z > 0) front.push(wheel);
  }
  return {
    group,
    body: bodyGroup,
    setBrake(on) { mats.tail.emissiveIntensity = on ? 1.6 : 0.55; },
    setBlink(left, right) {
      for (const m of mats.amberL) m.emissiveIntensity = left ? 1.6 : 0;
      for (const m of mats.amberR) m.emissiveIntensity = right ? 1.6 : 0;
    },
    setSteer(rad) { for (const w of front) w.rotation.y = rad; },
    setWheelSpin(rad) { for (const w of wheels) w.userData.spin.rotation.x = rad; },
    setHeadlights(on) { mats.drl.color.setScalar(on ? 1.25 : 0.93); },
    dims: { wheelbase: DIMS.wheelbase, wheelR: DIMS.wheelR, length: DIMS.length },
  };
}

// ---- user model (models/car.json + a glTF), see viewer/models/README.md ----
//   fitModelCar(THREE, root, cfg) -> same API as buildCar, plus { meshes, wheelsFound }
// root is the loaded gltf.scene. cfg = car.json: { file, lengthM, paint, credit, source: { axes } }.
//   axes     "+Y up, -Z forward" style string (cfg.axes or cfg.source.axes). Without it the nose is
//            guessed from where the head lamps sit relative to the tail lamps.
//   lengthM  overall length in metres; the model is scaled uniformly to it.
//   paint    body colour; materials named like "exterior_paint" (or paint / body colour / carpaint)
//            become a pearl-white clearcoat in that colour.
// Lamps by material name: taillight / tail light / rear lamp brighten on brake; signature / headlight /
// DRL brighten with the headlights. The outer left / right parts of both (|x| > 0.42 m) flash amber as
// blinkers (triangles are split per side into their own material groups, so one mesh spanning the
// whole width still blinks per side). Wheels: tyre meshes (tire / tyre) mark the four wheel centres;
// wheel, rim, rotor and caliper meshes near a centre are re-parented to a pivot there, so the fronts
// steer and all four spin (calipers steer but don't spin).
const AXIS_RE = (word) => new RegExp(`([+-])\\s*([xyz])\\s*${word}`, "i");
const PAINT_RE = /exterior_paint|paint|body.?colou?r|carpaint/i;
const NOT_PAINT_RE = /glass|window|lamp|light|lens|led|tire|tyre|wheel|rim|brake|caliper|rotor|interior|leather|seat|trim|rubber/i;
const TAIL_RE = /taillight|tail.?light|rear.?lamp|brake.?light/i;
const HEAD_RE = /signature|headlight|head.?lamp|front.?light|\bdrl\b/i;
const TYRE_RE = /tire|tyre/i;
const WHEEL_RE = /wheel|rim|tire|tyre|brake|caliper|rotor|hub|spoke/i;

export function fitModelCar(THREE, root, cfg = {}) {
  const label = (o, m) => `${o.name} ${m?.name ?? ""}`;
  const matsOf = (o) => (Array.isArray(o.material) ? o.material : [o.material]);
  const meshes = [];
  root.traverse((o) => { if (o.isMesh) meshes.push(o); });
  const boxOf = (o) => new THREE.Box3().setFromObject(o);
  // ---- orientation: model axes -> car-local (+x left, +y up, +z forward) ----
  const axesStr = cfg.axes ?? cfg.source?.axes ?? "";
  const parseAxis = (word) => {
    const m = AXIS_RE(word).exec(axesStr);
    if (!m) return null;
    const v = new THREE.Vector3();
    v.setComponent("xyz".indexOf(m[2].toLowerCase()), m[1] === "-" ? -1 : 1);
    return v;
  };
  const up = parseAxis("up") ?? new THREE.Vector3(0, 1, 0);
  let fwd = parseAxis("forward") ?? parseAxis("front");
  root.updateMatrixWorld(true);
  if (!fwd) { // guess: the nose is where the head lamps are (vs the tail lamps), along the longest horizontal axis
    const size = boxOf(root).getSize(new THREE.Vector3());
    const ax = Math.abs(up.x) > 0.5 ? (size.y > size.z ? 1 : 2) : Math.abs(up.y) > 0.5 ? (size.x > size.z ? 0 : 2) : (size.x > size.y ? 0 : 1);
    let f = 0, nf = 0, r = 0, nr = 0;
    for (const o of meshes) for (const m of matsOf(o)) {
      const c = boxOf(o).getCenter(new THREE.Vector3()).getComponent(ax);
      if (TAIL_RE.test(label(o, m))) { r += c; nr++; } else if (HEAD_RE.test(label(o, m))) { f += c; nf++; }
    }
    fwd = new THREE.Vector3();
    fwd.setComponent(ax, nf && nr && f / nf < r / nr ? -1 : 1);
  }
  const left = new THREE.Vector3().crossVectors(up, fwd).normalize();
  const orient = new THREE.Group();
  orient.quaternion.setFromRotationMatrix(new THREE.Matrix4().set(
    left.x, left.y, left.z, 0, up.x, up.y, up.z, 0, fwd.x, fwd.y, fwd.z, 0, 0, 0, 0, 1));
  orient.add(root);
  // ---- scale to lengthM, then put the wheelbase middle at the origin with the tyres on y = 0 ----
  const fit = new THREE.Group();
  fit.add(orient);
  fit.updateMatrixWorld(true);
  const measured = Math.max(1e-6, boxOf(fit).getSize(new THREE.Vector3()).z);
  fit.scale.setScalar(Number(cfg.lengthM) > 0 ? Number(cfg.lengthM) / measured : 1);
  fit.updateMatrixWorld(true);
  const tyres = meshes.filter((o) => matsOf(o).some((m) => TYRE_RE.test(label(o, m))));
  {
    const all = boxOf(fit), tb = tyres.map(boxOf), cz = tb.map((b) => (b.min.z + b.max.z) / 2);
    const ground = tb.length ? Math.min(...tb.map((b) => b.min.y)) : all.min.y;
    const zMid = cz.length >= 2 ? (Math.max(...cz) + Math.min(...cz)) / 2 : (all.min.z + all.max.z) / 2;
    fit.position.set(-(all.min.x + all.max.x) / 2, -ground, -zMid);
  }
  const group = new THREE.Group(), body = new THREE.Group();
  group.add(body);
  body.add(fit);
  group.updateMatrixWorld(true);
  const all = boxOf(fit);
  const tyreBoxes = tyres.map(boxOf);
  const centres = tyreBoxes.map((b) => b.getCenter(new THREE.Vector3()));
  // ---- materials: paint, lamps (with per-side blinker groups), shadows ----
  const paint = new THREE.Color(cfg.paint || "#f2f3f5");
  const amber = new THREE.Color(1.0, 0.36, 0.0);
  const clones = new Map();
  const lamp = { tail: [], head: [], blinkL: [], blinkR: [] };
  const remat = (m, make) => { if (!clones.has(m)) clones.set(m, make(m)); return clones.get(m); };
  const paintMat = (m) => {
    const p = new THREE.MeshPhysicalMaterial({ name: m.name, color: paint, metalness: 0.02, roughness: 0.3, clearcoat: 1, clearcoatRoughness: 0.035,
      sheen: 0.35, sheenColor: new THREE.Color(0.92, 0.95, 1.0), sheenRoughness: 0.45, side: m.side, envMapIntensity: 1.1 });
    return p;
  };
  const lampMat = (m, kind) => {
    const c = m.clone();
    c.toneMapped = false;
    c.userData = { kind, base: c.emissive?.clone() ?? new THREE.Color(), baseI: c.emissiveIntensity ?? 1 };
    return c;
  };
  const SIDE = 0.42;
  const _p = new THREE.Vector3(), _q = new THREE.Vector3(), _r = new THREE.Vector3();
  for (const o of meshes) {
    o.castShadow = o.receiveShadow = false;
    const mats = matsOf(o);
    if (mats.length !== 1 || !mats[0]) continue;
    const m = mats[0], lb = label(o, m);
    if (PAINT_RE.test(lb) && !NOT_PAINT_RE.test(lb)) { o.material = remat(m, paintMat); continue; }
    const kind = TAIL_RE.test(lb) ? "tail" : HEAD_RE.test(lb) ? "head" : null;
    if (!kind) continue;
    const base = lampMat(m, kind);
    lamp[kind].push(base);
    // split triangles into centre / left / right material groups by their car-local x
    const g = o.geometry.index ? o.geometry.clone() : o.geometry.clone().setIndex([...Array(o.geometry.attributes.position.count).keys()]);
    const pos = g.attributes.position, idx = g.index;
    const buckets = [[], [], []];
    for (let i = 0; i < idx.count; i += 3) {
      const a = idx.getX(i), b = idx.getX(i + 1), c = idx.getX(i + 2);
      _p.fromBufferAttribute(pos, a).add(_q.fromBufferAttribute(pos, b)).add(_r.fromBufferAttribute(pos, c)).multiplyScalar(1 / 3).applyMatrix4(o.matrixWorld);
      buckets[_p.x > SIDE ? 1 : _p.x < -SIDE ? 2 : 0].push(a, b, c);
    }
    if (!buckets[1].length && !buckets[2].length) { o.material = base; continue; }
    g.setIndex([...buckets[0], ...buckets[1], ...buckets[2]]);
    g.clearGroups();
    let start = 0;
    buckets.forEach((bk, k) => { if (bk.length) g.addGroup(start, bk.length, k); start += bk.length; });
    const L = lampMat(m, kind), R = lampMat(m, kind);
    lamp[kind].push(L, R); lamp.blinkL.push(L); lamp.blinkR.push(R);
    o.geometry = g;
    o.material = [base, L, R];
  }
  // ---- wheels ----
  const wheels = [], front = [];
  const used = new Set();
  for (const c of centres) {
    const pivot = new THREE.Group(), spin = new THREE.Group();
    pivot.position.copy(c);
    pivot.add(spin);
    group.add(pivot);
    pivot.updateMatrixWorld(true);
    wheels.push({ pivot, spin });
    if (c.z > 0) front.push(pivot);
  }
  if (centres.length === 4) {
    const r = Math.max(...tyreBoxes.map((b) => (b.max.y - b.min.y) / 2));
    for (const o of meshes) {
      if (used.has(o) || !matsOf(o).some((m) => WHEEL_RE.test(label(o, m)))) continue;
      const b = boxOf(o), c = b.getCenter(new THREE.Vector3()), s = b.getSize(new THREE.Vector3());
      if (Math.max(s.x, s.y, s.z) > r * 2.4) continue;
      let best = -1, bd = Infinity;
      centres.forEach((w, k) => { const d = w.distanceTo(c); if (d < bd) { bd = d; best = k; } });
      if (bd > r * 0.9) continue;
      used.add(o);
      const caliper = matsOf(o).some((m) => /caliper/i.test(label(o, m)));
      (caliper ? wheels[best].pivot : wheels[best].spin).attach(o);
    }
  }
  const wheelR = tyreBoxes.length ? Math.max(...tyreBoxes.map((b) => (b.max.y - b.min.y) / 2)) : DIMS.wheelR;
  const zs = centres.map((c) => c.z);
  const setI = (m, i) => { m.emissiveIntensity = i; };
  let brake = false, blink = [false, false], heads = false;
  const apply = () => {
    for (const m of lamp.tail) { m.emissive.copy(m.userData.base); setI(m, m.userData.baseI * (brake ? 1.5 : 0.5)); }
    for (const m of lamp.head) { m.emissive.copy(m.userData.base); setI(m, m.userData.baseI * (heads ? 1.15 : 0.8)); }
    for (const [list, on] of [[lamp.blinkL, blink[0]], [lamp.blinkR, blink[1]]]) for (const m of list) if (on) { m.emissive.copy(amber); setI(m, 1.0); }
  };
  apply();
  return {
    group, body, meshes, wheelsFound: wheels.length === 4 && used.size > 0,
    setBrake(on) { if (on !== brake) { brake = on; apply(); } },
    setBlink(l, r) { if (l !== blink[0] || r !== blink[1]) { blink = [l, r]; apply(); } },
    setSteer(rad) { for (const p of front) p.rotation.y = rad; },
    setWheelSpin(rad) { for (const w of wheels) w.spin.rotation.x = rad; },
    setHeadlights(on) { if (on !== heads) { heads = on; apply(); } },
    dims: { wheelbase: zs.length >= 2 ? Math.max(...zs) - Math.min(...zs) : DIMS.wheelbase, wheelR, length: all.max.z - all.min.z },
  };
}
