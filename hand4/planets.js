/* =====================================================================
   planets.js — the six planets of "The Hand You're Dealt", drawn in light.

   Every planet is made only of light: star points, a few faint wireframe and constellation lines, and soft additive
   glows (radial-gradient canvas sprites, like the hand's). No textured solid spheres. One planet costs three draw
   calls (one Points, one LineSegments, one glow mesh) plus an optional dark occluder disc.

   ES module with no imports: pass THREE in.

     makePlanet(THREE, kind, fingerBaseRadius, opts) -> {
       group, radius, ringOuter, extent, kind, counts, quaternion,
       update(dt, t, camera), setOpacity(a), setReveal(p), setClock(s), dispose()
     }

   Everything moves in the vertex shader (spin, band drift, swirls, ring orbits, twinkle, the formation), so there
   are no per-frame buffer writes. Light comes from the upper right in VIEW space, as in v2 (SUN_V), so the planet
   is lit the same way however the caller orients it; lighting and occlusion use the group's model-view matrix, so
   the caller may position, rotate and uniformly scale the group freely.
   ===================================================================== */

const DEG = Math.PI / 180;
const TAU = Math.PI * 2;
const GOLDEN = Math.PI * (3 - Math.sqrt(5));

// vertex kinds (aA.x)
const K_SURF = 0;     // on the body: position = (lon, lat, height in R); spins with the planet, plus its own drift
const K_RING = 1;     // in the equatorial plane: position = (radius in R, angle, y in R); orbits at its own rate
const K_VIEW = 2;     // screen-aligned around the centre: position = (x, y, z) in R, view space (corona, limb)
const K_VORTEX = 3;   // an oval vortex on the body: position = (s 0..1, angle, height); swirls at aB.x rad/s
const K_HEMI = 4;     // a latitude row kept on the camera's side: position = (u 0..1 along the row, lat, height).
                      // Rows are symmetric about the axis, so their stars need only cover the visible span; they
                      // slide with the spin and wrap out of sight behind the limb (twice the density for the budget).
const HEMI_MARGIN = 0.32;

export const PLANET_KINDS = ['sun', 'jupiter', 'venus', 'saturn', 'mercury', 'mars'];

/* Body radius in finger-base radii, the suggested orientation (Euler degrees about X then Z, order 'ZXY', as v2's
   Q_SAT), spin (rad/s) and the spin phase that puts each planet's signature feature in front at the start. */
export const PLANET_SPECS = {
  sun:     { size: 1.5,  tilt: [8, -10],  spin: 0.05, spin0: 0.0 },
  jupiter: { size: 1.15, tilt: [-6, -8],  spin: 0.07, spin0: -0.3 },
  venus:   { size: 0.8,  tilt: [10, -10], spin: 0.035, spin0: 0.1 },
  saturn:  { size: 0.95, tilt: [25, -13], spin: 0.2,  spin0: 0.0 },
  mercury: { size: 0.45, tilt: [8, -6],   spin: 0.07, spin0: 0.25 },
  mars:    { size: 0.6,  tilt: [20, -12], spin: 0.1, spin0: 88 * DEG },
};

/** The suggested orientation for a kind (not applied by makePlanet: the caller orients the group). */
export function planetQuaternion(THREE, kind) {
  const [x, z] = PLANET_SPECS[kind].tilt;
  return new THREE.Quaternion().setFromEuler(new THREE.Euler(x * DEG, 0, z * DEG, 'ZXY'));
}

/* ---------------------------------------------------------------------
   small helpers
   --------------------------------------------------------------------- */
function mulberry32(a) {
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const lerp = (a, b, t) => a + (b - a) * t;
const clamp01 = x => Math.max(0, Math.min(1, x));
const mixc = (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
const wrapPi = a => Math.atan2(Math.sin(a), Math.cos(a));
// great-circle distance between two (lon, lat) points, radians
function angDist(lon1, lat1, lon2, lat2) {
  const c = Math.sin(lat1) * Math.sin(lat2) + Math.cos(lat1) * Math.cos(lat2) * Math.cos(lon1 - lon2);
  return Math.acos(Math.max(-1, Math.min(1, c)));
}
// a point at angular distance rho and bearing b from (lon0, lat0) (b = 0 toward north)
function offsetLL(lon0, lat0, rho, b) {
  const lat = Math.asin(Math.sin(lat0) * Math.cos(rho) + Math.cos(lat0) * Math.sin(rho) * Math.cos(b));
  const lon = lon0 + Math.atan2(Math.sin(b) * Math.sin(rho) * Math.cos(lat0), Math.cos(rho) - Math.sin(lat0) * Math.sin(lat));
  return [lon, lat];
}
const Z4 = [0, 0, 0, 0];

/* ---------------------------------------------------------------------
   glow textures: the hand's radial-gradient star sprite, and a broader soft one for body light
   --------------------------------------------------------------------- */
function gradientTexture(THREE, stops) {
  const S = 128, c = document.createElement('canvas');
  c.width = c.height = S;
  const g = c.getContext('2d');
  const grd = g.createRadialGradient(S / 2, S / 2, 0, S / 2, S / 2, S / 2);
  for (const [o, a] of stops) grd.addColorStop(o, `rgba(255,255,255,${a})`);
  g.fillStyle = grd; g.fillRect(0, 0, S, S);
  const t = new THREE.CanvasTexture(c);
  t.generateMipmaps = true; t.minFilter = THREE.LinearMipmapLinearFilter;
  return t;
}
const GLOW_STOPS = [[0, 1], [0.06, 0.9], [0.18, 0.32], [0.42, 0.08], [1, 0]];   // identical to the hand's GLOW
const SOFT_STOPS = [[0, 1], [0.22, 0.72], [0.45, 0.3], [0.7, 0.08], [1, 0]];

/* ---------------------------------------------------------------------
   shaders
   --------------------------------------------------------------------- */
const COMMON_GLSL = /* glsl */`
  #define PI 3.141592653589793
  attribute vec4 aA;      // kind, rate (rad/s), phase 0..1, formation delay 0..0.45
  attribute vec4 aB;      // per-kind extras (see planets.js)
  attribute vec3 aColor;
  uniform float uTime, uSpin, uReveal, uOpacity, uR, uOblate, uNight, uNightTint, uTermW, uBack, uTw, uTwSpeed, uLimbDark, uFade;
  uniform vec3 uLight;      // view space, toward the light
  uniform vec4 uVortex;     // lon0, lat0, half-width (rad of arc), half-height (rad)
  uniform float uRings;     // 1: the body is shadowed and crossed by Saturn's rings
  varying vec3 vColor;
  varying float vAlpha;

  vec3 surfPos(float lon, float lat, float h) {
    float rr = uR * (1.0 + h);
    return rr * vec3(cos(lat) * sin(lon), sin(lat) * uOblate, cos(lat) * cos(lon));
  }
  float ringDens(float k) {          // Saturn's C, B and A rings (the Cassini division is empty)
    return 0.22 * step(1.24, k) * step(k, 1.52) + 0.85 * step(1.52, k) * step(k, 1.95) + 0.55 * step(2.03, k) * step(k, 2.27);
  }
  // 1 = in front of / beside the sphere (centre C, radius Ro, view space), 0 = hidden behind it
  float occRay(vec3 P, vec3 C, float Ro) {
    float L = length(P);
    vec3 dir = P / max(L, 1e-6);
    float tca = dot(C, dir);
    float d2 = dot(C, C) - tca * tca, r2 = Ro * Ro;
    if (d2 >= r2 || tca < 0.0) return 1.0;
    float t0 = tca - sqrt(r2 - d2);
    float behind = smoothstep(0.0, 0.08 * Ro, L - t0);
    float edge = 1.0 - smoothstep(0.86 * r2, r2, d2);
    return 1.0 - behind * edge;
  }

  float gMu = 1.0;          // how squarely the surface faces the camera (1 centre, 0 limb), for surface kinds
  // computes the formed (target) position in view space plus lighting; returns T, writes lit / vis / day
  vec3 planetVertex(out float lit, out float vis, out float day, out vec3 C, out float Rv) {
    int kind = int(aA.x + 0.5);
    float rate = aA.y;
    C = (modelViewMatrix * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
    float sc = length(modelViewMatrix[0].xyz);
    Rv = uR * sc;
    vec3 T;
    lit = 1.0; vis = 1.0; day = 1.0;
    if (kind == 0 || kind == 3 || kind == 4) {
      float lon, lat, h = position.z;
      float hollow = 1.0;
      if (kind == 0) {
        lon = position.x + uSpin + rate * uTime;
        lat = position.y + aB.x * sin(position.x * aB.y + uTime * aB.z);
      } else if (kind == 4) {
        vec3 camL = -(transpose(mat3(modelViewMatrix)) * C) / (sc * sc);
        float lonV = atan(camL.x, camL.z);
        float tanE = camL.y / max(length(camL.xz), 1e-5);
        float halfW = acos(clamp(-tan(position.y) * tanE, -1.0, 1.0));
        float span = min(2.0 * PI, 2.0 * (halfW + ${HEMI_MARGIN.toFixed(3)}));
        lon = lonV - 0.5 * span + fract(position.x + (uSpin + rate * uTime) / span) * span;
        lat = position.y + aB.x * sin(lon * aB.y + uTime * aB.z);
      } else {
        float th = position.y + aB.x * uTime;
        lon = uVortex.x + uSpin + rate * uTime + uVortex.z * position.x * cos(th) / cos(uVortex.y);
        lat = uVortex.y + uVortex.w * position.x * sin(th);
      }
      if (kind != 3 && aB.w > 0.5) {                            // keep out of the vortex's hollow
        float dl = mod(lon - (uVortex.x + uSpin) + PI, 2.0 * PI) - PI;
        vec2 q = vec2(dl * cos(uVortex.y) / uVortex.z, (lat - uVortex.y) / uVortex.w);
        hollow = smoothstep(1.12, 1.4, length(q));
      }
      vec3 pL = surfPos(lon, lat, h);
      vec3 nL = normalize(vec3(pL.x, pL.y / (uOblate * uOblate), pL.z));
      T = (modelViewMatrix * vec4(pL, 1.0)).xyz;
      vec3 nV = normalize(normalMatrix * nL);
      float ndl = dot(nV, uLight);
      day = smoothstep(-uTermW, uTermW, ndl);
      lit = uNight + (1.0 - uNight) * day * (0.62 + 0.38 * clamp(ndl, 0.0, 1.0));
      float mu = dot(nV, normalize(-T));
      gMu = clamp(mu, 0.0, 1.0);
      vis = mix(uBack, 1.0, smoothstep(-0.1, 0.1, mu));
      lit *= mix(1.0, 0.42 + 0.58 * pow(max(mu, 0.0), 0.6), uLimbDark);
      vis *= hollow;
      if (uRings > 0.5) {
        vec3 n = normalize(normalMatrix * vec3(0.0, 1.0, 0.0));
        float dn = dot(uLight, n);
        if (abs(dn) > 1e-3) {                                   // the rings' shadow on the body
          float s = -dot(T - C, n) / dn;
          if (s > 0.0) lit *= 1.0 - 0.7 * ringDens(length(T + s * uLight - C) / Rv);
        }
        float pn = dot(T, n);
        if (abs(pn) > 1e-6) {                                   // the near rings crossing in front of the body
          float s2 = dot(C, n) / pn;
          if (s2 > 0.0 && s2 < 1.0) vis *= 1.0 - 0.55 * ringDens(length(T * s2 - C) / Rv);
        }
      }
    } else if (kind == 1) {
      float ph = position.y + rate * uTime;
      vec3 pL = uR * vec3(position.x * sin(ph), position.z, position.x * cos(ph));
      T = (modelViewMatrix * vec4(pL, 1.0)).xyz;
      vec3 d = T - C;
      float tl = dot(d, uLight);                                // the planet's shadow across the rings
      float sh = tl < 0.0 ? smoothstep(Rv * 0.92, Rv * 1.06, length(d - tl * uLight)) : 1.0;
      lit = mix(0.1, 1.0, sh);
      vis = occRay(T, C, Rv * 0.985);
    } else {
      float a = rate * uTime;
      float br = 1.0 + aB.x * sin(uTime * aB.y + aA.z * 6.2831);
      vec2 xy = mat2(cos(a), sin(a), -sin(a), cos(a)) * position.xy;
      xy = xy * mix(1.0, br, smoothstep(1.0, 1.08, length(position.xy)));
      T = C + vec3(xy, position.z) * Rv;
      vis = occRay(T, C, Rv * 0.985);
      if (aB.w > 0.0) {                                          // limb lines: brighter on the lit side
        float k = dot(normalize(position.xy), normalize(uLight.xy + vec2(1e-4)));
        lit = mix(1.0, 0.18 + 0.82 * smoothstep(-0.5, 0.7, k), aB.w);
      }
      if (aB.z > 0.0) lit *= 1.0 - aB.z * (0.5 + 0.5 * sin(uTime * (2.3 + 3.1 * aA.z) + aA.z * 40.0));
      if (aB.w < -0.5) {                                         // a point on the facing disc (the Sun's photosphere)
        gMu = clamp(position.z / max(length(position), 1e-4), 0.0, 1.0);
        lit *= mix(1.0, 0.42 + 0.58 * pow(gMu, 0.6), uLimbDark);
      }
    }
    return T;
  }
`;

const STAR_VS = /* glsl */`
  attribute vec3 aSrc;     // where this star starts the formation (group-local)
  attribute vec2 aSA;      // size (world units at the reference distance), alpha
  uniform float uViewH, uMinS, uDotTo, uMaxPt, uAtten, uRef;
  varying float vSmall;
  ${COMMON_GLSL}
  void main() {
    float lit, vis, day, Rv; vec3 C;
    vec3 T = planetVertex(lit, vis, day, C, Rv);
    float ph = aA.z;
    // the formation: from the source along a gentle curve, easing in; exactly home at q = 1
    float q = clamp((uReveal - aA.w) / 0.55, 0.0, 1.0);
    float e = 1.0 - pow(1.0 - q, 3.0);
    vec3 S = (modelViewMatrix * vec4(aSrc, 1.0)).xyz;
    vec3 dv = T - S;
    vec3 P = mix(S, T, e) + vec3(-dv.y, dv.x, 0.0) * (fract(ph * 13.7) - 0.5) * 0.8 * e * (1.0 - e);
    float xb = clamp((q - 0.55) / 0.45, 0.0, 1.0);
    float bump = 4.0 * xb * (1.0 - xb);                          // a brief brightening as each star lands
    float formed = smoothstep(0.35, 0.95, uReveal);
    lit = mix(1.0, lit, e);
    vis = mix(1.0, vis, formed);
    day = mix(1.0, day, e);
    float tw = 1.0 - uTw * (0.5 + 0.5 * sin(uTime * uTwSpeed * (0.6 + fract(ph * 7.31) * 1.5) + ph * 6.2831));
    float a = aSA.y * uOpacity * lit * vis * tw * mix(0.5, 1.0, e) * (1.0 + 0.8 * bump);
    vec3 nightC = aColor * vec3(0.62, 0.58, 1.0);
    vec3 baseC = mix(mix(aColor, nightC, uNightTint), aColor, day) * mix(vec3(1.0), vec3(1.0, 0.7, 0.46), uLimbDark * pow(1.0 - gMu, 1.5));
    vColor = mix(vec3(1.0, 0.86, 0.74), baseC, e);
    vec4 mv = vec4(P, 1.0);
    float d = max(0.01, -P.z);
    float sc = length(modelViewMatrix[0].xyz);
    float px = aSA.y > 0.0 ? aSA.x * sc * projectionMatrix[1][1] * uViewH * 0.5 / (pow(d, uAtten) * pow(uRef, 1.0 - uAtten)) : 0.0;
    px *= mix(0.75, 1.0, e);
    if (px < uMinS) { a *= px / uMinS; px = uMinS; }
    vSmall = 1.0 - smoothstep(uMinS, uDotTo, px);
    gl_PointSize = min(px, uMaxPt);
    vAlpha = a;
    gl_Position = projectionMatrix * mv;
  }
`;
const STAR_FS = /* glsl */`
  uniform sampler2D uMap;
  varying vec3 vColor;
  varying float vAlpha;
  varying float vSmall;
  void main() {
    float r = clamp(1.0 - length(gl_PointCoord - 0.5) * 2.0, 0.0, 1.0);
    float m = mix(texture2D(uMap, gl_PointCoord).a, r * r * 1.4, vSmall) * vAlpha;
    if (m < 0.002) discard;
    gl_FragColor = vec4(vColor * m, 1.0);
  }
`;
const LINE_VS = /* glsl */`
  attribute float aAl;
  ${COMMON_GLSL}
  void main() {
    float lit, vis, day, Rv; vec3 C;
    vec3 T = planetVertex(lit, vis, day, C, Rv);
    vAlpha = aAl * uOpacity * uFade * lit * vis;
    vColor = mix(mix(aColor, aColor * vec3(0.62, 0.58, 1.0), uNightTint), aColor, day);
    gl_Position = projectionMatrix * vec4(T, 1.0);
  }
`;
const LINE_FS = /* glsl */`
  varying vec3 vColor;
  varying float vAlpha;
  void main() { gl_FragColor = vec4(vColor * vAlpha, 1.0); }
`;
const GLOW_VS = /* glsl */`
  attribute vec4 aG;     // diameter (R), alpha, offset toward the light (R), flicker
  attribute vec4 aG2;    // colour, texture (0 star glow, 1 soft)
  uniform float uR, uOpacity, uFade, uTime;
  uniform vec3 uLight;
  varying vec2 vUv;
  varying vec3 vC;
  varying float vA;
  varying float vT;
  void main() {
    vec3 C = (modelViewMatrix * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
    float Rv = uR * length(modelViewMatrix[0].xyz);
    vec3 P = C + uLight * aG.z * Rv + vec3(position.xy * aG.x * Rv, 0.0);
    vUv = uv; vC = aG2.rgb; vT = aG2.a;
    vA = aG.y * uOpacity * uFade * (1.0 + aG.w * (0.6 * sin(uTime * 1.7 + aG.x * 3.0) + 0.4 * sin(uTime * 4.3 + aG.x)));
    gl_Position = projectionMatrix * vec4(P, 1.0);
  }
`;
const GLOW_FS = /* glsl */`
  uniform sampler2D uGlow, uSoft;
  varying vec2 vUv;
  varying vec3 vC;
  varying float vA;
  varying float vT;
  void main() {
    float m = mix(texture2D(uGlow, vUv).a, texture2D(uSoft, vUv).a, vT) * vA;
    if (m < 0.001) discard;
    gl_FragColor = vec4(vC * m, 1.0);
  }
`;
const OCC_VS = /* glsl */`
  uniform float uR;
  varying vec2 vP;
  void main() {
    vec3 C = (modelViewMatrix * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
    float Rv = uR * length(modelViewMatrix[0].xyz);
    vP = position.xy * 2.0;                                     // -1..1 across the quad
    gl_Position = projectionMatrix * vec4(C + vec3(position.xy * 2.0 * Rv * 1.04, 0.0), 1.0);
  }
`;
const OCC_FS = /* glsl */`
  uniform float uA, uOpacity, uFade;
  uniform vec3 uCol;
  varying vec2 vP;
  void main() {
    float d = length(vP) * 1.04;                                 // 1 at the planet's limb
    float a = uA * uOpacity * uFade * (1.0 - smoothstep(0.9, 1.01, d));
    if (a < 0.002) discard;
    gl_FragColor = vec4(uCol, a);
  }
`;

/* ---------------------------------------------------------------------
   builders: stars and line segments as flat attribute lists
   --------------------------------------------------------------------- */
class Stars {
  constructor(rnd) { this.rnd = rnd; this.p = []; this.a = []; this.b = []; this.c = []; this.s = []; this.stage = []; }
  add(kind, x, y, z, rate, b, col, size, alpha, stage = 0) {
    this.p.push(x, y, z); this.a.push(kind, rate, this.rnd(), 0); this.b.push(...b);
    this.c.push(col[0], col[1], col[2]); this.s.push(size, alpha); this.stage.push(stage);
  }
  get n() { return this.stage.length; }
}
class Lines {
  constructor() { this.p = []; this.a = []; this.b = []; this.c = []; this.al = []; }
  vert(v) {
    this.p.push(v.p[0], v.p[1], v.p[2]); this.a.push(v.k, v.rate || 0, v.ph || 0, 0);
    const b = v.b || Z4; this.b.push(b[0], b[1], b[2], b[3]);
    this.c.push(v.c[0], v.c[1], v.c[2]); this.al.push(v.a);
  }
  seg(v0, v1) { this.vert(v0); this.vert(v1); }
  poly(verts, closed) {
    for (let i = 0; i + 1 < verts.length; i++) this.seg(verts[i], verts[i + 1]);
    if (closed && verts.length > 2) this.seg(verts[verts.length - 1], verts[0]);
  }
  get segments() { return this.al.length / 2; }
}
// a closed loop on the surface from f(u) -> [lon, lat], u in [0, 1)
function surfLoop(ln, f, n, col, alpha, h = 0.004, rate = 0, closed = true) {
  const v = [];
  for (let i = 0; i < n + (closed ? 0 : 1); i++) {
    const [lon, lat] = f(i / n);
    v.push({ k: K_SURF, p: [lon, lat, h], rate, c: col, a: alpha });
  }
  ln.poly(v, closed);
}
function greatCircles(ln, count, col, alpha, phase = 0, n = 72) {
  for (let m = 0; m < count; m++) {
    const lon0 = phase + (m / count) * Math.PI;
    surfLoop(ln, u => {
      const a = u * TAU;                              // around the meridian circle through both poles
      const lat = Math.asin(Math.sin(a));
      const lon = Math.cos(a) >= 0 ? lon0 : lon0 + Math.PI;
      return [lon, lat];
    }, n, col, alpha);
  }
}
function latCircle(ln, lat, col, alpha, n = 72, h = 0.004) {
  surfLoop(ln, u => [u * TAU, lat], n, col, alpha, h);
}
function viewCircle(ln, radius, n, col, alpha, litW = 1) {
  const v = [];
  for (let i = 0; i < n; i++) {
    const a = (i / n) * TAU;
    v.push({ k: K_VIEW, p: [Math.cos(a) * radius, Math.sin(a) * radius, 0], b: [0, 0, 0, litW], c: col, a: alpha });
  }
  ln.poly(v, true);
}
// a roughly even scatter over the sphere (Fibonacci lattice with jitter)
function fibSphere(n, rnd, jitter = 0.45) {
  const out = [], sp = Math.sqrt(4 * Math.PI / n);
  for (let i = 0; i < n; i++) {
    const y = 1 - 2 * (i + 0.5) / n;
    let lat = Math.asin(y), lon = i * GOLDEN;
    lat += (rnd() - 0.5) * sp * jitter;
    lon += (rnd() - 0.5) * sp * jitter / Math.max(0.2, Math.cos(lat));
    lat = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, lat));
    out.push([wrapPi(lon), lat]);
  }
  return out;
}

/* =====================================================================
   SATURN — pale gold bands as rows of stars, rings of star dust with the Cassini division, the planet's shadow
   across the rings and the rings' shadow across the body; the near side hides the far ring.
   ===================================================================== */
function buildSaturn(c) {
  const { st, ln, rnd, PX } = c;
  const R = (a, b) => lerp(a, b, rnd());
  const BANDS = [   // |lat| up to (deg), colour, alpha, size (px)
    [10, [1.0, 0.92, 0.74], 1.0, 9.0],
    [20, [0.94, 0.74, 0.5], 0.78, 8.2],
    [30, [1.0, 0.89, 0.68], 0.95, 8.8],
    [43, [0.9, 0.7, 0.48], 0.72, 8.0],
    [58, [0.97, 0.85, 0.64], 0.8, 8.2],
    [91, [0.82, 0.76, 0.64], 0.58, 7.2],
  ];
  const band = lat => BANDS.find(b => Math.abs(lat) / DEG < b[0]);
  const ROWS = 15;
  for (let k = 0; k < ROWS; k++) {
    const lat = (-79 + (158 * k) / (ROWS - 1)) * DEG;
    const b = band(lat);
    const n = Math.max(8, Math.round(44 * Math.pow(Math.cos(lat), 0.8)));
    const off = rnd(), drift = (k % 2 ? 1 : -1) * 0.012;
    for (let i = 0; i < n; i++) {
      const u = (i + off + (rnd() - 0.5) * 0.35) / n;
      const col = mixc(b[1], [1, 1, 1], R(0, 0.1));
      st.add(K_HEMI, u, lat + (rnd() - 0.5) * 0.014, 0, drift, Z4, col, PX * b[3] * R(0.85, 1.12), b[2] * R(0.72, 1.0), 0);
    }
  }
  for (const [lon, lat] of fibSphere(60, rnd, 0.9)) {         // a little fill between the rows
    const b = band(lat);
    st.add(K_SURF, lon, lat, 0, 0, Z4, b[1], PX * 6, b[2] * 0.45, 0);
  }
  // rings (radii in R): C 1.24-1.52, B 1.52-1.95, the Cassini division, A 2.03-2.27 with the Encke gap
  const ringK = () => {
    for (;;) {
      const x = rnd();
      const k = x < 0.15 ? R(1.24, 1.52) : x < 0.68 ? R(1.52, 1.95) : R(2.03, 2.27);
      if (k > 2.19 && k < 2.215 && rnd() < 0.85) continue;
      return k;
    }
  };
  const bright = k => (k < 1.52 ? 0.38 : k < 1.95 ? 0.95 : 0.7);
  const tint = k => (k < 1.52 ? [0.86, 0.74, 0.62] : k < 1.95 ? [1.0, 0.88, 0.68] : [0.96, 0.84, 0.7]);
  const omega = k => 0.16 * Math.pow(1.4 / k, 1.5);
  for (let i = 0; i < 800; i++) {
    const k = ringK(), big = rnd() < 0.06;
    st.add(K_RING, k, rnd() * TAU, (rnd() - 0.5) * 0.008, omega(k), Z4, tint(k),
      PX * (big ? R(8, 10) : R(4.6, 7.4)), bright(k) * R(0.6, 1.0) * (big ? 1.25 : 1), 1);
  }
  // ring edges, drawn in light
  for (const [k, a] of [[1.24, 0.07], [1.52, 0.12], [1.95, 0.24], [2.03, 0.18], [2.27, 0.26]]) {
    const v = [];
    for (let i = 0; i < 128; i++) v.push({ k: K_RING, p: [k, (i / 128) * TAU, 0], c: [1.0, 0.86, 0.7], a });
    ln.poly(v, true);
  }
  greatCircles(ln, 3, [0.98, 0.82, 0.66], 0.09);
  // the north polar hexagon
  surfLoop(ln, u => {
    const a = u * TAU, rho = (12 * DEG) / Math.cos(((a * 3) / Math.PI % 1 - 0.5) * (Math.PI / 3));
    return [a, Math.PI / 2 - Math.min(rho, 14 * DEG)];
  }, 36, [1.0, 0.86, 0.66], 0.12);
  return {
    rings: true, ringOuter: 2.27, extent: 2.27,
    glows: [
      [6.5, 0.24, 0.0, 0, [1.0, 0.8, 0.6], 0],
      [2.5, 0.42, 0.32, 0, [0.98, 0.8, 0.58], 1],
    ],
    look: { oblate: 0.93, night: 0.13, nightTint: 1, termW: 0.22, back: 0.05, tw: 0.2, twSpeed: 1, limbDark: 0, occ: 0.4 },
  };
}

/* =====================================================================
   JUPITER — big and oblate; rows of stars in cream zones and tan/rust belts that drift against each other, the
   Great Red Spot as an oval swirl of red stars in its hollow, festoons on the equator, a string of white ovals.
   ===================================================================== */
function buildJupiter(c) {
  const { st, ln, rnd, PX } = c;
  const R = (a, b) => lerp(a, b, rnd());
  const ZONE = [1.0, 0.95, 0.84], ZONE2 = [0.97, 0.88, 0.72], TAN = [0.93, 0.66, 0.44], RUST = [0.96, 0.5, 0.3], POLE = [0.8, 0.7, 0.62];
  // the Great Red Spot sits on 22 S, on the south edge of the South Equatorial Belt
  const G_LAT = -22 * DEG, G_A = 0.25, G_B = 0.13;
  // Rows of stars: zones are bright cream rows of larger stars; belts are dimmer, redder rows of smaller, denser
  // stars with a faint line of their colour along them. Each row drifts at its own rate (the jets); rows round the
  // Red Spot part around its hollow.
  const ZB = (lat, col, alpha, drift, o = {}) => ({ lat, col, alpha, drift, size: 8.6, n: 44, line: 0, ...o });
  const BELT = (lat, col, alpha, drift, line, o = {}) => ({ lat, col, alpha, drift, size: 6.8, n: 64, line, ...o });
  const RUSTB = [0.92, 0.4, 0.22], TANB = [0.9, 0.6, 0.38];
  const ROWS = [
    ZB(-73, POLE, 0.36, 0, { size: 7.4, n: 40 }), ZB(-63, POLE, 0.42, 0.01, { size: 7.8 }), BELT(-55, TANB, 0.42, -0.01, 0.08),
    ZB(-48, ZONE2, 0.6, 0.012), ZB(-41, ZONE2, 0.68, 0.016), BELT(-34.5, TANB, 0.5, -0.014, 0.13),
    ZB(-28, ZONE, 0.88, 0.02, { hollow: 1 }), ZB(-22, ZONE, 0.9, -0.02, { hollow: 1 }),                       // STrZ
    BELT(-17, RUSTB, 0.62, 0.024, 0.22, { hollow: 1 }), BELT(-13.5, RUSTB, 0.62, 0.02, 0.22, { hollow: 1 }), BELT(-10, RUSTB, 0.58, 0.03, 0.2),   // SEB
    ZB(-5, ZONE, 0.95, 0.04), ZB(0, ZONE, 0.88, 0.042), ZB(5, ZONE, 0.95, 0.04),                                 // EZ
    BELT(9.5, RUSTB, 0.62, 0.03, 0.22, { fest: 1.8 }), BELT(13, RUSTB, 0.62, -0.016, 0.22), BELT(16.5, RUSTB, 0.58, -0.016, 0.2),   // NEB
    ZB(21.5, ZONE, 0.85, 0.02), BELT(26.5, TANB, 0.5, -0.02, 0.13), BELT(31, TANB, 0.46, -0.02, 0.11),        // NTrZ, NTB
    ZB(36, ZONE2, 0.68, 0.014), BELT(42, TANB, 0.42, -0.012, 0.09), ZB(48.5, ZONE2, 0.55, 0.01),
    BELT(56, TANB, 0.38, -0.01, 0.07), ZB(64, POLE, 0.42, 0, { size: 7.8 }), ZB(73, POLE, 0.36, 0, { size: 7.4, n: 40 }),
  ];
  for (const row of ROWS) {
    const lat = row.lat * DEG, hollow = row.hollow || 0;
    const n = Math.max(10, Math.round(row.n * Math.pow(Math.cos(lat), 0.8)));
    const off = rnd();
    for (let i = 0; i < n; i++) {
      const u = (i + off + (rnd() - 0.5) * 0.4) / n;
      const b = [(row.fest || 0) * DEG, 11, 0.12, hollow];
      st.add(K_HEMI, u, lat + (rnd() - 0.5) * 0.014, 0, row.drift, b, mixc(row.col, [1, 1, 1], R(0, 0.06)), PX * row.size * R(0.85, 1.12), row.alpha * R(0.72, 1.0), 0);
    }
    if (row.line) {
      const v = [];
      for (let i = 0; i < 96; i++) v.push({ k: K_SURF, p: [(i / 96) * TAU, lat, 0.003], rate: row.drift, b: [0, 0, 0, hollow], c: row.col, a: row.line });
      ln.poly(v, true);
    }
  }
  // the Red Spot itself: concentric ovals, a spiral twist, the collar swirling fastest
  const RINGS = [0.16, 0.32, 0.48, 0.64, 0.8, 1.0];
  RINGS.forEach((s, j) => {
    const n = Math.round(6 + 26 * s);
    for (let i = 0; i < n; i++) {
      const th = (i / n) * TAU + s * 2.6 + rnd() * 0.15;
      const col = mixc([1.0, 0.58, 0.42], [1.0, 0.4, 0.26], s);
      st.add(K_VORTEX, s * R(0.96, 1.04), th, 0.004, 0, [0.1 + 0.26 * s, 0, 0, 0], col, PX * (j === 5 ? 9.4 : 8.4), (j === 5 ? 1.0 : 0.86) * R(0.85, 1), 0);
    }
  });
  st.add(K_VORTEX, 0, 0, 0.004, 0, Z4, [1.0, 0.66, 0.52], PX * 11, 1.0, 0);
  // the string of pearls: small white ovals on 40 S
  for (let m = 0; m < 6; m++) {
    const lon0 = -1.35 + m * 0.5 + (rnd() - 0.5) * 0.12, lat0 = -40 * DEG;
    for (let i = 0; i < 6; i++) {
      const [lon, lat] = offsetLL(lon0, lat0, 0.042, (i / 6) * TAU);
      st.add(K_SURF, lon, lat, 0.002, 0.016, Z4, [1, 0.97, 0.92], PX * 7, 0.85, 0);
    }
  }
  // lines: the Red Spot's outline, the equatorial edges, a few meridians
  {
    const v = [];
    for (let i = 0; i < 48; i++) v.push({ k: K_VORTEX, p: [1.16, (i / 48) * TAU, 0.006], b: Z4, c: [1.0, 0.58, 0.46], a: 0.34 });
    ln.poly(v, true);
  }
  greatCircles(ln, 3, [0.98, 0.8, 0.64], 0.07, 0.3);
  return {
    vortex: [0, G_LAT, G_A, G_B],
    extent: 1.0,
    glows: [
      [6.0, 0.2, 0.0, 0, [1.0, 0.8, 0.62], 0],
      [2.5, 0.36, 0.3, 0, [0.98, 0.84, 0.66], 1],
    ],
    look: { oblate: 0.935, night: 0.12, nightTint: 1, termW: 0.2, back: 0.05, tw: 0.18, twSpeed: 1, limbDark: 0, occ: 0.4 },
  };
}

/* =====================================================================
   VENUS — pale cream-gold and smooth, wrapped in curved streams of cloud stars that super-rotate faster than the
   planet: the sideways "Y" across the equator, mid-latitude streams and the polar collars; a bright soft glow and
   a thin atmospheric rim.
   ===================================================================== */
function buildVenus(c) {
  const { st, ln, rnd, PX } = c;
  const R = (a, b) => lerp(a, b, rnd());
  const BASE = [1.0, 0.92, 0.74];
  // the haze: a fine random scatter (no lattice), faintly banded
  for (let i = 0; i < 820; i++) {
    const lat = Math.asin(rnd() * 2 - 1), lon = rnd() * TAU;
    const w = 0.5 + 0.5 * Math.sin(lat * 5 + Math.sin(lon * 2) * 0.8);
    st.add(K_SURF, lon, lat, 0, 0, Z4, mixc(BASE, [1, 0.96, 0.88], w), PX * R(5.4, 6.8), (0.44 + 0.18 * w) * R(0.75, 1), 0);
  }
  const CLOUD = [1.0, 0.97, 0.9], GOLD = [1.0, 0.86, 0.64];
  const SUPER = 0.09;                                       // the clouds outrun the planet
  // a stream: a soft band of stars (several loose strands) along a curve, tapering at both ends, with a faint
  // line of light down its middle
  const stream = (f, n, col, alpha, size, line, width = 0.07, strands = 3) => {
    const verts = [];
    for (let i = 0; i < n; i++) {
      const u = i / (n - 1);
      const [lon, lat] = f(u);
      const taper = Math.pow(Math.sin(Math.PI * Math.min(1, 0.04 + u * 0.96)), 0.8);
      for (let k = 0; k < strands; k++) {
        const o = (strands > 1 ? (k / (strands - 1) - 0.5) : 0) * width * (0.5 + 0.5 * taper) + (rnd() - 0.5) * 0.02;
        const edge = strands > 1 ? 1 - 0.45 * Math.abs(k / (strands - 1) - 0.5) * 2 : 1;
        st.add(K_SURF, lon + (rnd() - 0.5) * 0.03, lat + o, 0.025, SUPER, Z4, col, PX * size * R(0.85, 1.1), alpha * edge * (0.2 + 0.8 * taper) * R(0.75, 1), 0);
      }
      if (line && i % 2 === 0) verts.push({ k: K_SURF, p: [lon, lat, 0.03], rate: SUPER, c: col, a: line * (0.15 + 0.85 * taper) });
    }
    if (line) ln.poly(verts, false);
  };
  // the sideways "Y" of Venus's cloud tops: a broad stem along the equator forking into two arms that sweep
  // toward the poles, with a fainter chevron inside the fork and swirls at mid-latitudes
  stream(u => [lerp(-1.9, -0.35, u), 0.03 * Math.sin(u * 5)], 30, CLOUD, 0.8, 6.8, 0.16, 0.1, 3);
  for (const sg of [1, -1]) {
    stream(u => { const d = u * 1.75; return [-0.4 + d, sg * 1.25 * (1 - Math.exp(-(0.06 + d * 0.8)))]; }, 30, CLOUD, 0.78, 6.8, 0.15, 0.1, 3);
    stream(u => { const d = u * 1.3; return [0.25 + d, sg * 1.1 * (1 - Math.exp(-(0.05 + d * 0.65)))]; }, 22, GOLD, 0.6, 6.2, 0.08, 0.06, 2);
    stream(u => { const lon = -2.6 + 2.0 * u; return [lon, sg * (0.95 + 0.12 * Math.sin(lon * 1.8 + 0.6))]; }, 26, GOLD, 0.6, 6.2, 0.07, 0.07, 2);
  }
  // polar collars: a loose spiral round each pole
  for (const sgn of [1, -1]) stream(u => [u * TAU * 1.25, sgn * (1.36 - 0.14 * u)], 36, GOLD, 0.5, 6, 0.05, 0, 1);
  viewCircle(ln, 1.035, 128, [1.0, 0.9, 0.72], 0.2, 1);
  greatCircles(ln, 3, [1.0, 0.86, 0.7], 0.06, 0.6);
  latCircle(ln, 0, [1.0, 0.86, 0.7], 0.05);
  return {
    extent: 1.05,
    glows: [
      [7.5, 0.3, 0.0, 0, [1.0, 0.88, 0.7], 0],
      [2.6, 0.44, 0.3, 0, [1.0, 0.92, 0.78], 1],
    ],
    look: { oblate: 1, night: 0.16, nightTint: 1, termW: 0.4, back: 0.05, tw: 0.16, twSpeed: 0.8, limbDark: 0, occ: 0.5 },
  };
}

/* =====================================================================
   MERCURY — the smallest: silver-grey points, crater rims as tiny rings of stars (Caloris the largest), a bright
   rayed crater, and a hard day/night terminator (no air to soften it).
   ===================================================================== */
function buildMercury(c) {
  const { st, ln, rnd, PX } = c;
  const R = (a, b) => lerp(a, b, rnd());
  // craters: lon, lat (deg), angular radius (rad)
  const CRATERS = [
    [38, 26, 0.36], [-40, 32, 0.2], [-14, -30, 0.22], [52, -24, 0.17], [-62, -6, 0.16], [8, 4, 0.13],
    [-24, 58, 0.15], [86, 12, 0.18], [-92, 30, 0.2], [20, -58, 0.16], [120, -30, 0.2], [-130, -20, 0.22],
    [160, 30, 0.2], [-160, 50, 0.18], [70, 55, 0.14],
  ].map(([lo, la, r]) => [lo * DEG, la * DEG, r]);
  const RAYED = [-48 * DEG, -40 * DEG];                      // a young bright rayed crater
  const plains = (lon, lat) => Math.exp(-(angDist(lon, lat, 0.5, 0.95) ** 2) / 0.12) + Math.exp(-(angDist(lon, lat, -2.2, -0.3) ** 2) / 0.2);
  for (const [lon, lat] of fibSphere(800, rnd, 0.6)) {
    if (CRATERS.some(([clo, cla, r]) => angDist(lon, lat, clo, cla) < r * 0.8)) continue;    // crater floors stay dark
    const smooth = plains(lon, lat);
    const g = R(0.8, 0.94);
    st.add(K_SURF, lon, lat, 0, 0, Z4, [g * 0.98, g, g * 1.03], PX * R(5.8, 7.2), (0.82 - 0.26 * smooth) * R(0.75, 1), 0);
  }
  for (const [lon0, lat0, r] of CRATERS) {
    const n = Math.round(8 + r * 62);
    for (let i = 0; i < n; i++) {
      const b = (i / n) * TAU + rnd() * 0.1;
      const [lon, lat] = offsetLL(lon0, lat0, r * R(0.95, 1.05), b);
      st.add(K_SURF, lon, lat, 0.008, 0, Z4, [0.97, 0.97, 1.0], PX * R(6.2, 7.4), R(0.72, 0.95), 0);
    }
    if (r > 0.17) st.add(K_SURF, lon0, lat0, 0.004, 0, Z4, [0.9, 0.9, 0.95], PX * 6.5, 0.75, 0);   // central peak
    if (r > 0.3) {                                            // Caloris: a second, inner ring
      for (let i = 0; i < 18; i++) {
        const [lon, lat] = offsetLL(lon0, lat0, r * 0.6, (i / 18) * TAU + 0.2);
        st.add(K_SURF, lon, lat, 0.004, 0, Z4, [0.88, 0.88, 0.92], PX * 6, 0.55, 0);
      }
    }
  }
  // the rayed crater: a bright little rim and rays of stars
  {
    const [lon0, lat0] = RAYED;
    st.add(K_SURF, lon0, lat0, 0.01, 0, Z4, [1, 1, 1], PX * 11, 1.0, 0);
    for (let i = 0; i < 7; i++) {
      const [lon, lat] = offsetLL(lon0, lat0, 0.06, (i / 7) * TAU);
      st.add(K_SURF, lon, lat, 0.008, 0, Z4, [1, 1, 1], PX * 6.8, 0.95, 0);
    }
    for (let k = 0; k < 9; k++) {
      const b = (k / 9) * TAU + rnd() * 0.4, len = R(0.35, 0.75);
      for (let j = 1; j <= 6; j++) {
        const u = j / 6;
        const [lon, lat] = offsetLL(lon0, lat0, 0.06 + len * u, b + (rnd() - 0.5) * 0.05);
        st.add(K_SURF, lon, lat, 0.006, 0, Z4, [0.96, 0.97, 1.0], PX * (7 - 2 * u), (1 - 0.7 * u) * 0.95, 0);
      }
    }
  }
  // rims of the three largest craters as faint lines, plus a cartographic wireframe
  for (const [lon0, lat0, r] of CRATERS.slice(0, 3)) {
    surfLoop(ln, u => offsetLL(lon0, lat0, r, u * TAU), 28, [0.88, 0.9, 0.98], 0.2, 0.008);
  }
  greatCircles(ln, 3, [0.86, 0.86, 0.94], 0.1, 0.2);
  latCircle(ln, 0, [0.86, 0.86, 0.94], 0.1);
  latCircle(ln, 45 * DEG, [0.86, 0.86, 0.94], 0.08, 56);
  latCircle(ln, -45 * DEG, [0.86, 0.86, 0.94], 0.08, 56);
  return {
    extent: 1.0,
    glows: [
      [4.5, 0.1, 0.0, 0, [0.85, 0.82, 0.86], 0],
      [2.3, 0.2, 0.35, 0, [0.86, 0.86, 0.92], 1],
    ],
    look: { oblate: 1, night: 0.035, nightTint: 1, termW: 0.06, back: 0.04, tw: 0.16, twSpeed: 1, limbDark: 0, occ: 0.45 },
  };
}

/* =====================================================================
   MARS — plain and even: rust-red points with only a gentle random mix toward ochre (no polar caps, no maria, no
   canyons or volcanoes), lit and shaded like the rest, with a thin rose limb of air.
   ===================================================================== */
function buildMars(c) {
  const { st, ln, rnd, PX } = c;
  const R = (a, b) => lerp(a, b, rnd());
  const RUST = [1.0, 0.5, 0.3], OCHRE = [1.0, 0.62, 0.4];
  for (const [lon, lat] of fibSphere(1180, rnd, 0.7)) {
    st.add(K_SURF, lon, lat, 0, 0, Z4, mixc(RUST, OCHRE, rnd() * 0.45), PX * R(6.6, 8), 0.86 * R(0.8, 1), 0);
  }
  viewCircle(ln, 1.025, 112, [1.0, 0.62, 0.52], 0.14, 1);
  greatCircles(ln, 3, [1.0, 0.7, 0.6], 0.08, 0.15);
  latCircle(ln, 0, [1.0, 0.7, 0.6], 0.07);
  return {
    extent: 1.03,
    glows: [
      [5.2, 0.2, 0.0, 0, [1.0, 0.56, 0.42], 0],
      [2.4, 0.4, 0.3, 0, [1.0, 0.56, 0.38], 1],
    ],
    look: { oblate: 1, night: 0.09, nightTint: 1, termW: 0.13, back: 0.04, tw: 0.16, twSpeed: 1, limbDark: 0, occ: 0.45 },
  };
}

/* =====================================================================
   THE SUN — the largest: a dense warm white-gold disc of granules (bright cells with dark lanes) that shimmer,
   darker toward the limb, two sunspots, a corona of streamers out to ~2.3 finger radii, and two rose prominence
   arcs at the limb. It is the light, so it has no night side.
   ===================================================================== */
function buildSun(c) {
  const { st, ln, rnd, PX } = c;
  const R = (a, b) => lerp(a, b, rnd());
  // Granulation, on the face turned to the camera (granules are short-lived, so the pattern needn't turn with the
  // Sun; the wireframe and the sunspot do): bright cells of clustered stars with dark lanes between, every star
  // flickering on its own. Facing-only, the whole budget lands where it can be seen.
  const toXYZ = (lon, lat) => [Math.cos(lat) * Math.sin(lon), Math.sin(lat), Math.cos(lat) * Math.cos(lon)];
  const CELLS = fibSphere(600, rnd, 1.1).map(([lo, la]) => toXYZ(lo, la)).filter(v => v[2] > 0.03);
  const cellR = Math.sqrt((4 * Math.PI) / 600) * 0.5;
  const SPOT = [-0.3, -0.16], SPOT_R = 0.1;                   // a sunspot, low and left of centre (disc units)
  const dSpot = (x, y) => Math.hypot(x - SPOT[0], y - SPOT[1]);
  for (const cpos of CELLS) {
    const n = 3 + Math.floor(rnd() * 2.2);
    // a tangent frame at the cell
    const tx = [cpos[2], 0, -cpos[0]], tl = Math.hypot(tx[0], tx[2]) || 1;
    const t1 = [tx[0] / tl, 0, tx[2] / tl];
    const t2 = [cpos[1] * t1[2] - cpos[2] * t1[1], cpos[2] * t1[0] - cpos[0] * t1[2], cpos[0] * t1[1] - cpos[1] * t1[0]];
    for (let i = 0; i < n; i++) {
      let p = cpos;
      if (i) {
        const a = (i / (n - 1)) * TAU + rnd(), d = cellR * R(0.35, 0.62);
        p = [0, 1, 2].map(j => cpos[j] + (t1[j] * Math.cos(a) + t2[j] * Math.sin(a)) * d);
        const l = Math.hypot(p[0], p[1], p[2]); p = p.map(x => x / l);
      }
      if (p[2] < 0.01 || dSpot(p[0], p[1]) < SPOT_R) continue;
      const hot = i === 0;
      const col = hot ? [1.0, 0.96, 0.84] : mixc([1.0, 0.86, 0.6], [1.0, 0.93, 0.76], rnd());
      st.add(K_VIEW, p[0], p[1], p[2], 0, [0, 0, 0.12, -1], col, PX * (hot ? R(10.5, 13) : R(8, 10)), (hot ? 1.0 : 0.82) * R(0.8, 1), 0);
    }
  }
  // the sunspot: a ring of dim orange penumbra stars round a dark umbra
  for (let i = 0; i < 12; i++) {
    const a = (i / 12) * TAU, x = SPOT[0] + Math.cos(a) * SPOT_R * 0.9, y = SPOT[1] + Math.sin(a) * SPOT_R * 0.8;
    st.add(K_VIEW, x, y, Math.sqrt(Math.max(0, 1 - x * x - y * y)), 0, [0, 0, 0.1, -1], [1.0, 0.6, 0.34], PX * 7, 0.55, 0);
  }
  st.add(K_VIEW, SPOT[0], SPOT[1], Math.sqrt(1 - SPOT[0] ** 2 - SPOT[1] ** 2), 0, [0, 0, 0, -1], [0.8, 0.3, 0.2], PX * 7, 0.45, 0);
  // the corona: streamers (screen-aligned) as soft streaks of light with stars strung along them, longest near the
  // equator, short plumes toward the poles
  const NRAYS = 44;
  for (let k = 0; k < NRAYS; k++) {
    const eq = k % 4 !== 0;
    const ang = eq ? (rnd() < 0.5 ? 0 : Math.PI) + (rnd() - 0.5) * 1.9 : (Math.PI / 2) * (rnd() < 0.5 ? 1 : -1) + (rnd() - 0.5) * 1.0;
    const len = eq ? R(0.3, 0.62) : R(0.16, 0.3);
    const n = Math.round(3 + len * 9);
    for (let j = 0; j < n; j++) {
      const u = (j + rnd() * 0.7) / n;
      const rad = 1.03 + len * u;
      const a = ang + (rnd() - 0.5) * 0.04 * (1 + u * 2);
      st.add(K_VIEW, Math.cos(a) * rad, Math.sin(a) * rad, 0, 0.012, [0.02, 0.6, 0, 0], mixc([1.0, 0.9, 0.7], [1.0, 0.76, 0.54], u),
        PX * lerp(7.5, 4.5, u), (1 - u) ** 1.3 * 0.8, 1);
    }
    const v = [];
    for (let j = 0; j <= 5; j++) {
      const u = j / 5, rad = 1.01 + len * 1.05 * u;
      v.push({ k: K_VIEW, p: [Math.cos(ang) * rad, Math.sin(ang) * rad, 0], rate: 0.012, b: [0.02, 0.6, 0, 0], c: [1.0, 0.84, 0.62], a: (eq ? 0.26 : 0.18) * Math.pow(1 - u, 1.2) });
    }
    ln.poly(v, false);
  }
  // a fine diffuse scatter of corona stars between the streamers
  for (let i = 0; i < 70; i++) {
    const a = rnd() * TAU, u = rnd() ** 1.6, rad = 1.04 + 0.48 * u;
    st.add(K_VIEW, Math.cos(a) * rad, Math.sin(a) * rad, 0, 0.012, [0.02, 0.5, 0, 0], [1.0, 0.82, 0.6], PX * 5, 0.42 * (1 - u), 1);
  }
  // prominences: rose arcs of stars standing on the limb
  for (const [ang, w, h, n] of [[2.55, 0.2, 0.26, 52], [-0.75, 0.13, 0.17, 38]]) {
    const v = [];
    for (let s = 0; s < 2; s++) {
      for (let i = 0; i < n / 2; i++) {
        const u = (i + rnd() * 0.5) / (n / 2);
        const a = ang + w * (2 * u - 1) + (rnd() - 0.5) * 0.02;
        const rad = 0.99 + h * Math.pow(Math.sin(Math.PI * u), 0.8) * (s ? 0.82 : 1) + (rnd() - 0.5) * 0.02;
        st.add(K_VIEW, Math.cos(a) * rad, Math.sin(a) * rad, 0.02, 0.012, [0, 0, 0.35, 0], mixc([1.0, 0.56, 0.48], [1.0, 0.74, 0.6], rnd() * 0.5),
          PX * R(6.5, 8.5), R(0.75, 1.0), 1);
      }
    }
    for (let i = 0; i <= 16; i++) {
      const u = i / 16, a = ang + w * (2 * u - 1), rad = 0.99 + h * Math.pow(Math.sin(Math.PI * u), 0.8);
      v.push({ k: K_VIEW, p: [Math.cos(a) * rad, Math.sin(a) * rad, 0.02], rate: 0.012, c: [1.0, 0.6, 0.5], a: 0.38 });
    }
    ln.poly(v, false);
  }
  viewCircle(ln, 1.005, 144, [1.0, 0.8, 0.52], 0.2, 0);
  greatCircles(ln, 3, [1.0, 0.86, 0.62], 0.05, 0.4);
  return {
    extent: 2.3 / 1.5,
    glows: [
      [8.0, 0.34, 0.0, 0.05, [1.0, 0.72, 0.44], 0],
      [3.3, 0.3, 0.0, 0.04, [1.0, 0.84, 0.58], 1],
      [2.1, 0.2, 0.0, 0.03, [1.0, 0.94, 0.8], 1],
    ],
    look: { oblate: 1, night: 1, nightTint: 0, termW: 1, back: 0.03, tw: 0.42, twSpeed: 2.6, limbDark: 1, occ: 0 },
  };
}

const BUILDERS = { saturn: buildSaturn, jupiter: buildJupiter, venus: buildVenus, mercury: buildMercury, mars: buildMars, sun: buildSun };
const SEEDS = { saturn: 61, jupiter: 23, venus: 37, mercury: 11, mars: 47, sun: 5 };

/* ---------------------------------------------------------------------
   the CPU twin of the shader's rest position (spin at spin0, time 0), used to plan the formation
   --------------------------------------------------------------------- */
function restLocal(kind, p, b, R, look, vortex, spin0) {
  if (kind === K_SURF || kind === K_VORTEX || kind === K_HEMI) {
    let lon, lat;
    if (kind === K_SURF) { lon = p[0] + spin0; lat = p[1] + b[0] * Math.sin(p[0] * b[1]); }
    else if (kind === K_HEMI) { const span = Math.PI + 2 * HEMI_MARGIN; lon = -span / 2 + p[0] * span; lat = p[1]; }
    else { lon = vortex[0] + spin0 + vortex[2] * p[0] * Math.cos(p[1]) / Math.cos(vortex[1]); lat = vortex[1] + vortex[3] * p[0] * Math.sin(p[1]); }
    const rr = R * (1 + p[2]);
    return [rr * Math.cos(lat) * Math.sin(lon), rr * Math.sin(lat) * look.oblate, rr * Math.cos(lat) * Math.cos(lon)];
  }
  if (kind === K_RING) return [R * p[0] * Math.sin(p[1]), R * p[2], R * p[0] * Math.cos(p[1])];
  return [p[0] * R, p[1] * R, p[2] * R];
}

/* =====================================================================
   makePlanet
   ===================================================================== */
let texCache = null, texRefs = 0;

export function makePlanet(THREE, kind, fingerBaseRadius, opts = {}) {
  const spec = PLANET_SPECS[kind];
  if (!spec) throw new Error(`makePlanet: unknown kind "${kind}"`);
  const r = fingerBaseRadius;
  const R = spec.size * r;
  const PX = r * 0.0074;          // about one device pixel at the page's zoomed-in framing (camera ~1.84 for r = 0.089)
  const rnd = mulberry32(opts.seed ?? SEEDS[kind]);
  const st = new Stars(rnd), ln = new Lines();
  const built = BUILDERS[kind]({ st, ln, rnd, PX, R, r });
  const look = built.look;
  const vortex = built.vortex || [0, 0, 0.1, 0.1];
  const spin0 = opts.spin0 ?? spec.spin0;
  const spinRate = opts.spinRate ?? spec.spin;

  /* ---------- formation plan: a source and a delay for every star ---------- */
  const N = st.n;
  const src = new Float32Array(N * 3);
  const rest = [];
  for (let i = 0; i < N; i++) {
    rest.push(restLocal(st.a[i * 4], st.p.slice(i * 3, i * 3 + 3), st.b.slice(i * 4, i * 4 + 4), R, look, vortex, spin0));
  }
  const STAGE = [[0, 0.3], [0.14, 0.45]];       // the body forms first, then the rings / corona / prominences
  const sp = opts.sourcePoints;
  if (sp && sp.length) {
    // map coherently: sort the sources and the stars along the source's long axis, then pour rank to rank
    const S = sp.map(v => [v.x, v.y, v.z]);
    const cen = S.reduce((a, v) => [a[0] + v[0] / S.length, a[1] + v[1] / S.length, a[2] + v[2] / S.length], [0, 0, 0]);
    const far = (from) => S.reduce((best, v) => {
      const d = (v[0] - from[0]) ** 2 + (v[1] - from[1]) ** 2 + (v[2] - from[2]) ** 2;
      return d > best[0] ? [d, v] : best;
    }, [-1, null])[1];
    const A = far(cen), B = far(A);
    let ax = [B[0] - A[0], B[1] - A[1], B[2] - A[2]];
    const al = Math.hypot(ax[0], ax[1], ax[2]) || 1;
    ax = ax.map(x => x / al);
    const dot = v => v[0] * ax[0] + v[1] * ax[1] + v[2] * ax[2];
    const sIdx = S.map((v, i) => i).sort((i, j) => dot(S[i]) - dot(S[j]));
    const pIdx = rest.map((v, i) => i).sort((i, j) => dot(rest[i]) - dot(rest[j]));
    const spacing = al / Math.max(1, S.length - 1);
    let maxD = 1e-6;
    const dist = new Float32Array(N);
    pIdx.forEach((pi, rank) => {
      const s = S[sIdx[Math.min(S.length - 1, Math.floor((rank / N) * S.length))]];
      const j = spacing * 0.6;
      src.set([s[0] + (rnd() - 0.5) * j, s[1] + (rnd() - 0.5) * j, s[2] + (rnd() - 0.5) * j], pi * 3);
      const q = rest[pi];
      dist[pi] = Math.hypot(q[0] - src[pi * 3], q[1] - src[pi * 3 + 1], q[2] - src[pi * 3 + 2]);
      maxD = Math.max(maxD, dist[pi]);
    });
    for (let i = 0; i < N; i++) {
      const [lo, hi] = STAGE[st.stage[i]];
      st.a[i * 4 + 3] = lo + (hi - lo) * (0.62 * (dist[i] / maxD) + 0.38 * rnd());
    }
  } else {
    // a scattered shell: each star comes in from roughly its own side of the sky
    for (let i = 0; i < N; i++) {
      const q = rest[i], l = Math.hypot(q[0], q[1], q[2]) || 1;
      let d = [q[0] / l, q[1] / l, q[2] / l];
      const u = rnd() * 2 - 1, t = rnd() * TAU, s = Math.sqrt(1 - u * u);
      d = [d[0] + 0.95 * s * Math.cos(t), d[1] + 0.95 * u, d[2] + 0.95 * s * Math.sin(t)];
      const dl = Math.hypot(d[0], d[1], d[2]) || 1, rad = R * lerp(2.4, 4.6, rnd()) * Math.max(1, built.extent * 0.8);
      src.set([d[0] / dl * rad, d[1] / dl * rad, d[2] / dl * rad], i * 3);
      const [lo, hi] = STAGE[st.stage[i]];
      st.a[i * 4 + 3] = lo + (hi - lo) * rnd();
    }
  }

  /* ---------- shared uniforms ---------- */
  if (!texCache) texCache = { glow: gradientTexture(THREE, GLOW_STOPS), soft: gradientTexture(THREE, SOFT_STOPS) };
  texRefs++;
  const light = (opts.lightDir ? opts.lightDir.clone() : new THREE.Vector3(0.62, 0.5, 0.6)).normalize();
  const U = {
    uTime: { value: 0 }, uSpin: { value: spin0 }, uReveal: { value: 1 }, uOpacity: { value: 1 }, uR: { value: R },
    uOblate: { value: look.oblate }, uNight: { value: look.night }, uNightTint: { value: look.nightTint },
    uTermW: { value: look.termW }, uTw: { value: look.tw }, uTwSpeed: { value: look.twSpeed },
    uLimbDark: { value: look.limbDark }, uLight: { value: light },
    uVortex: { value: new THREE.Vector4(...vortex) }, uRings: { value: built.rings ? 1 : 0 },
  };
  const additive = {
    transparent: true, depthWrite: false, depthTest: true,
    blending: THREE.CustomBlending, blendEquation: THREE.AddEquation, blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor,
  };
  const ro = opts.renderOrder ?? 5;
  const group = new THREE.Group();
  group.name = `planet-${kind}`;

  /* ---------- stars ---------- */
  const sg = new THREE.BufferGeometry();
  sg.setAttribute('position', new THREE.Float32BufferAttribute(st.p, 3));
  sg.setAttribute('aSrc', new THREE.BufferAttribute(src, 3));
  sg.setAttribute('aA', new THREE.Float32BufferAttribute(st.a, 4));
  sg.setAttribute('aB', new THREE.Float32BufferAttribute(st.b, 4));
  sg.setAttribute('aColor', new THREE.Float32BufferAttribute(st.c, 3));
  sg.setAttribute('aSA', new THREE.Float32BufferAttribute(st.s, 2));
  const starMat = new THREE.ShaderMaterial({
    uniforms: {
      ...U, uBack: { value: look.back }, uFade: { value: 1 }, uMap: { value: texCache.glow },
      uViewH: { value: 1000 }, uMinS: { value: 2 }, uDotTo: { value: 7 }, uMaxPt: { value: 256 },
      uAtten: { value: opts.atten ?? 0.85 }, uRef: { value: opts.refDistance ?? 1.84 },
    },
    vertexShader: STAR_VS, fragmentShader: STAR_FS, ...additive,
  });
  const stars = new THREE.Points(sg, starMat);
  stars.frustumCulled = false; stars.renderOrder = ro + 3;
  const vp = new THREE.Vector4();
  let maxPtKnown = false;
  stars.onBeforeRender = (renderer) => {
    renderer.getCurrentViewport(vp);
    const pr = renderer.getPixelRatio();
    starMat.uniforms.uViewH.value = vp.w;
    starMat.uniforms.uMinS.value = 2 * pr;
    starMat.uniforms.uDotTo.value = 7 * pr;
    if (!maxPtKnown) {
      const gl = renderer.getContext(), rng = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE);
      starMat.uniforms.uMaxPt.value = Math.min((rng && rng[1]) || 256, 300 * pr);
      maxPtKnown = true;
    }
  };

  /* ---------- lines ---------- */
  const lg = new THREE.BufferGeometry();
  lg.setAttribute('position', new THREE.Float32BufferAttribute(ln.p, 3));
  lg.setAttribute('aA', new THREE.Float32BufferAttribute(ln.a, 4));
  lg.setAttribute('aB', new THREE.Float32BufferAttribute(ln.b, 4));
  lg.setAttribute('aColor', new THREE.Float32BufferAttribute(ln.c, 3));
  lg.setAttribute('aAl', new THREE.Float32BufferAttribute(ln.al, 1));
  const lineMat = new THREE.ShaderMaterial({
    uniforms: { ...U, uBack: { value: 0.0 }, uFade: { value: 1 } },
    vertexShader: LINE_VS, fragmentShader: LINE_FS, ...additive,
  });
  const lines = new THREE.LineSegments(lg, lineMat);
  lines.frustumCulled = false; lines.renderOrder = ro + 2;

  /* ---------- glows: camera-facing quads with the radial-gradient textures ---------- */
  const glows = built.glows;
  const gq = new THREE.BufferGeometry();
  {
    const pos = [], uv = [], aG = [], aG2 = [], idx = [];
    glows.forEach(([d, a, off, fl, col, tex], k) => {
      for (const [x, y] of [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]) {
        pos.push(x, y, 0); uv.push(x + 0.5, y + 0.5); aG.push(d, a, off, fl); aG2.push(col[0], col[1], col[2], tex);
      }
      idx.push(k * 4, k * 4 + 1, k * 4 + 2, k * 4, k * 4 + 2, k * 4 + 3);
    });
    gq.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    gq.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
    gq.setAttribute('aG', new THREE.Float32BufferAttribute(aG, 4));
    gq.setAttribute('aG2', new THREE.Float32BufferAttribute(aG2, 4));
    gq.setIndex(idx);
  }
  const glowMat = new THREE.ShaderMaterial({
    uniforms: { uR: U.uR, uOpacity: U.uOpacity, uTime: U.uTime, uLight: U.uLight, uFade: { value: 1 }, uGlow: { value: texCache.glow }, uSoft: { value: texCache.soft } },
    vertexShader: GLOW_VS, fragmentShader: GLOW_FS, ...additive,
  });
  const glow = new THREE.Mesh(gq, glowMat);
  glow.frustumCulled = false; glow.renderOrder = ro + 1;

  /* ---------- the occluder: a faint dark disc, so the body hides the sky and dust behind it (v2's sphere did) ---------- */
  let occ = null;
  if (opts.occluder !== false && look.occ > 0) {
    const og = new THREE.PlaneGeometry(1, 1);
    const occMat = new THREE.ShaderMaterial({
      uniforms: { uR: U.uR, uOpacity: U.uOpacity, uFade: { value: 1 }, uA: { value: opts.occluderAlpha ?? look.occ }, uCol: { value: new THREE.Color(0.016, 0.012, 0.024) } },
      vertexShader: OCC_VS, fragmentShader: OCC_FS,
      transparent: true, depthWrite: false, depthTest: true, blending: THREE.NormalBlending,
    });
    occ = new THREE.Mesh(og, occMat);
    occ.frustumCulled = false; occ.renderOrder = ro;
    group.add(occ);
  }
  group.add(glow, lines, stars);

  /* ---------- API ---------- */
  let clock = 0, reveal = 1, disposed = false;
  const applyReveal = () => {
    starMat.uniforms.uReveal.value = reveal;
    lineMat.uniforms.uFade.value = smooth(0.72, 1.0, reveal);
    glowMat.uniforms.uFade.value = smooth(0.2, 1.0, reveal);
    if (occ) occ.material.uniforms.uFade.value = smooth(0.3, 1.0, reveal);
  };
  function smooth(a, b, x) { const t = clamp01((x - a) / (b - a)); return t * t * (3 - 2 * t); }
  const setClock = s => {
    clock = s;
    U.uTime.value = clock;
    U.uSpin.value = spin0 + clock * spinRate;
  };
  setClock(0);
  applyReveal();

  return {
    kind,
    group,
    radius: R,
    ringOuter: built.ringOuter ? built.ringOuter * R : undefined,
    extent: built.extent * R,                        // the visual radius: rings, corona or atmosphere included
    counts: { points: N, segments: ln.segments },
    quaternion: planetQuaternion(THREE, kind),       // suggested orientation (not applied)
    /** dt drives the planet's own clock (spin, drift, swirl, twinkle), so it always appears in its hero pose.
        t and camera are accepted for the contract; light is view-space, so no camera math is needed. */
    update(dt /* , t, camera */) {
      setClock(clock + Math.min(Math.max(dt || 0, 0), 0.1));
    },
    setClock,
    setOpacity(a) { U.uOpacity.value = clamp01(a); group.visible = a > 0.001; },
    setReveal(p) { reveal = clamp01(p); applyReveal(); },
    setLight(v) { U.uLight.value.copy(v).normalize(); },
    dispose() {
      if (disposed) return;
      disposed = true;
      group.removeFromParent();
      sg.dispose(); lg.dispose(); gq.dispose(); starMat.dispose(); lineMat.dispose(); glowMat.dispose();
      if (occ) { occ.geometry.dispose(); occ.material.dispose(); }
      if (--texRefs <= 0 && texCache) { texCache.glow.dispose(); texCache.soft.dispose(); texCache = null; texRefs = 0; }
    },
  };
}
