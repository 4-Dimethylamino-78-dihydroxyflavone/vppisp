// vppisp compute kernels. The ISP functions are an f32 mirror of isp.rs (PPISP,
// NVIDIA, Apache-2.0); keep the two in step.

alias Cam = array<f32, 27>;
alias M3 = array<vec3<f32>, 3>;   // rows

const ZM = array<vec4<f32>, 4>(
    vec4<f32>(0.0480542, -0.0043631, -0.0043631, 0.0481283),
    vec4<f32>(0.0580570, -0.0179872, -0.0179872, 0.0431061),
    vec4<f32>(0.0433336, -0.0180537, -0.0180537, 0.0580500),
    vec4<f32>(0.0128369, -0.0034654, -0.0034654, 0.0128158),
);

fn softplus(x: f32) -> f32 {
    return select(log(1.0 + exp(x)), x, x > 20.0);
}

fn vig(c: Cam, k: u32, uv: vec2<f32>) -> f32 {
    let b = 5u * k;
    let d = uv - vec2<f32>(c[b], c[b + 1u]);
    let r2 = dot(d, d);
    return clamp(1.0 + r2 * (c[b + 2u] + r2 * (c[b + 3u] + r2 * c[b + 4u])), 0.0, 1.0);
}

fn crf(c: Cam, k: u32, x0: f32) -> f32 {
    let b = 15u + 4u * k;
    let tau = 0.3 + softplus(c[b]);
    let eta = 0.3 + softplus(c[b + 1u]);
    let gam = 0.1 + softplus(c[b + 2u]);
    let xi = 1.0 / (1.0 + exp(-c[b + 3u]));
    let a = eta * xi / (tau * (1.0 - xi) + eta * xi);
    let x = clamp(x0, 0.0, 1.0);
    var f: f32;
    if (x <= xi) {
        f = a * pow(max(x / xi, 1e-12), tau);
    } else {
        f = 1.0 - (1.0 - a) * pow(max((1.0 - x) / (1.0 - xi), 1e-12), eta);
    }
    return pow(max(f, 1e-12), gam);
}

fn mul3(a: M3, b: M3) -> M3 {
    var c: M3;
    for (var i = 0u; i < 3u; i++) {
        c[i] = vec3<f32>(dot(a[i], vec3<f32>(b[0].x, b[1].x, b[2].x)),
                         dot(a[i], vec3<f32>(b[0].y, b[1].y, b[2].y)),
                         dot(a[i], vec3<f32>(b[0].z, b[1].z, b[2].z)));
    }
    return c;
}

fn homography(z: array<f32, 8>) -> M3 {
    var d: array<vec2<f32>, 4>;
    for (var k = 0u; k < 4u; k++) {
        let m = ZM[k];
        d[k] = vec2<f32>(m.x * z[2u * k] + m.y * z[2u * k + 1u], m.z * z[2u * k] + m.w * z[2u * k + 1u]);
    }
    let tb = vec3<f32>(d[0].x, d[0].y, 1.0);
    let tr = vec3<f32>(1.0 + d[1].x, d[1].y, 1.0);
    let tg = vec3<f32>(d[2].x, 1.0 + d[2].y, 1.0);
    let tw = vec3<f32>(1.0 / 3.0 + d[3].x, 1.0 / 3.0 + d[3].y, 1.0);
    let tm = M3(vec3<f32>(tb.x, tr.x, tg.x), vec3<f32>(tb.y, tr.y, tg.y), vec3<f32>(tb.z, tr.z, tg.z));
    let cx = M3(vec3<f32>(0.0, -tw.z, tw.y), vec3<f32>(tw.z, 0.0, -tw.x), vec3<f32>(-tw.y, tw.x, 0.0));
    let m = mul3(cx, tm);
    var k = cross(m[0], m[1]);
    if (dot(k, k) < 1e-20) { k = cross(m[0], m[2]); }
    if (dot(k, k) < 1e-20) { k = cross(m[1], m[2]); }
    let tk = M3(tm[0] * k, tm[1] * k, tm[2] * k);
    let sinv = M3(vec3<f32>(-1.0, -1.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0));
    let h = mul3(tk, sinv);
    let n = h[2].z;
    return M3(h[0] / n, h[1] / n, h[2] / n);
}

fn apply_cc(h: M3, rgb: vec3<f32>) -> vec3<f32> {
    let i = rgb.x + rgb.y + rgb.z;
    let x = vec3<f32>(rgb.x, rgb.y, i);
    let y = vec3<f32>(dot(h[0], x), dot(h[1], x), dot(h[2], x));
    let s = i / (y.z + 1e-5);
    return vec3<f32>(y.x * s, y.y * s, y.z * s - y.x * s - y.y * s);
}

fn forward(c: Cam, dt: f32, h: M3, uv: vec2<f32>, l: vec3<f32>) -> vec3<f32> {
    let e = exp2(dt);
    let x = vec3<f32>(l.x * e * vig(c, 0u, uv), l.y * e * vig(c, 1u, uv), l.z * e * vig(c, 2u, uv));
    let y = apply_cc(h, x);
    return vec3<f32>(crf(c, 0u, y.x), crf(c, 1u, y.y), crf(c, 2u, y.z));
}

// ---------------------------------------------------------------- z-buffer
// One invocation per (point, frame): atomicMin of the depth's bit pattern,
// which orders like the float for positive values.

struct ZU { npts: u32, zw: u32, blocks_x: u32, pad: u32 }

@group(0) @binding(0) var<storage, read> zpts: array<f32>;
@group(0) @binding(1) var<storage, read> zcams: array<f32>;   // 24 per frame
@group(0) @binding(2) var<storage, read_write> zb: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> zu: ZU;

@compute @workgroup_size(256)
fn zbuffer(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let i = (wg.z * zu.blocks_x + wg.x) * 256u + lid;
    if (i >= zu.npts) { return; }
    let c = wg.y * 24u;
    let p = vec3<f32>(zpts[3u * i], zpts[3u * i + 1u], zpts[3u * i + 2u]);
    let d = p - vec3<f32>(zcams[c + 9u], zcams[c + 10u], zcams[c + 11u]);
    let z = zcams[c + 6u] * d.x + zcams[c + 7u] * d.y + zcams[c + 8u] * d.z;
    if (z <= 1e-6) { return; }
    let x = zcams[c] * d.x + zcams[c + 1u] * d.y + zcams[c + 2u] * d.z;
    let y = zcams[c + 3u] * d.x + zcams[c + 4u] * d.y + zcams[c + 5u] * d.z;
    let xn = (zcams[c + 12u] * x / z + zcams[c + 14u] + zcams[c + 16u]) * zcams[c + 18u];
    let yn = (zcams[c + 13u] * y / z + zcams[c + 15u] + zcams[c + 17u]) * zcams[c + 19u];
    if (xn < 0.0 || xn >= 1.0 || yn < 0.0 || yn >= 1.0) { return; }
    let zh = bitcast<u32>(zcams[c + 20u]);
    let base = bitcast<u32>(zcams[c + 21u]);
    let cell = min(u32(yn * f32(zh)), zh - 1u) * zu.zw + min(u32(xn * f32(zu.zw)), zu.zw - 1u);
    atomicMin(&zb[base + cell], bitcast<u32>(z));
}

// ------------------------------------------------------ normal equations
// One workgroup per chunk of observations (a chunk never spans two frames).
// Each invocation accumulates J^T W J (packed upper triangle), J^T W r and
// the Huber cost privately; the workgroup then reduces entry by entry.
// Jacobians are central differences. mode bit 0: frame parameters (9, H
// rebuilt from the perturbed latents) instead of camera ones; bit 1: cost only.

const STRIDE: u32 = 406u;   // 378 triangle + 27 gradient + cost
const TRI: u32 = 378u;

struct NP {
    cam: array<vec4<f32>, 7>,
    free: array<vec4<u32>, 7>,
    mode: u32,
    nfree: u32,
    huber: f32,
    pad: u32,
}

@group(0) @binding(10) var<storage, read> obs: array<vec4<f32>>;     // 2 per observation
@group(0) @binding(11) var<storage, read> order: array<u32>;
@group(0) @binding(12) var<storage, read> chunks: array<vec4<u32>>;  // start, end, slot, -
@group(0) @binding(13) var<storage, read> rad: array<vec4<f32>>;
@group(0) @binding(14) var<storage, read> frames: array<f32>;        // 20 per frame
@group(0) @binding(15) var<uniform> np: NP;
@group(0) @binding(16) var<storage, read_write> partials: array<f32>;

var<workgroup> red: array<f32, 64>;

fn tri(a: u32, b: u32, k: u32) -> u32 {
    return a * (2u * k - a + 1u) / 2u + (b - a);
}

fn frame_h(f: u32) -> M3 {
    let b = f * 20u + 9u;
    return M3(vec3<f32>(frames[b], frames[b + 1u], frames[b + 2u]),
              vec3<f32>(frames[b + 3u], frames[b + 4u], frames[b + 5u]),
              vec3<f32>(frames[b + 6u], frames[b + 7u], frames[b + 8u]));
}

fn eval_frame(c: Cam, q: array<f32, 9>, uv: vec2<f32>, l: vec3<f32>) -> vec3<f32> {
    var z: array<f32, 8>;
    for (var j = 0u; j < 8u; j++) { z[j] = q[j + 1u]; }
    return forward(c, q[0], homography(z), uv, l);
}

@compute @workgroup_size(64)
fn normal(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let ch = chunks[wg.x];
    let frame_mode = (np.mode & 1u) != 0u;
    let cost_only = (np.mode & 2u) != 0u;
    let k = select(np.nfree, 9u, frame_mode);
    let hub = np.huber;
    var c0: Cam;
    for (var j = 0u; j < 27u; j++) { c0[j] = np.cam[j / 4u][j % 4u]; }

    var acc: array<f32, 406>;
    for (var i = ch.x + tid; i < ch.y; i += 64u) {
        let o = order[i];
        let a0 = obs[2u * o];
        let px = obs[2u * o + 1u].xyz;
        let l = rad[bitcast<u32>(a0.x)].xyz;
        let f = bitcast<u32>(a0.y);
        let uv = a0.zw;
        var q: array<f32, 9>;
        for (var j = 0u; j < 9u; j++) { q[j] = frames[f * 20u + j]; }
        let hf = frame_h(f);
        var r0: vec3<f32>;
        if (frame_mode) { r0 = eval_frame(c0, q, uv, l) - px; } else { r0 = forward(c0, q[0], hf, uv, l) - px; }
        var w: vec3<f32>;
        for (var c = 0u; c < 3u; c++) {
            let a = abs(r0[c]);
            if (a <= hub) { acc[405] += 0.5 * r0[c] * r0[c]; w[c] = 1.0; }
            else { acc[405] += hub * (a - 0.5 * hub); w[c] = hub / a; }
        }
        if (cost_only) { continue; }
        var jac: array<vec3<f32>, 27>;
        for (var a = 0u; a < k; a++) {
            if (frame_mode) {
                var qp = q;
                let h = 1e-3 * max(1.0, abs(q[a]));
                qp[a] = q[a] + h;
                let rp = eval_frame(c0, qp, uv, l);
                qp[a] = q[a] - h;
                let rm = eval_frame(c0, qp, uv, l);
                jac[a] = (rp - rm) / (2.0 * h);
            } else {
                let idx = np.free[a / 4u][a % 4u];
                var cp = c0;
                let h = 1e-3 * max(1.0, abs(c0[idx]));
                cp[idx] = c0[idx] + h;
                let rp = forward(cp, q[0], hf, uv, l);
                cp[idx] = c0[idx] - h;
                let rm = forward(cp, q[0], hf, uv, l);
                jac[a] = (rp - rm) / (2.0 * h);
            }
        }
        for (var a = 0u; a < k; a++) {
            let wj = w * jac[a];
            acc[TRI + a] += dot(wj, r0);
            for (var b = a; b < k; b++) {
                acc[tri(a, b, k)] += dot(wj, jac[b]);
            }
        }
    }

    let ntri = k * (k + 1u) / 2u;
    for (var e = 0u; e < STRIDE; e++) {
        let used = e == 405u || (!cost_only && (e < ntri || (e >= TRI && e < TRI + k)));
        if (!used) { continue; }
        red[tid] = acc[e];
        workgroupBarrier();
        for (var s = 32u; s > 0u; s >>= 1u) {
            if (tid < s) { red[tid] += red[tid + s]; }
            workgroupBarrier();
        }
        if (tid == 0u) { partials[wg.x * STRIDE + e] = red[0]; }
        workgroupBarrier();
    }
}

// --------------------------------------------------------------- correct
// Pixel -> radiance through the frame's ISP -> pixel through the reference
// ISP (exposure 0, identity colour, no vignetting, camera CRF via LUT).

struct CU {
    hi: array<vec4<f32>, 3>,
    cam: array<vec4<f32>, 7>,
    gain: f32,
    w: u32,
    h: u32,
    vig: u32,
    shoulder: vec4<f32>,   // knee, b, -, - (see isp::shoulder)
}

@group(0) @binding(20) var<storage, read> src: array<u32>;
@group(0) @binding(21) var<storage, read_write> dst: array<u32>;
@group(0) @binding(22) var<storage, read> luts: array<f32>;   // inv 3x256, fwd 3x4097
@group(0) @binding(23) var<uniform> cu: CU;

@compute @workgroup_size(16, 16)
fn correct(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= cu.w || g.y >= cu.h) { return; }
    let i = g.y * cu.w + g.x;
    let p = src[i];
    let lin = vec3<f32>(luts[p & 255u], luts[256u + ((p >> 8u) & 255u)], luts[512u + ((p >> 16u) & 255u)]);
    let c = apply_cc(M3(cu.hi[0].xyz, cu.hi[1].xyz, cu.hi[2].xyz), lin);
    var cam: Cam;
    for (var j = 0u; j < 27u; j++) { cam[j] = cu.cam[j / 4u][j % 4u]; }
    let mx = f32(max(cu.w, cu.h));
    let uv = vec2<f32>((f32(g.x) + 0.5 - 0.5 * f32(cu.w)) / mx, (f32(g.y) + 0.5 - 0.5 * f32(cu.h)) / mx);
    var out = 0xff000000u;
    for (var k = 0u; k < 3u; k++) {
        var v = 1.0;
        if (cu.vig != 0u) { v = max(vig(cam, k, uv), 0.2); }
        var r = c[k] * cu.gain / v;
        if (r > cu.shoulder.x) { r = cu.shoulder.x + (r - cu.shoulder.x) / (1.0 + (r - cu.shoulder.x) * cu.shoulder.y); }
        let x = clamp(r, 0.0, 1.0) * 4096.0;
        let j = min(u32(x), 4095u);
        let b = 768u + k * 4097u;
        let y = mix(luts[b + j], luts[b + j + 1u], x - f32(j));
        out |= u32(clamp(y, 0.0, 1.0) * 255.0 + 0.5) << (8u * k);
    }
    dst[i] = out;
}
