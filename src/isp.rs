//! The PPISP image formation model (Deutsch et al. 2026, NVIDIA, Apache-2.0),
//! reimplemented from `ppisp_math.cuh` / `tests/torch_reference.py`:
//!
//!   image = CRF( CC( vignette(uv) * 2^dt * L ) )
//!
//! exposure dt and 8 colour latents per frame; vignetting (3 channels x
//! [cx, cy, a1, a2, a3]) and CRF (3 channels x [tau, eta, gamma, xi], raw)
//! per camera. Every stage has a closed-form inverse, which is what lets a
//! point cloud stand in for PPISP's radiance field.

pub const FRAME: usize = 9;
pub const CAM: usize = 27;
const VIG: usize = 0;
const CRF: usize = 15;

/// ZCA preconditioners for the latent pairs [B, R, G, N] (ppisp_math.cuh:422).
const Z: [[[f64; 2]; 2]; 4] = [
    [[0.0480542, -0.0043631], [-0.0043631, 0.0481283]],
    [[0.0580570, -0.0179872], [-0.0179872, 0.0431061]],
    [[0.0433336, -0.0180537], [-0.0180537, 0.0580500]],
    [[0.0128369, -0.0034654], [-0.0034654, 0.0128158]],
];

/// Camera parameters at their starting point: no vignetting, and a CRF of
/// x^(1/2.2) rather than PPISP's identity. JPEGs are display-referred, so this
/// puts the latent radiance near linear from the first iteration.
pub fn camera_init() -> [f64; CAM] {
    let mut p = [0.0; CAM];
    for k in 0..3 {
        // tau = eta = 0.3 + softplus(r) = 1;  gamma = 0.1 + softplus(r) = 1/2.2;  xi = 0.5
        p[CRF + 4 * k] = 0.013659;
        p[CRF + 4 * k + 1] = 0.013659;
        p[CRF + 4 * k + 2] = ((1.0f64 / 2.2 - 0.1).exp() - 1.0).ln();
        p[CRF + 4 * k + 3] = 0.0;
    }
    p
}

fn softplus(x: f64) -> f64 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// Real chromaticity offsets of the 4 control points from the 8 latents.
pub fn offsets(z: &[f64]) -> [[f64; 2]; 4] {
    let mut d = [[0.0; 2]; 4];
    for k in 0..4 {
        d[k] = [Z[k][0][0] * z[2 * k] + Z[k][0][1] * z[2 * k + 1], Z[k][1][0] * z[2 * k] + Z[k][1][1] * z[2 * k + 1]];
    }
    d
}

/// Latents for given offsets (inverse of `offsets`); used to re-centre the gauge.
pub fn latents(d: &[[f64; 2]; 4]) -> [f64; 8] {
    let mut z = [0.0; 8];
    for k in 0..4 {
        let m = Z[k];
        let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        z[2 * k] = (m[1][1] * d[k][0] - m[0][1] * d[k][1]) / det;
        z[2 * k + 1] = (-m[1][0] * d[k][0] + m[0][0] * d[k][1]) / det;
    }
    z
}

/// The RGI-space homography of a frame's colour latents: four chromaticity
/// control points (B, R, G, neutral) moved by their offsets, solved as a DLT.
pub fn homography(z: &[f64]) -> [[f64; 3]; 3] {
    let d = offsets(z);
    let t = |s: [f64; 3], o: [f64; 2]| [s[0] + o[0], s[1] + o[1], s[2]];
    let (tb, tr, tg) = (t([0.0, 0.0, 1.0], d[0]), t([1.0, 0.0, 1.0], d[1]), t([0.0, 1.0, 1.0], d[2]));
    let tw = t([1.0 / 3.0, 1.0 / 3.0, 1.0], d[3]);
    // T = [tB tR tG] as columns; M = [tW]x T; k = null vector of M.
    let tm = [[tb[0], tr[0], tg[0]], [tb[1], tr[1], tg[1]], [tb[2], tr[2], tg[2]]];
    let cx = [[0.0, -tw[2], tw[1]], [tw[2], 0.0, -tw[0]], [-tw[1], tw[0], 0.0]];
    let m = mul(&cx, &tm);
    let cross = |a: [f64; 3], b: [f64; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let norm2 = |v: [f64; 3]| v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    let mut k = cross(m[0], m[1]);
    if norm2(k) < 1e-20 {
        k = cross(m[0], m[2]);
    }
    if norm2(k) < 1e-20 {
        k = cross(m[1], m[2]);
    }
    let tk = [[tm[0][0] * k[0], tm[0][1] * k[1], tm[0][2] * k[2]],
              [tm[1][0] * k[0], tm[1][1] * k[1], tm[1][2] * k[2]],
              [tm[2][0] * k[0], tm[2][1] * k[1], tm[2][2] * k[2]]];
    let s_inv = [[-1.0, -1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let mut h = mul(&tk, &s_inv);
    let n = h[2][2];
    for r in h.iter_mut() {
        for v in r.iter_mut() {
            *v /= n;
        }
    }
    h
}

fn mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    c
}

pub fn invert(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let c = |i: usize, j: usize| m[(i + 1) % 3][(j + 1) % 3] * m[(i + 2) % 3][(j + 2) % 3] - m[(i + 1) % 3][(j + 2) % 3] * m[(i + 2) % 3][(j + 1) % 3];
    let det = m[0][0] * c(0, 0) + m[0][1] * c(0, 1) + m[0][2] * c(0, 2);
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[j][i] = c(i, j) / det;
        }
    }
    r
}

/// Apply an RGI homography, preserving intensity R+G+B (so colour never
/// trades against exposure).
#[inline]
pub fn apply_cc(h: &[[f64; 3]; 3], rgb: [f64; 3]) -> [f64; 3] {
    let i = rgb[0] + rgb[1] + rgb[2];
    let x = [rgb[0], rgb[1], i];
    let y = [
        h[0][0] * x[0] + h[0][1] * x[1] + h[0][2] * x[2],
        h[1][0] * x[0] + h[1][1] * x[1] + h[1][2] * x[2],
        h[2][0] * x[0] + h[2][1] * x[1] + h[2][2] * x[2],
    ];
    let s = i / (y[2] + 1e-5);
    let (r, g, ii) = (y[0] * s, y[1] * s, y[2] * s);
    [r, g, ii - r - g]
}

/// Per-channel falloff at `uv` = (pixel - centre) / max(W, H).
#[inline]
pub fn vignette(cam: &[f64], k: usize, uv: [f64; 2]) -> f64 {
    let p = &cam[VIG + 5 * k..VIG + 5 * k + 5];
    let (dx, dy) = (uv[0] - p[0], uv[1] - p[1]);
    let r2 = dx * dx + dy * dy;
    (1.0 + r2 * (p[2] + r2 * (p[3] + r2 * p[4]))).clamp(0.0, 1.0)
}

struct Curve {
    tau: f64,
    eta: f64,
    gamma: f64,
    xi: f64,
    a: f64,
}

fn curve(cam: &[f64], k: usize) -> Curve {
    let p = &cam[CRF + 4 * k..CRF + 4 * k + 4];
    let (tau, eta, gamma) = (0.3 + softplus(p[0]), 0.3 + softplus(p[1]), 0.1 + softplus(p[2]));
    let xi = 1.0 / (1.0 + (-p[3]).exp());
    let a = eta * xi / (tau * (1.0 - xi) + eta * xi);
    Curve { tau, eta, gamma, xi, a }
}

/// Toe/shoulder power curve then a gamma: monotone, C1, f(0)=0, f(1)=1.
#[inline]
pub fn crf(cam: &[f64], k: usize, x: f64) -> f64 {
    let c = curve(cam, k);
    let x = x.clamp(0.0, 1.0);
    let f = if x <= c.xi {
        c.a * (x / c.xi).powf(c.tau)
    } else {
        1.0 - (1.0 - c.a) * ((1.0 - x) / (1.0 - c.xi)).powf(c.eta)
    };
    f.max(0.0).powf(c.gamma)
}

#[inline]
pub fn crf_inv(cam: &[f64], k: usize, z: f64) -> f64 {
    let c = curve(cam, k);
    let y = z.clamp(0.0, 1.0).powf(1.0 / c.gamma);
    if y <= c.a {
        c.xi * (y / c.a).powf(1.0 / c.tau)
    } else {
        1.0 - (1.0 - c.xi) * ((1.0 - y) / (1.0 - c.a)).max(0.0).powf(1.0 / c.eta)
    }
}

/// Radiance -> pixel for one frame (`h` = that frame's homography).
#[inline]
pub fn forward(cam: &[f64], dt: f64, h: &[[f64; 3]; 3], uv: [f64; 2], l: [f64; 3]) -> [f64; 3] {
    let e = dt.exp2();
    let x = [l[0] * e * vignette(cam, 0, uv), l[1] * e * vignette(cam, 1, uv), l[2] * e * vignette(cam, 2, uv)];
    let y = apply_cc(h, x);
    [crf(cam, 0, y[0]), crf(cam, 1, y[1]), crf(cam, 2, y[2])]
}

/// Pixel -> radiance for one frame (`hi` = the inverse homography).
#[inline]
pub fn inverse(cam: &[f64], dt: f64, hi: &[[f64; 3]; 3], uv: [f64; 2], px: [f64; 3]) -> [f64; 3] {
    let y = [crf_inv(cam, 0, px[0]), crf_inv(cam, 1, px[1]), crf_inv(cam, 2, px[2])];
    let x = apply_cc(hi, y);
    let e = (-dt).exp2();
    [0, 1, 2].map(|k| x[k] * e / vignette(cam, k, uv).max(0.2))
}
