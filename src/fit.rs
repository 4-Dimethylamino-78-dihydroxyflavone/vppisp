//! Fitting PPISP to multi-view correspondences from a dvlt first pass.
//!
//! PPISP learns its parameters inside a radiance-field trainer, where the
//! field supplies each pixel's true radiance. Here a sample of the dvlt cloud
//! plays that part: every point gets a free linear radiance, observed in the
//! frames that see it, and the problem alternates between
//!   radiance  per point, the median of its inverse-ISP observations;
//!   frames    9 parameters each, Levenberg-Marquardt, batched over frames;
//!   camera    27 shared parameters, Levenberg-Marquardt over everything.
//! The normal equations come from the engine (GPU or CPU); the tiny solves,
//! priors and gauge stay here.

use crate::engine::{self, Engine, Mode, Sys};
use crate::io::{Cam, Geo};
use crate::isp::{self, CAM, FRAME};
use image::RgbImage;
use rayon::prelude::*;

/// A point counts as visible when it is within this factor of the frame's
/// z-buffer depth. dvlt's views disagree on depth by a few percent.
const DEPTH_TOL: f32 = 1.05;
/// z-buffer columns; rows follow the frame's aspect.
const ZBUF_W: usize = 256;
/// Pull of each frame's exposure towards its EXIF value, relative to the mean
/// data cost. Weak on purpose: it pins PPISP's exposure-vs-CRF power ambiguity
/// (supp. A.3) without overriding the images.
const W_EV: f64 = 1e-4;
/// Ridge on colour latents, so frames with few observations stay near identity.
const W_LAT: f64 = 1e-5;
const SWEEPS: usize = 6;
const LM_ITERS: usize = 8;

pub struct Obs {
    pub p: u32,
    pub f: u16,
    pub uv: [f32; 2],
    pub px: [f32; 3],
}

/// Sample up to `n` points (skipping pure black, i.e. vsky-keyed sky) and
/// record every frame that sees each one, by z-buffer, with the 3x3
/// neighbourhood mean colour at working resolution. Saturated or near-black
/// observations carry no gradient through PPISP's clamps and are dropped.
/// Returns observations sorted by point, and the point count.
#[allow(clippy::too_many_arguments)]
pub fn correspond(
    engine: &mut Engine,
    xyz: &[[f32; 3]],
    rgb: &[[u8; 3]],
    cams: &[Cam],
    geo: &[Geo],
    ims: &[RgbImage],
    dims: &[(u32, u32)],
    n: usize,
) -> (Vec<Obs>, usize) {
    let zh: Vec<usize> = geo.iter().map(|g| ((ZBUF_W as f32 * (g.inv[0] / g.inv[1])).round() as usize).max(1)).collect();
    let zbs = engine.zbuffers(xyz, cams, geo, ZBUF_W, &zh);

    let valid: Vec<usize> = (0..xyz.len()).filter(|&i| rgb[i] != [0, 0, 0]).collect();
    let step = (valid.len() / n.max(1)).max(1);
    let sample: Vec<usize> = valid.iter().step_by(step).copied().collect();

    let per_frame: Vec<Vec<Obs>> = (0..cams.len())
        .into_par_iter()
        .map(|f| {
            let (cam, g, im, zb, zh) = (&cams[f], &geo[f], &ims[f], &zbs[f], zh[f]);
            let (w, h) = (im.width() as i64, im.height() as i64);
            let (sw, sh) = (dims[f].0 as f32, dims[f].1 as f32);
            let m = sw.max(sh);
            let mut out = Vec::new();
            for (si, &i) in sample.iter().enumerate() {
                let Some((z, xn, yn)) = cam.project(g, xyz[i]) else { continue };
                let cell = ((yn * zh as f32) as usize).min(zh - 1) * ZBUF_W + ((xn * ZBUF_W as f32) as usize).min(ZBUF_W - 1);
                if z > zb[cell] * DEPTH_TOL {
                    continue;
                }
                let (cx, cy) = ((xn * w as f32) as i64, (yn * h as f32) as i64);
                let (mut acc, mut cnt) = ([0f32; 3], 0f32);
                for y in (cy - 1).max(0)..=(cy + 1).min(h - 1) {
                    for x in (cx - 1).max(0)..=(cx + 1).min(w - 1) {
                        let q = im.get_pixel(x as u32, y as u32);
                        for k in 0..3 {
                            acc[k] += q[k] as f32;
                        }
                        cnt += 1.0;
                    }
                }
                let px = acc.map(|v| v / cnt / 255.0);
                if px.iter().any(|&v| v > 0.97) || px.iter().all(|&v| v < 0.03) {
                    continue;
                }
                out.push(Obs { p: si as u32, f: f as u16, uv: [(xn - 0.5) * sw / m, (yn - 0.5) * sh / m], px });
            }
            out
        })
        .collect();

    let mut obs: Vec<Obs> = per_frame.into_iter().flatten().collect();
    let mut cnt = vec![0u32; sample.len()];
    for o in &obs {
        cnt[o.p as usize] += 1;
    }
    obs.retain(|o| cnt[o.p as usize] >= 3);
    let mut remap = vec![u32::MAX; sample.len()];
    let mut next = 0u32;
    for o in obs.iter_mut() {
        if remap[o.p as usize] == u32::MAX {
            remap[o.p as usize] = next;
            next += 1;
        }
        o.p = remap[o.p as usize];
    }
    obs.sort_by_key(|o| o.p);
    (obs, next as usize)
}

pub struct Model {
    pub cam: [f64; CAM],
    pub frames: Vec<[f64; FRAME]>,
}

pub struct Report {
    pub before: f64,
    pub after: f64,
    pub points: usize,
    pub obs: usize,
}

/// `ev[f]`: zero-mean EXIF exposure in stops, None when the tags are missing.
pub fn fit(engine: &mut Engine, obs: &[Obs], n_pts: usize, nf: usize, ev: &[Option<f64>], vignette: bool) -> (Model, Report) {
    let mut starts = vec![0usize; n_pts + 1];
    for o in obs {
        starts[o.p as usize + 1] += 1;
    }
    for i in 0..n_pts {
        starts[i + 1] += starts[i];
    }
    let mut of_frame: Vec<Vec<usize>> = vec![Vec::new(); nf];
    for (i, o) in obs.iter().enumerate() {
        of_frame[o.f as usize].push(i);
    }
    engine.prepare(obs, &of_frame, n_pts);

    let mut m = Model { cam: isp::camera_init(), frames: vec![[0.0; FRAME]; nf] };
    let before = error(obs, &starts, &m);
    for (f, e) in ev.iter().enumerate() {
        m.frames[f][0] = e.unwrap_or(0.0);
    }
    let cam_free: Vec<usize> = if vignette { (0..CAM).collect() } else { (15..CAM).collect() };
    let p = Problem { obs, of_frame: &of_frame, ev, cam_free: &cam_free };

    for sweep in 0..SWEEPS {
        let l = radiance(obs, &starts, &m);
        engine.set_radiance(&l);
        lm(engine, Mode::Frames, &p, &l, &mut m);
        lm(engine, Mode::Camera, &p, &l, &mut m);
        gauge(&mut m);
        eprintln!("[vppisp] sweep {}  median error {:.2}/255", sweep + 1, 255.0 * error(obs, &starts, &m));
    }
    let after = error(obs, &starts, &m);
    (m, Report { before, after, points: n_pts, obs: obs.len() })
}

struct Problem<'a> {
    obs: &'a [Obs],
    of_frame: &'a [Vec<usize>],
    ev: &'a [Option<f64>],
    cam_free: &'a [usize],
}

/// Levenberg-Marquardt for every slot of `mode` at once: one engine call
/// builds all the systems, one more prices all the proposals, and each slot
/// keeps its own damping and accept/reject. Data costs are means over a
/// slot's residuals, so prior weights mean what they mean in PPISP.
fn lm(engine: &mut Engine, mode: Mode, p: &Problem, l: &[[f64; 3]], m: &mut Model) {
    let nf = m.frames.len();
    let slots = if mode == Mode::Camera { 1 } else { nf };
    let free: Vec<usize> = if mode == Mode::Camera { p.cam_free.to_vec() } else { (0..FRAME).collect() };
    let n_res = |s: usize| 3.0 * if mode == Mode::Camera { p.obs.len() } else { p.of_frame[s].len() } as f64;
    let params = |m: &Model, s: usize| -> Vec<f64> { if mode == Mode::Camera { m.cam.to_vec() } else { m.frames[s].to_vec() } };
    let prior = |s: usize, x: &[f64]| -> Vec<f64> { if mode == Mode::Camera { camera_prior(x) } else { frame_prior(x, p.ev[s]) } };
    let systems = |engine: &mut Engine, m: &Model, cost_only: bool| {
        engine.systems(mode, p.obs, p.of_frame, l, &m.cam, &m.frames, &free, cost_only)
    };

    let mut active: Vec<bool> = (0..slots).map(|s| mode == Mode::Camera || p.of_frame[s].len() >= 50).collect();
    let mut lambda = vec![1e-3; slots];
    let mut sys = systems(engine, m, false);
    let mut c0 = vec![0.0; slots];
    for s in 0..slots {
        let x = params(m, s);
        c0[s] = sys[s].cost / n_res(s).max(1.0) + sq(&prior(s, &x));
        finish(&mut sys[s], n_res(s), &x, &free, |x| prior(s, x));
    }
    for _ in 0..LM_ITERS {
        let mut trial = Model { cam: m.cam, frames: m.frames.clone() };
        let mut proposed = vec![false; slots];
        let live: Vec<usize> = (0..slots).filter(|&s| active[s]).collect();
        for s in live {
            let k = sys[s].k;
            let mut a = sys[s].jtj.clone();
            for d in 0..k {
                a[d * k + d] += lambda[s] * a[d * k + d].max(1e-12) + 1e-12;
            }
            let Some(step) = solve(a, sys[s].jtr.iter().map(|v| -v).collect(), k) else {
                active[s] = false;
                continue;
            };
            let x = if mode == Mode::Camera { &mut trial.cam[..] } else { &mut trial.frames[s][..] };
            for (a, &fi) in free.iter().enumerate() {
                x[fi] += step[a];
            }
            proposed[s] = true;
        }
        if !proposed.iter().any(|&v| v) {
            break;
        }
        let costs = systems(engine, &trial, true);
        let mut accepted = false;
        for s in (0..slots).filter(|&s| proposed[s]) {
            let x = params(&trial, s);
            let c1 = costs[s].cost / n_res(s).max(1.0) + sq(&prior(s, &x));
            if c1 < c0[s] {
                if mode == Mode::Camera {
                    m.cam = trial.cam;
                } else {
                    m.frames[s] = trial.frames[s];
                }
                if c0[s] - c1 < 1e-6 * c0[s] {
                    active[s] = false;
                }
                c0[s] = c1;
                lambda[s] = (lambda[s] / 3.0).max(1e-9);
                accepted = true;
            } else {
                lambda[s] *= 4.0;
                if lambda[s] > 1e6 {
                    active[s] = false;
                }
            }
        }
        if !active.iter().any(|&v| v) {
            break;
        }
        if accepted {
            sys = systems(engine, m, false);
            for s in 0..slots {
                let x = params(m, s);
                finish(&mut sys[s], n_res(s), &x, &free, |x| prior(s, x));
            }
        }
    }
}

fn sq(r: &[f64]) -> f64 {
    r.iter().map(|v| v * v).sum()
}

/// Scale a slot's data system to a mean and add its prior, numerically
/// differentiated (priors are a few dozen residuals at most).
fn finish(sys: &mut Sys, n_res: f64, x: &[f64], free: &[usize], prior: impl Fn(&[f64]) -> Vec<f64>) {
    let k = sys.k;
    let scale = 1.0 / n_res.max(1.0);
    sys.jtj.iter_mut().for_each(|v| *v *= scale);
    sys.jtr.iter_mut().for_each(|v| *v *= scale);
    let r0 = prior(x);
    let mut xp = x.to_vec();
    let jp: Vec<Vec<f64>> = free
        .iter()
        .map(|&fi| {
            let h = 1e-6 * xp[fi].abs().max(1.0);
            xp[fi] += h;
            let r1 = prior(&xp);
            xp[fi] -= h;
            r1.iter().zip(&r0).map(|(u, v)| (u - v) / h).collect()
        })
        .collect();
    for a in 0..k {
        sys.jtr[a] += jp[a].iter().zip(&r0).map(|(j, r)| j * r).sum::<f64>();
        for b in 0..k {
            sys.jtj[a * k + b] += jp[a].iter().zip(&jp[b]).map(|(u, v)| u * v).sum::<f64>();
        }
    }
}

/// Per point, the per-channel median of its observations pulled back through
/// each frame's inverse ISP.
fn radiance(obs: &[Obs], starts: &[usize], m: &Model) -> Vec<[f64; 3]> {
    let his: Vec<_> = m.frames.iter().map(|f| isp::invert(&isp::homography(&f[1..]))).collect();
    (0..starts.len() - 1)
        .into_par_iter()
        .map(|p| {
            let inv: Vec<[f64; 3]> = obs[starts[p]..starts[p + 1]]
                .iter()
                .map(|o| {
                    let px = [o.px[0] as f64, o.px[1] as f64, o.px[2] as f64];
                    isp::inverse(&m.cam, m.frames[o.f as usize][0], &his[o.f as usize], [o.uv[0] as f64, o.uv[1] as f64], px)
                })
                .collect();
            [0, 1, 2].map(|k| {
                let mut v: Vec<f64> = inv.iter().map(|x| x[k]).collect();
                v.sort_by(f64::total_cmp);
                v[v.len() / 2]
            })
        })
        .collect()
}

/// Median over observations of the mean absolute display-space error.
fn error(obs: &[Obs], starts: &[usize], m: &Model) -> f64 {
    let l = radiance(obs, starts, m);
    let hs: Vec<_> = m.frames.iter().map(|f| isp::homography(&f[1..])).collect();
    let mut e: Vec<f64> = obs
        .par_iter()
        .map(|o| {
            let r = engine::residual(&m.cam, m.frames[o.f as usize][0], &hs[o.f as usize], o, &l[o.p as usize]);
            r.iter().map(|v| v.abs()).sum::<f64>() / 3.0
        })
        .collect();
    e.sort_by(f64::total_cmp);
    e[e.len() / 2]
}

fn frame_prior(x: &[f64], ev: Option<f64>) -> Vec<f64> {
    let mut r: Vec<f64> = x[1..].iter().map(|z| W_LAT.sqrt() * z).collect();
    if let Some(e) = ev {
        r.push(W_EV.sqrt() * (x[0] - e));
    }
    r
}

/// PPISP's camera regularisers (Table 7, code weights): channel variance of
/// the vignetting and raw CRF parameters, vignetting centre, and a penalty on
/// brightening vignetting. Written as residuals whose squares are the terms.
fn camera_prior(p: &[f64]) -> Vec<f64> {
    let mut r = Vec::with_capacity(40);
    let var = |r: &mut Vec<f64>, base: usize, stride: usize, n: usize, w: f64| {
        for j in 0..n {
            let mean = (0..3).map(|k| p[base + stride * k + j]).sum::<f64>() / 3.0;
            for k in 0..3 {
                r.push((w / (3.0 * n as f64)).sqrt() * (p[base + stride * k + j] - mean));
            }
        }
    };
    var(&mut r, 0, 5, 5, 0.1);
    var(&mut r, 15, 4, 4, 0.1);
    for k in 0..3 {
        r.push((0.02f64 / 3.0).sqrt() * p[5 * k]);
        r.push((0.02f64 / 3.0).sqrt() * p[5 * k + 1]);
        for j in 2..5 {
            r.push((0.01 / 9.0 * p[5 * k + j].max(0.0)).sqrt());
        }
    }
    r
}

/// Hard gauge, in place of PPISP's soft mean priors: mean exposure and mean
/// chromaticity offset across frames are zero. The radiance absorbs both on
/// the next sweep.
fn gauge(m: &mut Model) {
    let nf = m.frames.len() as f64;
    let mean_dt = m.frames.iter().map(|f| f[0]).sum::<f64>() / nf;
    let offs: Vec<[[f64; 2]; 4]> = m.frames.iter().map(|f| isp::offsets(&f[1..])).collect();
    let mut mean = [[0.0; 2]; 4];
    for o in &offs {
        for k in 0..4 {
            mean[k][0] += o[k][0] / nf;
            mean[k][1] += o[k][1] / nf;
        }
    }
    for (f, o) in m.frames.iter_mut().zip(&offs) {
        f[0] -= mean_dt;
        let mut c = *o;
        for k in 0..4 {
            c[k][0] -= mean[k][0];
            c[k][1] -= mean[k][1];
        }
        f[1..].copy_from_slice(&isp::latents(&c));
    }
}

/// Gaussian elimination with partial pivoting.
fn solve(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| a[i * n + c].abs().total_cmp(&a[j * n + c].abs()))?;
        if a[p * n + c].abs() < 1e-300 {
            return None;
        }
        for j in 0..n {
            a.swap(c * n + j, p * n + j);
        }
        b.swap(c, p);
        for r in c + 1..n {
            let f = a[r * n + c] / a[c * n + c];
            for j in c..n {
                a[r * n + j] -= f * a[c * n + j];
            }
            b[r] -= f * b[c];
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        x[r] = (b[r] - (r + 1..n).map(|j| a[r * n + j] * x[j]).sum::<f64>()) / a[r * n + r];
    }
    Some(x)
}
