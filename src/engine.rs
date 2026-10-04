//! Where the heavy loops run: the GPU when one answers, the CPU otherwise.
//! Both rungs compute the same three things -- z-buffers, normal equations,
//! per-pixel correction -- so the solver never knows which one it got.

use crate::fit::Obs;
use crate::gpu::Gpu;
use crate::io::{Cam, Geo};
use crate::isp;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};

/// Huber threshold on display-space residuals (0..1); ~5/255.
pub const HUBER: f64 = 0.02;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Camera,
    Frames,
}

/// Gauss-Newton system of one slot (the camera, or one frame): J^T W J, J^T W r
/// and the summed Huber cost, all unscaled.
pub struct Sys {
    pub k: usize,
    pub jtj: Vec<f64>,
    pub jtr: Vec<f64>,
    pub cost: f64,
}

impl Sys {
    pub fn zero(k: usize) -> Sys {
        Sys { k, jtj: vec![0.0; k * k], jtr: vec![0.0; k], cost: 0.0 }
    }
}

pub enum Engine {
    Gpu(Box<Gpu>),
    Cpu,
}

impl Engine {
    /// `auto` takes the GPU if one initialises and says why not otherwise;
    /// naming a rung pins it.
    pub fn new(pref: &str) -> Result<Engine, String> {
        match pref {
            "cpu" => Ok(Engine::Cpu),
            "gpu" => Gpu::new().map(|g| Engine::Gpu(Box::new(g))),
            "auto" => Ok(match Gpu::new() {
                Ok(g) => Engine::Gpu(Box::new(g)),
                Err(e) => {
                    eprintln!("[vppisp] no GPU ({e}); using CPU");
                    Engine::Cpu
                }
            }),
            other => Err(format!("unknown --device '{other}' (auto|gpu|cpu)")),
        }
    }

    pub fn name(&self) -> String {
        match self {
            Engine::Gpu(g) => g.name.clone(),
            Engine::Cpu => format!("CPU ({} threads)", rayon::current_num_threads()),
        }
    }

    pub fn zbuffers(&mut self, xyz: &[[f32; 3]], cams: &[Cam], geo: &[Geo], zw: usize, zh: &[usize]) -> Vec<Vec<f32>> {
        match self {
            Engine::Gpu(g) => g.zbuffers(xyz, cams, geo, zw, zh),
            Engine::Cpu => (0..cams.len())
                .into_par_iter()
                .map(|f| {
                    let zb: Vec<AtomicU32> = (0..zw * zh[f]).map(|_| AtomicU32::new(f32::INFINITY.to_bits())).collect();
                    xyz.par_iter().for_each(|&p| {
                        if let Some((z, xn, yn)) = cams[f].project(&geo[f], p) {
                            let cell = ((yn * zh[f] as f32) as usize).min(zh[f] - 1) * zw + ((xn * zw as f32) as usize).min(zw - 1);
                            zb[cell].fetch_min(z.to_bits(), Ordering::Relaxed);
                        }
                    });
                    zb.into_iter().map(|a| f32::from_bits(a.into_inner())).collect()
                })
                .collect(),
        }
    }

    pub fn prepare(&mut self, obs: &[Obs], of_frame: &[Vec<usize>], n_pts: usize) {
        if let Engine::Gpu(g) = self {
            g.prepare(obs, of_frame, n_pts);
        }
    }

    pub fn set_radiance(&mut self, l: &[[f64; 3]]) {
        if let Engine::Gpu(g) = self {
            g.set_radiance(l);
        }
    }

    /// One system per slot; `cost_only` skips the Jacobians.
    #[allow(clippy::too_many_arguments)]
    pub fn systems(
        &mut self,
        mode: Mode,
        obs: &[Obs],
        of_frame: &[Vec<usize>],
        l: &[[f64; 3]],
        cam: &[f64; 27],
        frames: &[[f64; 9]],
        free: &[usize],
        cost_only: bool,
    ) -> Vec<Sys> {
        if let Engine::Gpu(g) = self {
            return g.systems(mode, cam, frames, free, cost_only, HUBER);
        }
        let hs: Vec<_> = frames.iter().map(|f| isp::homography(&f[1..])).collect();
        match mode {
            Mode::Camera => vec![accumulate(cam, free, obs.len(), cost_only, |x, i| {
                let o = &obs[i];
                residual(x, frames[o.f as usize][0], &hs[o.f as usize], o, &l[o.p as usize])
            })],
            Mode::Frames => (0..frames.len())
                .into_par_iter()
                .map(|f| {
                    let idx = &of_frame[f];
                    accumulate(&frames[f], &[0, 1, 2, 3, 4, 5, 6, 7, 8], idx.len(), cost_only, |x, i| {
                        let o = &obs[idx[i]];
                        residual(cam, x[0], &isp::homography(&x[1..]), o, &l[o.p as usize])
                    })
                })
                .collect(),
        }
    }

    /// Frame `f`'s correction of a packed RGBA image (alpha passes through).
    #[allow(clippy::too_many_arguments)]
    pub fn correct(&mut self, rgba: &[u8], w: u32, h: u32, frame: &[f64; 9], cam: &[f64; 27], vignette: bool, luts: &[f32]) -> Vec<u8> {
        let hi = isp::invert(&isp::homography(&frame[1..]));
        let gain = (-frame[0]).exp2();
        if let Engine::Gpu(g) = self {
            return g.correct(rgba, w, h, &hi, gain, cam, vignette, luts);
        }
        let (w, h) = (w as usize, h as usize);
        let mx = w.max(h) as f64;
        let lut = |k: usize, x: f64| {
            let s = x.clamp(0.0, 1.0) * 4096.0;
            let i = (s as usize).min(4095);
            let b = 768 + k * 4097;
            luts[b + i] as f64 + (luts[b + i + 1] as f64 - luts[b + i] as f64) * (s - i as f64)
        };
        let mut out = vec![0u8; rgba.len()];
        out.par_chunks_mut(4 * w).zip(rgba.par_chunks(4 * w)).enumerate().for_each(|(y, (dst, src))| {
            let v = (y as f64 + 0.5 - h as f64 / 2.0) / mx;
            for x in 0..w {
                let s = &src[4 * x..4 * x + 4];
                let lin = [luts[s[0] as usize] as f64, luts[256 + s[1] as usize] as f64, luts[512 + s[2] as usize] as f64];
                let c = isp::apply_cc(&hi, lin);
                let uv = [(x as f64 + 0.5 - w as f64 / 2.0) / mx, v];
                for k in 0..3 {
                    let vg = if vignette { isp::vignette(cam, k, uv).max(0.2) } else { 1.0 };
                    dst[4 * x + k] = (255.0 * lut(k, c[k] * gain / vg).clamp(0.0, 1.0) + 0.5) as u8;
                }
                dst[4 * x + 3] = 255;
            }
        });
        out
    }
}

#[inline]
pub fn residual(cam: &[f64], dt: f64, h: &[[f64; 3]; 3], o: &Obs, l: &[f64; 3]) -> [f64; 3] {
    let y = isp::forward(cam, dt, h, [o.uv[0] as f64, o.uv[1] as f64], *l);
    [y[0] - o.px[0] as f64, y[1] - o.px[1] as f64, y[2] - o.px[2] as f64]
}

pub fn huber(r: f64) -> (f64, f64) {
    let a = r.abs();
    if a <= HUBER {
        (0.5 * r * r, 1.0)
    } else {
        (HUBER * (a - 0.5 * HUBER), HUBER / a)
    }
}

/// CPU normal equations with forward-difference Jacobians in f64.
fn accumulate<F>(x0: &[f64], free: &[usize], n: usize, cost_only: bool, r: F) -> Sys
where
    F: Fn(&[f64], usize) -> [f64; 3] + Sync,
{
    let k = if cost_only { 0 } else { free.len() };
    let mut s = (0..n)
        .into_par_iter()
        .fold(
            || (Sys::zero(k), x0.to_vec()),
            |(mut s, mut xp), i| {
                let r0 = r(&xp, i);
                let mut w = [0.0; 3];
                for c in 0..3 {
                    let (h, wc) = huber(r0[c]);
                    s.cost += h;
                    w[c] = wc;
                }
                if k > 0 {
                    let mut j = vec![[0.0; 3]; k];
                    for (a, &fi) in free.iter().enumerate() {
                        let h = 1e-6 * xp[fi].abs().max(1.0);
                        xp[fi] += h;
                        let r1 = r(&xp, i);
                        xp[fi] -= h;
                        j[a] = [0, 1, 2].map(|c| (r1[c] - r0[c]) / h);
                    }
                    for a in 0..k {
                        for c in 0..3 {
                            s.jtr[a] += w[c] * j[a][c] * r0[c];
                        }
                        for b in a..k {
                            s.jtj[a * k + b] += (0..3).map(|c| w[c] * j[a][c] * j[b][c]).sum::<f64>();
                        }
                    }
                }
                (s, xp)
            },
        )
        .map(|(s, _)| s)
        .reduce(
            || Sys::zero(k),
            |mut a, b| {
                a.cost += b.cost;
                a.jtj.iter_mut().zip(&b.jtj).for_each(|(x, y)| *x += y);
                a.jtr.iter_mut().zip(&b.jtr).for_each(|(x, y)| *x += y);
                a
            },
        );
    for a in 0..k {
        for b in 0..a {
            s.jtj[a * k + b] = s.jtj[b * k + a];
        }
    }
    s
}
