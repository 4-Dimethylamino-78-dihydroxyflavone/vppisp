//! vppisp — make an image set photometrically consistent before dvlt sees it.
//!
//! Program AE and auto white balance re-expose every frame: on the building
//! capture the same wall varies by ~3.5 EV between views. vppisp fits NVIDIA's
//! PPISP camera model (exposure + colour per frame, vignetting + response
//! curve per camera) to multi-view correspondences from a first dvlt pass,
//! then re-renders every image at the mean exposure, neutral colour, no
//! vignetting, through the fitted camera curve.
//!
//!   dvlt (pass 1) -> vppisp -> dvlt (pass 2) on <input>_ppisp
//!
//! Detached like vfold and vsky: delete the binary and nothing else changes.

mod engine;
mod fit;
mod gpu;
mod io;
mod isp;

use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

const USAGE: &str = "\
vppisp - PPISP photometric compensation, calibrated on a dvlt first pass

USAGE:
    vppisp <IMAGE_DIR | list.txt> --recon <DVLT_OUT> [OPTIONS]

<DVLT_OUT> is a dvlt run over the same images in the same order (poses.json
pairs by position). Run it on vsky-keyed images or after vsky --cull, so sky
does not enter the fit. Writes corrected images as <stem>.jpg, upright, plus
ppisp.json with every fitted parameter.

OPTIONS:
        --recon <DIR>     dvlt output folder (poses.json + scene.ply)    [required]
        --out <DIR>       where corrected images go   [default: <input>_ppisp]
        --size <PX>       the -s that reconstruction ran at           [default: 504]
        --points <N>      cloud points sampled for the fit         [default: 120000]
        --no-vignette     fit exposure, colour and CRF only (lenses
                          the camera already corrects in-JPEG)
        --quality <Q>     JPEG quality of the output                    [default: 95]
        --device <D>      auto | gpu | cpu: auto takes the GPU (Vulkan, else
                          DX12/Metal) and falls back to CPU         [default: auto]
";

struct Args {
    input: PathBuf,
    recon: PathBuf,
    out: Option<PathBuf>,
    size: u32,
    points: usize,
    vignette: bool,
    quality: u8,
    device: String,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("vppisp: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let a = parse_args()?;
    let t = Instant::now();
    let mut engine = engine::Engine::new(&a.device)?;
    eprintln!("[vppisp] device {}", engine.name());
    let images = io::list_images(&a.input)?;
    let cams = io::read_poses(&a.recon.join("poses.json"))?;
    if cams.len() != images.len() {
        return Err(format!(
            "{} images but poses.json has {} frames; pass the folder or list the reconstruction ran on",
            images.len(),
            cams.len()
        ));
    }
    let dec = images.par_iter().map(|p| io::decode_work(p)).collect::<Result<Vec<_>, _>>()?;
    let dims: Vec<(u32, u32)> = dec.iter().map(|d| (d.1, d.2)).collect();
    let ims: Vec<_> = dec.into_iter().map(|d| d.0).collect();
    let geo = io::geometry(&dims, a.size, &a.recon)?;
    let raw_ev: Vec<Option<f64>> = images.iter().map(|p| io::exif_ev(p)).collect();
    let known: Vec<f64> = raw_ev.iter().flatten().copied().collect();
    let mean_ev = known.iter().sum::<f64>() / known.len().max(1) as f64;
    let ev: Vec<Option<f64>> = raw_ev.iter().map(|e| e.map(|v| v - mean_ev)).collect();
    if !known.is_empty() {
        let (lo, hi) = known.iter().fold((f64::MAX, f64::MIN), |(l, h), &v| (l.min(v), h.max(v)));
        eprintln!("[vppisp] {} images, EXIF exposure on {} spans {:.2} EV", images.len(), known.len(), hi - lo);
    }

    let (xyz, rgb) = io::read_ply(&a.recon.join("scene.ply"))?;
    let (obs, n_pts) = fit::correspond(&mut engine, &xyz, &rgb, &cams, &geo, &ims, &dims, a.points);
    drop((xyz, rgb, ims));
    if n_pts < 1000 {
        return Err(format!("only {n_pts} points are seen by 3+ frames; is --recon the right reconstruction?"));
    }
    eprintln!("[vppisp] {} observations of {} points ({:.1}s)", obs.len(), n_pts, t.elapsed().as_secs_f32());

    let (model, rep) = fit::fit(&mut engine, &obs, n_pts, images.len(), &ev, a.vignette);
    eprintln!(
        "[vppisp] cross-view error {:.2}/255 -> {:.2}/255 ({:.1}s)",
        255.0 * rep.before,
        255.0 * rep.after,
        t.elapsed().as_secs_f32()
    );

    let out = a.out.clone().unwrap_or_else(|| io::sibling(&a.input, "_ppisp"));
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    // CRF^-1 per 8-bit code, then the forward CRF on a fine grid: per pixel
    // the correction is two lookups, a 3x3 homography and a gain.
    let mut luts = Vec::with_capacity(768 + 3 * 4097);
    for k in 0..3 {
        luts.extend((0..256).map(|v| isp::crf_inv(&model.cam, k, v as f64 / 255.0) as f32));
    }
    for k in 0..3 {
        luts.extend((0..4097).map(|i| isp::crf(&model.cam, k, i as f64 / 4096.0) as f32));
    }
    // Decode and encode are CPU work; at most 4 frames of 24 MP in flight.
    let engine = Mutex::new(engine);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().map_err(|e| e.to_string())?;
    pool.install(|| {
        images.par_iter().enumerate().try_for_each(|(f, p)| -> Result<(), String> {
            let img = image::DynamicImage::ImageRgb8(io::decode_full(p)?).into_rgba8();
            let (w, h) = img.dimensions();
            let px = engine.lock().map_err(|e| e.to_string())?.correct(img.as_raw(), w, h, &model.frames[f], &model.cam, a.vignette, &luts);
            drop(img);
            let mut name = p.file_stem().unwrap_or_default().to_os_string();
            name.push(".jpg");
            let path = out.join(name);
            jpeg_encoder::Encoder::new_file(&path, a.quality)
                .and_then(|e| e.encode(&px, w as u16, h as u16, jpeg_encoder::ColorType::Rgba))
                .map_err(|e| format!("{}: {e}", path.display()))
        })
    })?;

    // Input order, so a list input keeps it: reconstruct from <out>/images.txt.
    let list: String = images.iter().map(|p| format!("{}.jpg
", p.file_stem().unwrap_or_default().to_string_lossy())).collect();
    std::fs::write(out.join("images.txt"), list).map_err(|e| e.to_string())?;
    std::fs::write(out.join("ppisp.json"), report(&images, &raw_ev, &model, &rep, a.vignette))
        .map_err(|e| e.to_string())?;
    eprintln!("[vppisp] wrote {} images to {} ({:.1}s)", images.len(), out.display(), t.elapsed().as_secs_f32());
    Ok(())
}

fn report(images: &[PathBuf], ev: &[Option<f64>], m: &fit::Model, r: &fit::Report, vignette: bool) -> String {
    let nums = |v: &[f64]| v.iter().map(|x| format!("{x:.6}")).collect::<Vec<_>>().join(", ");
    let frames: Vec<String> = images
        .iter()
        .zip(ev)
        .zip(&m.frames)
        .map(|((p, e), f)| {
            let name = p.file_name().unwrap_or_default().to_string_lossy().replace('"', "");
            let e = e.map_or("null".to_string(), |v| format!("{v:.4}"));
            format!("    {{\"file\": \"{name}\", \"exif_ev\": {e}, \"exposure\": {:.4}, \"color_latents\": [{}]}}", f[0], nums(&f[1..]))
        })
        .collect();
    format!(
        "{{\n  \"model\": \"PPISP (Deutsch et al. 2026), fitted on point correspondences\",\n  \"vignette\": {vignette},\n  \
         \"points\": {}, \"observations\": {},\n  \"error_before\": {:.6}, \"error_after\": {:.6},\n  \
         \"camera\": {{\"vignetting\": [{}], \"crf_raw\": [{}]}},\n  \"frames\": [\n{}\n  ]\n}}\n",
        r.points,
        r.obs,
        r.before,
        r.after,
        nums(&m.cam[..15]),
        nums(&m.cam[15..]),
        frames.join(",\n")
    )
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        input: PathBuf::new(),
        recon: PathBuf::new(),
        out: None,
        size: 504,
        points: 120_000,
        vignette: true,
        quality: 95,
        device: "auto".into(),
    };
    let (mut input, mut recon) = (None, None);
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = || it.next().ok_or(format!("{arg} needs a value"));
        match arg.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            "--recon" => recon = Some(PathBuf::from(val()?)),
            "--out" => a.out = Some(val()?.into()),
            "--size" => a.size = num(&arg, val()?)?,
            "--points" => a.points = num(&arg, val()?)?,
            "--quality" => a.quality = num(&arg, val()?)?,
            "--no-vignette" => a.vignette = false,
            "--device" => a.device = val()?,
            s if s.starts_with('-') => return Err(format!("unknown option {s}\n\n{USAGE}")),
            _ if input.is_none() => input = Some(PathBuf::from(&arg)),
            _ => return Err(format!("unexpected argument {arg}")),
        }
    }
    a.input = input.ok_or_else(|| USAGE.to_string())?;
    a.recon = recon.ok_or("--recon <DVLT_OUT> is required")?;
    Ok(a)
}

fn num<T: std::str::FromStr>(name: &str, v: String) -> Result<T, String> {
    v.parse().map_err(|_| format!("{name}: not a number: {v}"))
}
