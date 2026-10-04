//! Reading what dvlt reads and wrote: the image list in its order, frames as
//! its loader sees them, poses.json, scene.ply. Mirrors vsky; kept separate so
//! either tool can be deleted alone.

use image::imageops::{self, FilterType};
use image::metadata::Orientation;
use image::{DynamicImage, ImageDecoder, ImageReader, RgbImage};
use std::fs;
use std::path::{Path, PathBuf};

/// dvlt's ViT patch edge: each frame is center-cropped to a multiple of it.
const PATCH: u32 = 14;
/// Short edge the fit decodes to (JPEG DCT scaling: 6048x4032 -> 756x504).
pub const WORK: u32 = 480;

pub fn is_list(p: &Path) -> bool {
    p.is_file() && p.extension().is_some_and(|e| e.eq_ignore_ascii_case("txt"))
}

/// The frames dvlt.exe would read, in its order (dvlt.cu expand_dir/expand_list).
pub fn list_images(input: &Path) -> Result<Vec<PathBuf>, String> {
    let err = |e: std::io::Error| format!("{}: {e}", input.display());
    let out: Vec<PathBuf> = if is_list(input) {
        let text = fs::read_to_string(input).map_err(err)?;
        let base = input.parent().unwrap_or(Path::new(""));
        let mut out = Vec::new();
        for (n, line) in text.trim_start_matches('\u{feff}').lines().enumerate() {
            let s = line.trim();
            if s.is_empty() || s.starts_with('#') {
                continue;
            }
            let p = base.join(s);
            if !p.is_file() {
                return Err(format!("{} line {}: file not found: {}", input.display(), n + 1, p.display()));
            }
            out.push(p);
        }
        out
    } else {
        let mut out: Vec<PathBuf> = fs::read_dir(input)
            .map_err(err)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.is_file()
                    && p.extension()
                        .and_then(|e| e.to_str())
                        .is_some_and(|e| ["jpg", "jpeg", "png"].contains(&e.to_ascii_lowercase().as_str()))
            })
            .collect();
        out.sort_by(|x, y| x.file_name().cmp(&y.file_name()));
        out
    };
    if out.is_empty() {
        return Err(format!("{}: no .jpg/.jpeg/.png images", input.display()));
    }
    Ok(out)
}

/// `scene` -> `scene_ppisp`: beside the input, never inside an image folder.
pub fn sibling(input: &Path, suffix: &str) -> PathBuf {
    let abs = std::path::absolute(input).unwrap_or_else(|_| input.to_path_buf());
    let mut name = if is_list(&abs) { abs.file_stem() } else { abs.file_name() }.unwrap_or_default().to_os_string();
    name.push(suffix);
    abs.with_file_name(name)
}

fn is_jpeg(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref(),
        Some("jpg" | "jpeg")
    )
}

/// Upright full-resolution image, as dvlt loads it (EXIF orientation for JPEG only).
pub fn decode_full(path: &Path) -> Result<RgbImage, String> {
    let err = |e: &dyn std::fmt::Display| format!("{}: {e}", path.display());
    let mut dec = ImageReader::open(path).map_err(|e| err(&e))?.into_decoder().map_err(|e| err(&e))?;
    let orient = if is_jpeg(path) { dec.orientation().unwrap_or(Orientation::NoTransforms) } else { Orientation::NoTransforms };
    let mut img = DynamicImage::from_decoder(dec).map_err(|e| err(&e))?;
    img.apply_orientation(orient);
    Ok(img.into_rgb8())
}

/// Upright image at working resolution plus the upright size of the original.
pub fn decode_work(path: &Path) -> Result<(RgbImage, u32, u32), String> {
    let err = |e: &dyn std::fmt::Display| format!("{}: {e}", path.display());
    let mut dec = ImageReader::open(path).map_err(|e| err(&e))?.into_decoder().map_err(|e| err(&e))?;
    let orient = if is_jpeg(path) { dec.orientation().unwrap_or(Orientation::NoTransforms) } else { Orientation::NoTransforms };
    let (w0, h0) = dec.dimensions();
    let rgb = match if is_jpeg(path) { decode_jpeg_scaled(path, WORK).ok() } else { None } {
        Some(rgb) => rgb,
        None => {
            let full = DynamicImage::from_decoder(dec).map_err(|e| err(&e))?.into_rgb8();
            let s = WORK as f32 / w0.min(h0) as f32;
            if s < 1.0 {
                imageops::resize(&full, (w0 as f32 * s).round() as u32, (h0 as f32 * s).round() as u32, FilterType::Triangle)
            } else {
                full
            }
        }
    };
    let mut img = DynamicImage::ImageRgb8(rgb);
    img.apply_orientation(orient);
    let swap = matches!(
        orient,
        Orientation::Rotate90 | Orientation::Rotate270 | Orientation::Rotate90FlipH | Orientation::Rotate270FlipH
    );
    let (sw, sh) = if swap { (h0, w0) } else { (w0, h0) };
    Ok((img.into_rgb8(), sw, sh))
}

fn decode_jpeg_scaled(path: &Path, target: u32) -> Result<RgbImage, String> {
    use jpeg_decoder::{Decoder, PixelFormat};
    let file = fs::File::open(path).map_err(|e| e.to_string())?;
    let mut dec = Decoder::new(std::io::BufReader::new(file));
    dec.read_info().map_err(|e| e.to_string())?;
    let info = dec.info().ok_or("no jpeg info")?;
    let short = info.width.min(info.height) as u32;
    let mut denom = 1u32;
    while denom < 8 && short / (denom * 2) >= target {
        denom *= 2;
    }
    let (w, h) = dec
        .scale((info.width as u32 / denom) as u16, (info.height as u32 / denom) as u16)
        .map_err(|e| e.to_string())?;
    let px = dec.decode().map_err(|e| e.to_string())?;
    let px = match dec.info().ok_or("no jpeg info")?.pixel_format {
        PixelFormat::RGB24 => px,
        PixelFormat::L8 => px.iter().flat_map(|&g| [g, g, g]).collect(),
        other => return Err(format!("unsupported jpeg pixel format {other:?}")),
    };
    RgbImage::from_raw(w as u32, h as u32, px).ok_or_else(|| "bad jpeg buffer".into())
}

/// log2(t * ISO / N^2): the exposure the camera actually set, in stops.
pub fn exif_ev(path: &Path) -> Option<f64> {
    let file = fs::File::open(path).ok()?;
    let meta = exif::Reader::new().read_from_container(&mut std::io::BufReader::new(file)).ok()?;
    let rational = |tag| match &meta.get_field(tag, exif::In::PRIMARY)?.value {
        exif::Value::Rational(v) if !v.is_empty() => Some(v[0].to_f64()),
        _ => None,
    };
    let t = rational(exif::Tag::ExposureTime)?;
    let n = rational(exif::Tag::FNumber)?;
    let iso = meta.get_field(exif::Tag::PhotographicSensitivity, exif::In::PRIMARY)?.value.get_uint(0)? as f64;
    (t > 0.0 && n > 0.0 && iso > 0.0).then(|| (t * iso / (n * n)).log2())
}

/// Where a padded-batch pixel lands in the upright source, per dvlt.cu
/// load_frame + preprocess: resize longest edge to `size` (lroundf in f32),
/// center-crop to PATCH, center-pad to the batch max.
#[derive(Clone, Copy)]
pub struct Geo {
    pub off: [f32; 2],
    pub inv: [f32; 2],
}

pub fn geometry(dims: &[(u32, u32)], size: u32, dir: &Path) -> Result<Vec<Geo>, String> {
    let g: Vec<(u32, u32, u32, u32)> = dims
        .iter()
        .map(|&(w, h)| {
            let s = size as f32 / w.max(h) as f32;
            let (nw, nh) = ((w as f32 * s).round() as u32, (h as f32 * s).round() as u32);
            (nw, nh, nw / PATCH * PATCH, nh / PATCH * PATCH)
        })
        .collect();
    let bw = g.iter().map(|g| g.2).max().unwrap_or(0);
    let bh = g.iter().map(|g| g.3).max().unwrap_or(0);
    if let Some((rw, rh)) = report_resolution(dir) {
        if (rw, rh) != (bw, bh) {
            return Err(format!(
                "report.txt says dvlt ran at {rw}x{rh}, but --size {size} gives {bw}x{bh}; pass the -s it used"
            ));
        }
    }
    Ok(g.iter()
        .map(|&(nw, nh, cw, ch)| Geo {
            off: [((nw - cw) / 2) as f32 - ((bw - cw) / 2) as f32, ((nh - ch) / 2) as f32 - ((bh - ch) / 2) as f32],
            inv: [1.0 / nw as f32, 1.0 / nh as f32],
        })
        .collect())
}

fn report_resolution(dir: &Path) -> Option<(u32, u32)> {
    let s = fs::read_to_string(dir.join("report.txt")).ok()?;
    let line = s.lines().find(|l| l.trim_start().starts_with("resolution"))?;
    let (w, h) = line.split_whitespace().last()?.split_once('x')?;
    Some((w.parse().ok()?, h.parse().ok()?))
}

/// One camera of poses.json: world -> camera rotation, centre, fx fy cx cy.
#[derive(Clone, Copy)]
pub struct Cam {
    pub rt: [f32; 9],
    pub c: [f32; 3],
    pub k: [f32; 4],
}

impl Cam {
    /// Depth and normalised upright-source position of a world point.
    #[inline]
    pub fn project(&self, g: &Geo, p: [f32; 3]) -> Option<(f32, f32, f32)> {
        let d = [p[0] - self.c[0], p[1] - self.c[1], p[2] - self.c[2]];
        let z = self.rt[6] * d[0] + self.rt[7] * d[1] + self.rt[8] * d[2];
        if z <= 1e-6 {
            return None;
        }
        let x = self.rt[0] * d[0] + self.rt[1] * d[1] + self.rt[2] * d[2];
        let y = self.rt[3] * d[0] + self.rt[4] * d[1] + self.rt[5] * d[2];
        let xn = (self.k[0] * x / z + self.k[2] + g.off[0]) * g.inv[0];
        let yn = (self.k[1] * y / z + self.k[3] + g.off[1]) * g.inv[1];
        ((0.0..1.0).contains(&xn) && (0.0..1.0).contains(&yn)).then_some((z, xn, yn))
    }
}

pub fn read_poses(path: &Path) -> Result<Vec<Cam>, String> {
    let s = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let e = arrays(&s, "\"extrinsics_c2w\"")?;
    let k = arrays(&s, "\"intrinsics\"")?;
    if e.len() != k.len() || e.iter().any(|a| a.len() != 16) || k.iter().any(|a| a.len() != 9) {
        return Err(format!("{}: expected 4x4 extrinsics_c2w + 3x3 intrinsics per frame", path.display()));
    }
    Ok(e.iter()
        .zip(&k)
        .map(|(e, k)| Cam {
            rt: [e[0], e[4], e[8], e[1], e[5], e[9], e[2], e[6], e[10]],
            c: [e[3], e[7], e[11]],
            k: [k[0], k[4], k[2], k[5]],
        })
        .collect())
}

fn arrays(s: &str, key: &str) -> Result<Vec<Vec<f32>>, String> {
    s.match_indices(key)
        .map(|(i, _)| {
            let rest = &s[i + key.len()..];
            let (a, b) = (rest.find('[').ok_or("poses.json: no '['")?, rest.find(']').ok_or("poses.json: no ']'")?);
            rest[a + 1..b]
                .split(',')
                .map(|t| t.trim().parse::<f32>().map_err(|e| format!("poses.json: {e}")))
                .collect()
        })
        .collect()
}

/// xyz + rgb of every vertex in a binary little-endian PLY.
pub fn read_ply(path: &Path) -> Result<(Vec<[f32; 3]>, Vec<[u8; 3]>), String> {
    let b = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    const END: &[u8] = b"end_header\n";
    let body = b.windows(END.len()).position(|w| w == END).ok_or("scene.ply: no end_header")? + END.len();
    let header = std::str::from_utf8(&b[..body]).map_err(|_| "scene.ply: header is not text")?;
    let (mut count, mut stride, mut off) = (0usize, 0usize, [usize::MAX; 6]);
    for line in header.lines() {
        match line.split_whitespace().collect::<Vec<_>>().as_slice() {
            ["format", f, _] if *f != "binary_little_endian" => return Err(format!("scene.ply: format {f}")),
            ["element", "vertex", n] => count = n.parse().map_err(|e| format!("scene.ply: {e}"))?,
            ["element", other, _] => return Err(format!("scene.ply: element '{other}' unsupported")),
            ["property", ty, name] => {
                let size = match *ty {
                    "char" | "uchar" | "int8" | "uint8" => 1,
                    "short" | "ushort" | "int16" | "uint16" => 2,
                    "int" | "uint" | "float" | "int32" | "uint32" | "float32" => 4,
                    "double" | "float64" => 8,
                    _ => return Err(format!("scene.ply: property type '{ty}'")),
                };
                if let Some(i) = ["x", "y", "z", "red", "green", "blue"].iter().position(|a| a == name) {
                    let want = if i < 3 { matches!(*ty, "float" | "float32") } else { matches!(*ty, "uchar" | "uint8") };
                    if !want {
                        return Err(format!("scene.ply: {name} is {ty}"));
                    }
                    off[i] = stride;
                }
                stride += size;
            }
            _ => {}
        }
    }
    if off.contains(&usize::MAX) {
        return Err("scene.ply: needs x/y/z and red/green/blue".into());
    }
    let data = b.get(body..body + count * stride).ok_or("scene.ply: truncated")?;
    let f = |v: &[u8], o: usize| f32::from_le_bytes([v[o], v[o + 1], v[o + 2], v[o + 3]]);
    Ok(data
        .chunks(stride)
        .map(|v| ([f(v, off[0]), f(v, off[1]), f(v, off[2])], [v[off[3]], v[off[4]], v[off[5]]]))
        .unzip())
}
