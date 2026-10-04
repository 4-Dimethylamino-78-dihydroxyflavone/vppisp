//! The compute backend. wgpu, so one WGSL source runs on Vulkan (DX12 or
//! Metal where Vulkan is absent) on any vendor's GPU, with nothing installed.

use crate::engine::{Mode, Sys};
use crate::fit::Obs;
use crate::io::{Cam, Geo};
use wgpu::util::DeviceExt;

const SHADER: &str = include_str!("shaders.wgsl");
const STRIDE: usize = 406;
const TRI: usize = 378;
/// Observations per workgroup in the normal-equation kernel.
const CHUNK: usize = 512;

pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pub name: String,
    zbuf: wgpu::ComputePipeline,
    normal: wgpu::ComputePipeline,
    correct: wgpu::ComputePipeline,
    fit: Option<FitBufs>,
    img: Option<ImgBufs>,
}

struct FitBufs {
    rad: wgpu::Buffer,
    frames: wgpu::Buffer,
    params: wgpu::Buffer,
    partials: wgpu::Buffer,
    readback: wgpu::Buffer,
    bind: wgpu::BindGroup,
    slot: Vec<u32>,
}

struct ImgBufs {
    len: u64,
    src: wgpu::Buffer,
    dst: wgpu::Buffer,
    luts: wgpu::Buffer,
    uni: wgpu::Buffer,
    readback: wgpu::Buffer,
    bind: wgpu::BindGroup,
}

fn f32s(v: &[f32]) -> &[u8] {
    bytemuck::cast_slice(v)
}

impl Gpu {
    pub fn new() -> Result<Gpu, String> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::VULKAN | wgpu::Backends::DX12 | wgpu::Backends::METAL;
        let instance = wgpu::Instance::new(desc);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|e| e.to_string())?;
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            return Err(format!("only a software adapter ({})", info.name));
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("vppisp"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| e.to_string())?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vppisp"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipe = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let (zbuf, normal, correct) = (pipe("zbuffer"), pipe("normal"), pipe("correct"));
        Ok(Gpu {
            name: format!("{:?}: {}", info.backend, info.name),
            device,
            queue,
            zbuf,
            normal,
            correct,
            fit: None,
            img: None,
        })
    }

    fn storage(&self, data: &[u8], extra: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: data,
            usage: wgpu::BufferUsages::STORAGE | extra,
        })
    }

    fn empty(&self, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage, mapped_at_creation: false })
    }

    fn bind(&self, pipe: &wgpu::ComputePipeline, entries: &[(u32, &wgpu::Buffer)]) -> wgpu::BindGroup {
        let e: Vec<wgpu::BindGroupEntry> =
            entries.iter().map(|&(binding, b)| wgpu::BindGroupEntry { binding, resource: b.as_entire_binding() }).collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipe.get_bind_group_layout(0), entries: &e })
    }

    fn run(&self, pipe: &wgpu::ComputePipeline, bind: &wgpu::BindGroup, groups: (u32, u32, u32), copy: Option<(&wgpu::Buffer, &wgpu::Buffer, u64)>) {
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, bind, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, groups.2);
        }
        if let Some((from, to, size)) = copy {
            enc.copy_buffer_to_buffer(from, 0, to, 0, size);
        }
        self.queue.submit([enc.finish()]);
    }

    fn read(&self, buf: &wgpu::Buffer, size: u64) -> Vec<u8> {
        buf.map_async(wgpu::MapMode::Read, 0..size, |_| {});
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let v = buf.get_mapped_range(0..size).map(|m| m.to_vec()).unwrap_or_default();
        buf.unmap();
        v
    }

    /// Per-frame z-buffers (`zw` x `zh[f]`, row-major), +inf where empty.
    pub fn zbuffers(&mut self, xyz: &[[f32; 3]], cams: &[Cam], geo: &[Geo], zw: usize, zh: &[usize]) -> Vec<Vec<f32>> {
        let mut base = vec![0usize; zh.len() + 1];
        for f in 0..zh.len() {
            base[f + 1] = base[f] + zw * zh[f];
        }
        let total = base[zh.len()];
        let mut cd = Vec::with_capacity(cams.len() * 24);
        for (f, (c, g)) in cams.iter().zip(geo).enumerate() {
            cd.extend_from_slice(&c.rt);
            cd.extend_from_slice(&c.c);
            cd.extend_from_slice(&c.k);
            cd.extend_from_slice(&g.off);
            cd.extend_from_slice(&g.inv);
            cd.extend_from_slice(&[f32::from_bits(zh[f] as u32), f32::from_bits(base[f] as u32), 0.0, 0.0]);
        }
        let blocks = xyz.len().div_ceil(256) as u32;
        let bx = blocks.min(65535);
        let bz = blocks.div_ceil(bx);
        let pts = self.storage(bytemuck::cast_slice(xyz), wgpu::BufferUsages::empty());
        let cb = self.storage(f32s(&cd), wgpu::BufferUsages::empty());
        let zb = self.storage(bytemuck::cast_slice(&vec![f32::INFINITY.to_bits(); total]), wgpu::BufferUsages::COPY_SRC);
        let uni = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[xyz.len() as u32, zw as u32, bx, 0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let size = (total * 4) as u64;
        let rb = self.empty(size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        let bind = self.bind(&self.zbuf, &[(0, &pts), (1, &cb), (2, &zb), (3, &uni)]);
        self.run(&self.zbuf, &bind, (bx, cams.len() as u32, bz), Some((&zb, &rb, size)));
        let bits: Vec<u32> = bytemuck::pod_collect_to_vec(&self.read(&rb, size));
        (0..zh.len()).map(|f| bits[base[f]..base[f + 1]].iter().map(|&b| f32::from_bits(b)).collect()).collect()
    }

    /// Upload the observations, grouped by frame and cut into chunks.
    pub fn prepare(&mut self, obs: &[Obs], of_frame: &[Vec<usize>], n_pts: usize) {
        let mut order = Vec::with_capacity(obs.len());
        let (mut chunks, mut slot) = (Vec::new(), Vec::new());
        for (f, idx) in of_frame.iter().enumerate() {
            let start = order.len();
            order.extend(idx.iter().map(|&i| i as u32));
            for s in (0..idx.len()).step_by(CHUNK) {
                chunks.extend_from_slice(&[(start + s) as u32, (start + (s + CHUNK).min(idx.len())) as u32, f as u32, 0]);
                slot.push(f as u32);
            }
        }
        let mut od = Vec::with_capacity(obs.len() * 8);
        for o in obs {
            od.extend_from_slice(&[f32::from_bits(o.p), f32::from_bits(o.f as u32), o.uv[0], o.uv[1], o.px[0], o.px[1], o.px[2], 0.0]);
        }
        let ob = self.storage(f32s(&od), wgpu::BufferUsages::empty());
        let orb = self.storage(bytemuck::cast_slice(&order), wgpu::BufferUsages::empty());
        let chb = self.storage(bytemuck::cast_slice(&chunks), wgpu::BufferUsages::empty());
        let rad = self.empty((n_pts.max(1) * 16) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        let frames = self.empty((of_frame.len() * 80) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        let params = self.empty(240, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let size = (slot.len() * STRIDE * 4) as u64;
        let partials = self.empty(size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let readback = self.empty(size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        let bind = self.bind(
            &self.normal,
            &[(10, &ob), (11, &orb), (12, &chb), (13, &rad), (14, &frames), (15, &params), (16, &partials)],
        );
        self.fit = Some(FitBufs { rad, frames, params, partials, readback, bind, slot });
    }

    pub fn set_radiance(&mut self, l: &[[f64; 3]]) {
        let fb = self.fit.as_ref().expect("prepare first");
        let d: Vec<f32> = l.iter().flat_map(|v| [v[0] as f32, v[1] as f32, v[2] as f32, 0.0]).collect();
        self.queue.write_buffer(&fb.rad, 0, f32s(&d));
    }

    /// Normal equations (or costs only) per slot: the camera, or each frame.
    pub fn systems(&mut self, mode: Mode, cam: &[f64; 27], frames: &[[f64; 9]], free: &[usize], cost_only: bool, huber: f64) -> Vec<Sys> {
        let fb = self.fit.as_ref().expect("prepare first");
        let mut fd = Vec::with_capacity(frames.len() * 20);
        for f in frames {
            let h = crate::isp::homography(&f[1..]);
            fd.extend(f.iter().map(|&v| v as f32));
            fd.extend(h.iter().flatten().map(|&v| v as f32));
            fd.extend_from_slice(&[0.0, 0.0]);
        }
        self.queue.write_buffer(&fb.frames, 0, f32s(&fd));
        let mut p = [0u32; 60];
        for (i, &v) in cam.iter().enumerate() {
            p[i] = (v as f32).to_bits();
        }
        for (i, &v) in free.iter().enumerate() {
            p[28 + i] = v as u32;
        }
        p[56] = (mode == Mode::Frames) as u32 | (cost_only as u32) << 1;
        p[57] = free.len() as u32;
        p[58] = (huber as f32).to_bits();
        self.queue.write_buffer(&fb.params, 0, bytemuck::cast_slice(&p));
        let size = (fb.slot.len() * STRIDE * 4) as u64;
        self.run(&self.normal, &fb.bind, (fb.slot.len() as u32, 1, 1), Some((&fb.partials, &fb.readback, size)));
        let part: Vec<f32> = bytemuck::pod_collect_to_vec(&self.read(&fb.readback, size));

        let k = if mode == Mode::Frames { 9 } else { free.len() };
        let slots = if mode == Mode::Frames { frames.len() } else { 1 };
        let mut out: Vec<Sys> = (0..slots).map(|_| Sys::zero(if cost_only { 0 } else { k })).collect();
        for (c, &s) in fb.slot.iter().enumerate() {
            let s = if mode == Mode::Frames { s as usize } else { 0 };
            let row = &part[c * STRIDE..(c + 1) * STRIDE];
            let sys = &mut out[s];
            sys.cost += row[405] as f64;
            if cost_only {
                continue;
            }
            for a in 0..k {
                sys.jtr[a] += row[TRI + a] as f64;
                for b in a..k {
                    sys.jtj[a * k + b] += row[a * (2 * k - a + 1) / 2 + (b - a)] as f64;
                }
            }
        }
        for sys in out.iter_mut() {
            for a in 0..sys.k {
                for b in 0..a {
                    sys.jtj[a * sys.k + b] = sys.jtj[b * sys.k + a];
                }
            }
        }
        out
    }

    /// One frame's correction over a packed RGBA image.
    pub fn correct(&mut self, rgba: &[u8], w: u32, h: u32, hi: &[[f64; 3]; 3], gain: f64, cam: &[f64; 27], vignette: bool, luts: &[f32]) -> Vec<u8> {
        let len = rgba.len() as u64;
        if self.img.as_ref().is_none_or(|b| b.len != len) {
            let src = self.empty(len, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
            let dst = self.empty(len, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
            let lb = self.empty((luts.len() * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
            let uni = self.empty(176, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
            let readback = self.empty(len, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
            let bind = self.bind(&self.correct, &[(20, &src), (21, &dst), (22, &lb), (23, &uni)]);
            self.img = Some(ImgBufs { len, src, dst, luts: lb, uni, readback, bind });
        }
        let b = self.img.as_ref().expect("just made");
        let mut u = [0u32; 44];
        for r in 0..3 {
            for c in 0..3 {
                u[4 * r + c] = (hi[r][c] as f32).to_bits();
            }
        }
        for (i, &v) in cam.iter().enumerate() {
            u[12 + i] = (v as f32).to_bits();
        }
        u[40] = (gain as f32).to_bits();
        u[41] = w;
        u[42] = h;
        u[43] = vignette as u32;
        self.queue.write_buffer(&b.uni, 0, bytemuck::cast_slice(&u));
        self.queue.write_buffer(&b.luts, 0, f32s(luts));
        self.queue.write_buffer(&b.src, 0, rgba);
        self.run(&self.correct, &b.bind, (w.div_ceil(16), h.div_ceil(16), 1), Some((&b.dst, &b.readback, len)));
        self.read(&b.readback, len)
    }
}
