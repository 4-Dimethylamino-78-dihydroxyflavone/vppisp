# vppisp

Make an image set photometrically consistent before dvlt sees it.

Program AE and auto white balance re-expose every frame. On the 60-frame building
capture, the same brick wall varies by ~3.5 EV between views (EXIF agrees:
5.4 EV of shutter/aperture spread), so the point cloud's colours disagree
wherever views overlap. vppisp fits NVIDIA's PPISP camera model to multi-view
correspondences from a first dvlt pass, then re-renders every frame at the
mean exposure, neutral colour, no vignetting, through the fitted response
curve.

```
dvlt (pass 1) -> vppisp <images> --recon <pass1> -> dvlt (pass 2) on <images>_ppisp
```

`reconstruct.bat --ppisp` does all three. Detached: delete the binary and
nothing else changes.

## The model (PPISP, Deutsch et al. 2026, NVIDIA, Apache-2.0)

`image = CRF( CC( vignette(uv) · 2^dt · L ) )`

| | parameters | where |
|---|---|---|
| exposure `dt` | 1 per frame | log2 gain, seeded from EXIF |
| colour `CC` | 8 per frame | RGI homography from 4 chromaticity control points (B, R, G, neutral), intensity-preserving |
| vignetting | 15 per camera | per channel: centre + r², r⁴, r⁶ |
| CRF | 12 per camera | per channel: toe/shoulder power curve + gamma, monotone by construction |

Reimplemented from the reference kernel (`ppisp_math.cuh`, `torch_reference.py`),
including the ZCA preconditioners and the camera regularisers (Table 7).

## How it is fitted without a radiance field

PPISP learns inside a 3DGS/NeRF trainer, where the field supplies radiance.
Here a sample of the dvlt cloud (120k points) stands in for it: each point gets
a free linear radiance, observed in every frame that sees it (z-buffer
visibility over the whole cloud, 3×3 neighbourhood colour, saturated pixels
dropped). The solver alternates:

- **radiance**: per point, the median of its observations pulled back through
  each frame's inverse ISP (every stage inverts in closed form);
- **frames**: 9 parameters each, Levenberg–Marquardt, batched over all frames;
- **camera**: 27 shared parameters, Levenberg–Marquardt over everything.

Then comes a hard gauge (mean exposure 0, mean chromaticity offset 0). A weak
pull towards EXIF exposure pins PPISP's exposure-vs-CRF power ambiguity
(supp. A.3).

## Device

`--device auto` uses the GPU through wgpu: Vulkan, else DX12/Metal, any
vendor, nothing installed. CPU is the fallback rung. The z-buffers, the
normal-equation accumulation (central-difference Jacobians, per-workgroup
reduction) and the per-pixel correction run in WGSL. The small solves stay on
the CPU.

Measured on the building set (60 × 24 MP, RTX 4060 Ti vs 12 loaded CPU threads):

| stage | GPU | CPU |
|---|---|---|
| PPISP fit | **0.8 s** | 29.2 s |
| correct + write 60 frames | **17.1 s** | 94.6 s |
| total | **24.5 s** | 127.6 s |

Both lanes converge to the same error, sweep for sweep. Their images differ by
a mean of 0.03–0.13/255.

## Result

Median cross-view colour error over 170k observations: **11.0 → 7.6 /255**.
The fitted exposures track EXIF (3.8 vs 3.4 EV between the extremes).
Vignetting stays at identity, because Nikon corrects it in-camera and PPISP's
priors decline to invent it. Lifting an underexposed 8-bit JPEG clips its
brightest whites. RAW input would avoid that.

## Outputs

`<out>/<stem>.jpg`: upright, q95, no EXIF; `<out>/images.txt`: input order;
`<out>/ppisp.json`: every fitted parameter, the EXIF exposure, the errors.
