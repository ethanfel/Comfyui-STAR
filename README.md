# ComfyUI-STAR

ComfyUI custom nodes for [STAR (Spatial-Temporal Augmentation with Text-to-Video Models for Real-World Video Super-Resolution)](https://github.com/NJU-PCALab/STAR) — a diffusion-based video upscaling model (ICCV 2025).

## Features

- **Diffusion-based 4x video super-resolution** with temporal coherence
- **Two model variants**: `light_deg.pt` (light degradation) and `heavy_deg.pt` (heavy degradation)
- **Auto-download**: all models (UNet checkpoint, OpenCLIP text encoder, temporal VAE) download automatically on first use
- **VRAM offloading**: three modes to fit GPUs from 12GB to 40GB+
- **Long video support**: sliding-window chunking with 50% overlap
- **Color correction**: AdaIN and wavelet-based post-processing

## Installation

### ComfyUI Manager

Search for `ComfyUI-STAR` in ComfyUI Manager and install.

### Manual

```bash
cd ComfyUI/custom_nodes
git clone --recursive https://github.com/ethanfel/Comfyui-STAR.git
cd Comfyui-STAR
pip install -r requirements.txt
```

> The `--recursive` flag clones the STAR submodule. If you forgot it, run `git submodule update --init` afterwards.

## Nodes

### STAR Model Loader

Loads the STAR model components (UNet+ControlNet, OpenCLIP text encoder, temporal VAE).

| Input | Description |
|-------|-------------|
| **model_name** | `light_deg.pt` for mildly degraded video, `heavy_deg.pt` for heavily degraded video. Auto-downloaded from HuggingFace on first use. |
| **precision** | `fp16` (recommended), `bf16`, or `fp32`. |
| **offload** | `disabled` (~39GB VRAM), `model` (~16GB — swaps components to CPU when idle), `aggressive` (~12GB — model offload + single-frame VAE decode). |

### STAR Video Super-Resolution

Runs the STAR diffusion pipeline on an image batch.

| Input | Description |
|-------|-------------|
| **star_model** | Connect from STAR Model Loader. |
| **images** | Input video frames (IMAGE batch). |
| **upscale** | Upscale factor (1–8, default 4). |
| **steps** | Denoising steps (1–100, default 15). Ignored in `fast` mode. |
| **guide_scale** | Classifier-free guidance scale (1–20, default 7.5). |
| **prompt** | Text prompt. Leave empty for STAR's built-in quality prompt. |
| **solver_mode** | `fast` (optimized 15-step schedule) or `normal` (uniform schedule). |
| **max_chunk_len** | Max frames per chunk (4–128, default 32). Lower = less VRAM for long videos. |
| **seed** | Random seed for reproducibility. |
| **color_fix** | `adain` (match color stats), `wavelet` (preserve low-frequency color), or `none`. |
| **segment_size** | Process video in segments of this many frames to reduce RAM usage (0–256, default 0). 0 = process all at once. Recommended: 16–32 for long videos. Segments overlap by 25% with linear crossfade blending. |

## VRAM Requirements

| Offload Mode | Approximate VRAM | Notes |
|---|---|---|
| disabled | ~39 GB | Fastest — everything on GPU |
| model | ~16 GB | Components swap to CPU between stages |
| aggressive | ~12 GB | Model offload + frame-by-frame VAE decode |

Reducing `max_chunk_len` further lowers VRAM usage for long videos at the cost of slightly more processing time.

## Model Weights

Models are stored in `ComfyUI/models/star/` and auto-downloaded on first use:

| Model | Use Case | Source |
|-------|----------|--------|
| `light_deg.pt` | Low-res video from the web, mild compression | [HuggingFace](https://huggingface.co/SherryX/STAR/resolve/main/I2VGen-XL-based/light_deg.pt) |
| `heavy_deg.pt` | Heavily compressed/degraded video | [HuggingFace](https://huggingface.co/SherryX/STAR/resolve/main/I2VGen-XL-based/heavy_deg.pt) |

The OpenCLIP text encoder and SVD temporal VAE are downloaded automatically by their respective libraries on first load.

## Credits

- [STAR](https://github.com/NJU-PCALab/STAR) by Rui Xie, Yinhong Liu et al. (Nanjing University) — ICCV 2025
- Based on [I2VGen-XL](https://github.com/ali-vilab/VGen) and [VEnhancer](https://github.com/Vchitect/VEnhancer)

## License

This wrapper is MIT licensed. The STAR model weights follow their respective licenses (MIT for I2VGen-XL-based models).
