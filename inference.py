#!/usr/bin/env python3
"""STAR Video Super-Resolution — Standalone Inference Script

Memory-efficient video upscaling from the command line.  Works outside
ComfyUI — just activate the same Python environment.

Examples
--------
  # Video → video (audio is preserved automatically)
  python inference.py input.mp4 -o output.mp4

  # Lower VRAM (model offload + smaller segments)
  python inference.py input.mp4 -o output.mp4 --offload model --segment-size 8

  # Image sequence → image sequence
  python inference.py frames_in/ -o frames_out/

  # Image sequence → video
  python inference.py frames_in/ -o output.mp4 --fps 24

  # Single image
  python inference.py photo.png -o photo_4x.png
"""

# ── Comfy module stubs (must run before star_pipeline import) ────────────
import os
import sys
import types
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
STAR_REPO = SCRIPT_DIR / "STAR"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(STAR_REPO))

# Apply patches from patches/ directory to the STAR submodule.
import subprocess  # noqa: E402

_PATCHES_DIR = SCRIPT_DIR / "patches"
if _PATCHES_DIR.is_dir():
    for _patch in sorted(_PATCHES_DIR.iterdir()):
        if _patch.suffix != ".patch":
            continue
        if subprocess.call(
            ["git", "apply", "--check", "--reverse", str(_patch)],
            cwd=str(STAR_REPO), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ) != 0:
            if subprocess.call(["git", "apply", str(_patch)], cwd=str(STAR_REPO)) == 0:
                print(f"[STAR] Applied patch: {_patch.name}")
            else:
                print(f"[STAR] Warning: failed to apply patch: {_patch.name}")

import torch  # noqa: E402 — needed for stub defaults

_comfy = types.ModuleType("comfy")
_comfy_utils = types.ModuleType("comfy.utils")
_comfy_mm = types.ModuleType("comfy.model_management")


class _ProgressBar:
    """tqdm-based stand-in for comfy.utils.ProgressBar."""

    def __init__(self, total):
        from tqdm import tqdm

        self._bar = tqdm(total=total, desc="Denoising", unit="step")

    def update(self, n=1):
        self._bar.update(n)

    def __del__(self):
        if hasattr(self, "_bar"):
            self._bar.close()


_comfy_utils.ProgressBar = _ProgressBar
_comfy_mm.get_torch_device = lambda: torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
_comfy_mm.soft_empty_cache = lambda: torch.cuda.empty_cache()

_comfy.utils = _comfy_utils
_comfy.model_management = _comfy_mm

sys.modules["comfy"] = _comfy
sys.modules["comfy.utils"] = _comfy_utils
sys.modules["comfy.model_management"] = _comfy_mm

# ── Attention backend dispatcher ──────────────────────────────────────
import torch.nn.functional as F  # noqa: E402

_ATTN_BACKENDS = {"sdpa": None}

_real_xformers_mea = None
try:
    import xformers.ops
    _candidate = xformers.ops.memory_efficient_attention
    if not getattr(_candidate, "_is_star_dispatcher", False):
        _real_xformers_mea = _candidate
        _ATTN_BACKENDS["xformers"] = _real_xformers_mea
except ImportError:
    pass

_SAGE_VARIANTS = [
    "sageattn",
    "sageattn_qk_int8_pv_fp16_triton",
    "sageattn_qk_int8_pv_fp16_cuda",
    "sageattn_qk_int8_pv_fp8_cuda",
]
for _name in _SAGE_VARIANTS:
    try:
        _fn = getattr(__import__("sageattention", fromlist=[_name]), _name)
        _ATTN_BACKENDS[_name] = _fn
    except (ImportError, AttributeError):
        pass

# Manual attention (guaranteed correct, used as diagnostic baseline)
_ATTN_BACKENDS["math"] = "math"

_active_attn = "sdpa"


def _set_attn(backend: str):
    global _active_attn
    if backend not in _ATTN_BACKENDS:
        print(f"[STAR] Warning: backend '{backend}' not available, falling back to sdpa")
        backend = "sdpa"
    _active_attn = backend
    print(f"[STAR] Attention backend: {backend}")


def _dispatched_mea(q, k, v, attn_bias=None, op=None):
    if _active_attn == "xformers":
        return _real_xformers_mea(q, k, v, attn_bias=attn_bias, op=op)
    if _active_attn == "math":
        # Naive batched attention — slow but guaranteed correct.
        scale = q.shape[-1] ** -0.5
        cs = max(1, 2**28 // (q.shape[1] * q.shape[1] * max(q.element_size(), 1)))
        outs = []
        for i in range(0, q.shape[0], cs):
            qi, ki, vi = q[i:i+cs], k[i:i+cs], v[i:i+cs]
            a = torch.bmm(qi * scale, ki.transpose(1, 2))
            if attn_bias is not None:
                a = a + (attn_bias[i:i+cs] if attn_bias.shape[0] > 1 else attn_bias)
            outs.append(torch.bmm(a.softmax(dim=-1), vi))
        return torch.cat(outs)
    if _active_attn == "sdpa":
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
    # SageAttention variants: need 4D tensors (batch, heads, seq, dim)
    fn = _ATTN_BACKENDS[_active_attn]
    return fn(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
        tensor_layout="HND", is_causal=False,
    ).squeeze(0)


_dispatched_mea._is_star_dispatcher = True

if "xformers" in sys.modules:
    sys.modules["xformers"].ops.memory_efficient_attention = _dispatched_mea
else:
    _xformers = types.ModuleType("xformers")
    _xformers_ops = types.ModuleType("xformers.ops")
    _xformers_ops.memory_efficient_attention = _dispatched_mea
    _xformers.ops = _xformers_ops
    sys.modules["xformers"] = _xformers
    sys.modules["xformers.ops"] = _xformers_ops

print(f"[STAR] Available attention backends: {list(_ATTN_BACKENDS.keys())}")

# ── Standard imports ────────────────────────────────────────────────────
import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

# ── Constants ───────────────────────────────────────────────────────────
HF_REPO = "SherryX/STAR"
HF_MODELS = {
    "light_deg.pt": "I2VGen-XL-based/light_deg.pt",
    "heavy_deg.pt": "I2VGen-XL-based/heavy_deg.pt",
}
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".ts"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


# ── Argument parsing ───────────────────────────────────────────────────
def parse_args():
    class Fmt(argparse.ArgumentDefaultsHelpFormatter,
              argparse.RawDescriptionHelpFormatter):
        pass

    p = argparse.ArgumentParser(
        description="STAR Video Super-Resolution — standalone inference",
        formatter_class=Fmt,
        epilog=__doc__,
    )

    # -- I/O --
    p.add_argument("input", help="Input video file, image file, or directory of frames")
    p.add_argument("-o", "--output",
                   help="Output path (video file, image file, or directory). "
                        "Auto-generated with _star suffix if omitted.")

    # -- Model --
    g = p.add_argument_group("model")
    g.add_argument("--model", default="light_deg.pt",
                   help="Model name (light_deg.pt / heavy_deg.pt) or path to .pt file")
    g.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"],
                   help="Weight precision")
    g.add_argument("--offload", default="model",
                   choices=["disabled", "model", "aggressive"],
                   help="VRAM offloading strategy")

    # -- Processing --
    g = p.add_argument_group("processing")
    g.add_argument("--upscale", type=int, default=4, help="Upscale factor")
    g.add_argument("--segment-size", type=int, default=16,
                   help="Frames per segment (bounds peak RAM). 0 = all at once")
    g.add_argument("--steps", type=int, default=15, help="Denoising steps")
    g.add_argument("--guide-scale", type=float, default=7.5, help="Guidance scale")
    g.add_argument("--solver-mode", default="fast", choices=["fast", "normal"])
    g.add_argument("--max-chunk-len", type=int, default=32,
                   help="Temporal chunk length inside diffusion loop")
    g.add_argument("--seed", type=int, default=0, help="Random seed")
    g.add_argument("--color-fix", default="adain",
                   choices=["adain", "wavelet", "none"],
                   help="Post-processing color correction")
    g.add_argument("--prompt", default="",
                   help="Text prompt (empty = STAR built-in quality prompt)")
    g.add_argument("--attention", default="sdpa",
                   choices=list(_ATTN_BACKENDS.keys()),
                   help="Attention backend")

    # -- Video output --
    g = p.add_argument_group("video output")
    g.add_argument("--fps", type=float, default=None,
                   help="Output FPS (default: match input, or 24 for image sequences)")
    g.add_argument("--codec", default="libx264", help="FFmpeg video codec")
    g.add_argument("--crf", type=int, default=18,
                   help="FFmpeg CRF quality (lower = better)")
    g.add_argument("--pix-fmt", default="yuv420p", help="FFmpeg pixel format")
    g.add_argument("--no-audio", action="store_true",
                   help="Do not copy audio from input video")

    return p.parse_args()


# ── Model resolution ───────────────────────────────────────────────────
def resolve_model_path(model_arg: str) -> str:
    if os.path.isfile(model_arg):
        return model_arg

    search = [
        SCRIPT_DIR / "models" / model_arg,
        # Standard ComfyUI layout: custom_nodes/Comfyui-STAR/../../models/star/
        SCRIPT_DIR / ".." / ".." / "models" / "star" / model_arg,
    ]
    for candidate in search:
        candidate = candidate.resolve()
        if candidate.is_file():
            return str(candidate)

    if model_arg not in HF_MODELS:
        raise FileNotFoundError(
            f"Model '{model_arg}' not found locally and is not a known "
            "downloadable model. Provide a full path, or use "
            "light_deg.pt / heavy_deg.pt."
        )

    from huggingface_hub import hf_hub_download

    print(f"[STAR] Downloading {model_arg} from HuggingFace ({HF_REPO})...")
    dest_dir = str(SCRIPT_DIR / "models")
    os.makedirs(dest_dir, exist_ok=True)
    path = hf_hub_download(
        repo_id=HF_REPO, filename=HF_MODELS[model_arg], local_dir=dest_dir,
    )
    return path


# ── Model loading (mirrors STARModelLoader.load_model) ─────────────────
def load_model(model_path: str, precision: str, offload: str, device: torch.device):
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[precision]
    keep_on = device if offload == "disabled" else "cpu"

    from video_to_video.modules.embedder import FrozenOpenCLIPEmbedder
    from video_to_video.utils.config import cfg

    print("[STAR] Loading text encoder (OpenCLIP ViT-H-14)...")
    text_encoder = FrozenOpenCLIPEmbedder(device=device, pretrained="laion2b_s32b_b79k")
    text_encoder.model.to(device)
    negative_y = text_encoder(cfg.negative_prompt).detach()
    text_encoder.model.to(keep_on)

    from video_to_video.modules.unet_v2v import ControlledV2VUNet

    print("[STAR] Loading UNet + ControlNet...")
    generator = ControlledV2VUNet()
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    generator.load_state_dict(ckpt, strict=False)
    del ckpt
    generator = generator.to(device=keep_on, dtype=dtype)
    generator.eval()

    from video_to_video.diffusion.schedules_sdedit import noise_schedule
    from video_to_video.diffusion.diffusion_sdedit import GaussianDiffusion

    sigmas = noise_schedule(
        schedule="logsnr_cosine_interp", n=1000,
        zero_terminal_snr=True, scale_min=2.0, scale_max=4.0,
    )
    diffusion = GaussianDiffusion(sigmas=sigmas)

    from diffusers import AutoencoderKLTemporalDecoder

    print("[STAR] Loading temporal VAE...")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid",
        subfolder="vae", variant="fp16",
    )
    vae.eval()
    vae.requires_grad_(False)
    vae.to(keep_on)

    torch.cuda.empty_cache()
    print("[STAR] All models loaded.")

    return {
        "text_encoder": text_encoder,
        "generator": generator,
        "diffusion": diffusion,
        "vae": vae,
        "negative_y": negative_y,
        "device": device,
        "dtype": dtype,
        "offload": offload,
    }


# ── Input reading ──────────────────────────────────────────────────────
def _ffprobe(path: str):
    """Return (width, height, fps, nb_frames) via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", str(path),
    ]
    info = json.loads(subprocess.check_output(cmd))
    vs = next(s for s in info["streams"] if s["codec_type"] == "video")
    w, h = int(vs["width"]), int(vs["height"])

    num, den = map(int, vs.get("r_frame_rate", "24/1").split("/"))
    fps = num / den if den else 24.0

    nb = vs.get("nb_frames")
    if nb and nb != "N/A":
        n_frames = int(nb)
    else:
        dur = float(info.get("format", {}).get("duration", 0))
        n_frames = int(dur * fps) or 0

    return w, h, fps, n_frames


def read_video(path: str):
    """Read video → (np array [N,H,W,3] uint8, fps)."""
    w, h, fps, est = _ffprobe(path)
    print(f"[STAR] Input video: {w}x{h}, {fps:.2f} fps, ~{est} frames")

    cmd = [
        "ffmpeg", "-i", str(path),
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-v", "quiet", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    frames = []
    fsize = w * h * 3
    while True:
        raw = proc.stdout.read(fsize)
        if len(raw) < fsize:
            break
        frames.append(np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3))

    proc.wait()
    print(f"[STAR] Read {len(frames)} frames")
    return np.stack(frames), fps


def read_image_dir(directory: str):
    """Read image directory → (np array [N,H,W,3] uint8, None)."""
    d = Path(directory)
    files = sorted(f for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise FileNotFoundError(f"No image files in {d}")

    print(f"[STAR] Loading {len(files)} images from {d}")
    frames = [np.array(Image.open(f).convert("RGB")) for f in files]
    return np.stack(frames), None


def read_input(path: str):
    """Auto-detect input type → (np array [N,H,W,3] uint8, fps | None)."""
    p = Path(path)
    if p.is_dir():
        return read_image_dir(path)
    if p.suffix.lower() in VIDEO_EXTS:
        return read_video(path)
    if p.suffix.lower() in IMAGE_EXTS:
        img = np.array(Image.open(p).convert("RGB"))
        return img[np.newaxis], None
    raise ValueError(f"Unsupported input: {path}")


# ── Output writing ─────────────────────────────────────────────────────
class VideoWriter:
    """Stream RGB frames to ffmpeg, optionally copying audio from source."""

    def __init__(self, output_path, fps, width, height,
                 codec="libx264", crf=18, pix_fmt="yuv420p",
                 audio_source=None):
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-",
        ]
        if audio_source:
            cmd += ["-i", str(audio_source)]

        cmd += ["-map", "0:v:0"]
        if audio_source:
            cmd += ["-map", "1:a?", "-c:a", "copy"]

        cmd += [
            "-c:v", codec, "-crf", str(crf), "-pix_fmt", pix_fmt,
            "-movflags", "+faststart",
            "-v", "warning",
            str(output_path),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.count = 0

    def write_frame(self, frame_uint8):
        self.proc.stdin.write(frame_uint8.tobytes())
        self.count += 1

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


class ImageSequenceWriter:
    """Save frames as numbered image files."""

    def __init__(self, out_dir, ext=".png"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ext = ext
        self.count = 0

    def write_frame(self, frame_uint8):
        Image.fromarray(frame_uint8).save(
            self.out_dir / f"{self.count:06d}{self.ext}"
        )
        self.count += 1

    def close(self):
        pass


class SingleImageWriter:
    """Save a single output image."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.count = 0

    def write_frame(self, frame_uint8):
        Image.fromarray(frame_uint8).save(self.path)
        self.count += 1

    def close(self):
        pass


# ── Output path helpers ────────────────────────────────────────────────
def is_video_path(p):
    return Path(p).suffix.lower() in VIDEO_EXTS


def is_image_path(p):
    return Path(p).suffix.lower() in IMAGE_EXTS


def auto_output(input_path: str) -> str:
    p = Path(input_path)
    if p.is_dir():
        return str(p.parent / (p.name + "_star"))
    return str(p.parent / (p.stem + "_star" + p.suffix))


def make_writer(output_path, fps, w, h, args, input_path, is_single_image):
    """Create the appropriate writer for the output path."""
    if is_single_image and is_image_path(output_path):
        return SingleImageWriter(output_path)
    if is_video_path(output_path):
        audio_src = input_path if (
            is_video_path(input_path) and not args.no_audio
        ) else None
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        return VideoWriter(
            output_path, fps, w, h,
            codec=args.codec, crf=args.crf, pix_fmt=args.pix_fmt,
            audio_source=audio_src,
        )
    # Default: image sequence directory
    return ImageSequenceWriter(output_path, ext=".png")


# ── Tensor ↔ numpy ─────────────────────────────────────────────────────
def to_tensor(frames_uint8):
    """[N,H,W,3] uint8 numpy → [N,H,W,3] float32 torch in [0,1]."""
    return torch.from_numpy(frames_uint8.copy()).float() / 255.0


def to_uint8(tensor):
    """[N,H,W,3] float32 torch in [0,1] → [N,H,W,3] uint8 numpy."""
    return (tensor.clamp(0, 1) * 255).byte().cpu().numpy()


# ── Segment-based processing with streaming output ─────────────────────
def _run_segment(star_model, frames_uint8, args):
    """Process one segment through STAR → float32 tensor [F,H,W,3]."""
    from star_pipeline import run_star_inference

    tensor = to_tensor(frames_uint8)
    return run_star_inference(
        star_model=star_model,
        images=tensor,
        upscale=args.upscale,
        steps=args.steps,
        guide_scale=args.guide_scale,
        prompt=args.prompt,
        solver_mode=args.solver_mode,
        max_chunk_len=args.max_chunk_len,
        seed=args.seed,
        color_fix=args.color_fix,
    )


def _write_tensor(writer, tensor):
    """Write a float32 [F,H,W,3] tensor as uint8 frames."""
    arr = to_uint8(tensor)
    for i in range(arr.shape[0]):
        writer.write_frame(arr[i])


def process_and_stream(star_model, input_frames, writer, args):
    """Process in segments, blend overlaps, and stream to writer."""
    total = input_frames.shape[0]
    seg = args.segment_size

    # No segmentation — process everything at once
    if seg <= 0 or total <= seg:
        print(f"[STAR] Processing all {total} frame(s)...")
        result = _run_segment(star_model, input_frames, args)
        _write_tensor(writer, result)
        return

    overlap = max(2, seg // 4)
    stride = seg - overlap

    # Build segment boundaries
    segments = []
    start = 0
    while start < total:
        end = min(start + seg, total)
        segments.append((start, end))
        if end == total:
            break
        start += stride

    print(f"[STAR] {total} frames → {len(segments)} segment(s), "
          f"segment_size={seg}, overlap={overlap}")

    prev_tail = None  # float32 tensor on CPU

    for idx, (s, e) in enumerate(segments):
        print(f"\n[STAR] ── Segment {idx + 1}/{len(segments)}: "
              f"frames {s}–{e - 1} ──")

        seg_result = _run_segment(star_model, input_frames[s:e], args)

        # Blend overlap with previous segment's tail
        if prev_tail is not None:
            n = prev_tail.shape[0]
            head = seg_result[:n]
            w = torch.linspace(0, 1, n, dtype=seg_result.dtype).view(n, 1, 1, 1)
            blended = prev_tail * (1.0 - w) + head * w
            _write_tensor(writer, blended)
            remainder = seg_result[n:]
        else:
            remainder = seg_result

        if idx < len(segments) - 1:
            # Keep tail for blending, write the rest
            prev_tail = remainder[-overlap:].clone()
            _write_tensor(writer, remainder[:-overlap])
        else:
            # Last segment — write everything
            _write_tensor(writer, remainder)

        del seg_result
        torch.cuda.empty_cache()


# ── Main ────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Validate environment
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. STAR requires a CUDA GPU.")
        sys.exit(1)
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        input_p = Path(args.input)
        if input_p.suffix.lower() in VIDEO_EXTS or (
            args.output and Path(args.output).suffix.lower() in VIDEO_EXTS
        ):
            print("Error: ffmpeg/ffprobe not found. Install ffmpeg for video I/O.")
            sys.exit(1)

    # Read input
    input_frames, input_fps = read_input(args.input)
    total = input_frames.shape[0]
    h_in, w_in = input_frames.shape[1], input_frames.shape[2]
    h_out, w_out = h_in * args.upscale, w_in * args.upscale
    is_single = total == 1 and Path(args.input).is_file() and is_image_path(args.input)

    print(f"[STAR] {w_in}x{h_in} → {w_out}x{h_out} ({args.upscale}x), "
          f"{total} frame(s)")

    # Output path
    output_path = args.output or auto_output(args.input)
    fps = args.fps or input_fps or 24.0

    # Load model
    device = torch.device("cuda")
    model_path = resolve_model_path(args.model)
    print(f"[STAR] Model: {model_path}")
    star_model = load_model(model_path, args.precision, args.offload, device)

    _set_attn(args.attention)

    # Create writer and process
    writer = make_writer(output_path, fps, w_out, h_out, args, args.input, is_single)
    process_and_stream(star_model, input_frames, writer, args)
    writer.close()

    print(f"\n[STAR] Done! {writer.count} frame(s) → {output_path}")


if __name__ == "__main__":
    main()
