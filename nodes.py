import os
import sys
import torch
import folder_paths
import comfy.model_management as mm

# Register the "star" model folder so users can drop .pt weights there.
star_model_dir = os.path.join(folder_paths.models_dir, "star")
os.makedirs(star_model_dir, exist_ok=True)
folder_paths.folder_names_and_paths["star"] = (
    [star_model_dir],
    folder_paths.supported_pt_extensions,
)

# Put the cloned STAR repo on sys.path so its internal imports work.
STAR_REPO = os.path.join(os.path.dirname(os.path.realpath(__file__)), "STAR")

# Auto-initialize the git submodule if it's empty (e.g. cloned without --recursive).
if not os.path.isdir(os.path.join(STAR_REPO, "video_to_video")):
    import subprocess
    print("[STAR] Submodule not found — running git submodule update --init ...")
    subprocess.check_call(
        ["git", "submodule", "update", "--init", "--recursive"],
        cwd=os.path.dirname(os.path.realpath(__file__)),
    )

if STAR_REPO not in sys.path:
    sys.path.insert(0, STAR_REPO)

# Provide an xformers compatibility shim if xformers is not installed.
# The STAR UNet only uses xformers.ops.memory_efficient_attention.
# Priority: SageAttention (fastest, INT8 quantized) > PyTorch native SDPA (always available).
if "xformers" not in sys.modules:
    try:
        import xformers  # noqa: F401
    except ImportError:
        import types

        _xformers = types.ModuleType("xformers")
        _xformers_ops = types.ModuleType("xformers.ops")

        try:
            from sageattention import sageattn as _sageattn

            def _memory_efficient_attention(q, k, v, attn_bias=None, op=None):
                # STAR UNet passes 3D (B*heads, seq, dim); SageAttention needs 4D.
                return _sageattn(
                    q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                    tensor_layout="HND", is_causal=False,
                ).squeeze(0)

            print("[STAR] xformers not found — using SageAttention (fast INT8 quantized).")
        except ImportError:
            def _memory_efficient_attention(q, k, v, attn_bias=None, op=None):
                return torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_bias,
                )

            print("[STAR] xformers not found — using PyTorch native SDPA as fallback.")

        _xformers_ops.memory_efficient_attention = _memory_efficient_attention
        _xformers.ops = _xformers_ops
        sys.modules["xformers"] = _xformers
        sys.modules["xformers.ops"] = _xformers_ops

# Known models on HuggingFace that can be auto-downloaded.
HF_REPO = "SherryX/STAR"
HF_MODELS = {
    "light_deg.pt": "I2VGen-XL-based/light_deg.pt",
    "heavy_deg.pt": "I2VGen-XL-based/heavy_deg.pt",
}


def _get_model_list():
    """Return the union of files already on disk + known downloadable models."""
    on_disk = set(folder_paths.get_filename_list("star"))
    available = set(HF_MODELS.keys())
    return sorted(on_disk | available)


def _ensure_model(model_name: str) -> str:
    """Return the local path to model_name, downloading from HF if needed."""
    local = folder_paths.get_full_path("star", model_name)
    if local is not None:
        return local

    if model_name not in HF_MODELS:
        raise FileNotFoundError(
            f"Model '{model_name}' not found in {star_model_dir} and is not a known downloadable model."
        )

    from huggingface_hub import hf_hub_download

    print(f"[STAR] Downloading {model_name} from HuggingFace ({HF_REPO})...")
    path = hf_hub_download(
        repo_id=HF_REPO,
        filename=HF_MODELS[model_name],
        local_dir=star_model_dir,
    )
    # hf_hub_download may place the file in a subdirectory; symlink into the
    # star folder root so folder_paths can find it next time.
    dest = os.path.join(star_model_dir, model_name)
    if not os.path.exists(dest):
        os.symlink(path, dest)
    return dest


class STARModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_get_model_list(), {
                    "tooltip": "STAR checkpoint to load. light_deg for mildly degraded video, heavy_deg for heavily degraded video. Auto-downloaded from HuggingFace on first use.",
                }),
                "precision": (["fp16", "bf16", "fp32"], {
                    "default": "fp16",
                    "tooltip": "Weight precision. fp16 is recommended (fastest, lowest VRAM). bf16 for newer GPUs. fp32 for maximum quality at 2x VRAM cost.",
                }),
                "offload": (["disabled", "model", "aggressive"], {
                    "default": "disabled",
                    "tooltip": "disabled: all on GPU (~39GB). model: swap UNet/VAE/CLIP to CPU when idle (~16GB). aggressive: model offload + single-frame VAE decode (~12GB).",
                }),
            }
        }

    RETURN_TYPES = ("STAR_MODEL",)
    RETURN_NAMES = ("star_model",)
    FUNCTION = "load_model"
    CATEGORY = "STAR"
    DESCRIPTION = "Loads the STAR video super-resolution model (UNet+ControlNet, OpenCLIP text encoder, temporal VAE). All components are auto-downloaded on first use."

    def load_model(self, model_name, precision, offload="disabled"):
        device = mm.get_torch_device()
        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        dtype = dtype_map[precision]

        # Where to park models when not in use.
        keep_on = device if offload == "disabled" else "cpu"

        model_path = _ensure_model(model_name)

        # ---- Text encoder (OpenCLIP ViT-H-14) ----
        from video_to_video.modules.embedder import FrozenOpenCLIPEmbedder

        text_encoder = FrozenOpenCLIPEmbedder(
            device=device, pretrained="laion2b_s32b_b79k"
        )
        text_encoder.model.to(device)

        # Pre-compute the negative prompt embedding used during sampling.
        from video_to_video.utils.config import cfg

        negative_y = text_encoder(cfg.negative_prompt).detach()

        # Park text encoder after pre-computing embeddings.
        text_encoder.model.to(keep_on)

        # ---- UNet + ControlNet ----
        from video_to_video.modules.unet_v2v import ControlledV2VUNet

        generator = ControlledV2VUNet()
        load_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        if "state_dict" in load_dict:
            load_dict = load_dict["state_dict"]
        generator.load_state_dict(load_dict, strict=False)
        del load_dict
        generator = generator.to(device=keep_on, dtype=dtype)
        generator.eval()

        # ---- Noise schedule + diffusion helper ----
        from video_to_video.diffusion.schedules_sdedit import noise_schedule
        from video_to_video.diffusion.diffusion_sdedit import GaussianDiffusion

        sigmas = noise_schedule(
            schedule="logsnr_cosine_interp",
            n=1000,
            zero_terminal_snr=True,
            scale_min=2.0,
            scale_max=4.0,
        )
        diffusion = GaussianDiffusion(sigmas=sigmas)

        # ---- Temporal VAE (from HuggingFace diffusers) ----
        from diffusers import AutoencoderKLTemporalDecoder

        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid",
            subfolder="vae",
            variant="fp16",
        )
        vae.eval()
        vae.requires_grad_(False)
        vae.to(keep_on)

        torch.cuda.empty_cache()

        star_model = {
            "text_encoder": text_encoder,
            "generator": generator,
            "diffusion": diffusion,
            "vae": vae,
            "negative_y": negative_y,
            "device": device,
            "dtype": dtype,
            "offload": offload,
        }
        return (star_model,)


class STARVideoSuperResolution:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "star_model": ("STAR_MODEL", {
                    "tooltip": "Connect from STAR Model Loader.",
                }),
                "images": ("IMAGE", {
                    "tooltip": "Input video frames (IMAGE batch). Can come from LoadImage, VHS LoadVideo, etc.",
                }),
                "upscale": ("INT", {
                    "default": 4, "min": 1, "max": 8,
                    "tooltip": "Upscale factor applied to the input resolution. 4x is the default. Higher values need more VRAM.",
                }),
                "steps": ("INT", {
                    "default": 15, "min": 1, "max": 100,
                    "tooltip": "Number of denoising steps. Ignored in 'fast' solver mode (hardcoded 15). More steps = better quality but slower.",
                }),
                "guide_scale": ("FLOAT", {
                    "default": 7.5, "min": 1.0, "max": 20.0, "step": 0.5,
                    "tooltip": "Classifier-free guidance scale. Higher values follow the prompt more strongly. 7.5 is a good default.",
                }),
                "prompt": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Text prompt describing the desired output. Leave empty to use STAR's built-in quality prompt.",
                }),
                "solver_mode": (["fast", "normal"], {
                    "default": "fast",
                    "tooltip": "fast: optimized 15-step schedule (4 coarse + 11 fine). normal: uniform schedule using the steps parameter.",
                }),
                "max_chunk_len": ("INT", {
                    "default": 32, "min": 4, "max": 128,
                    "tooltip": "Max frames processed at once. Lower values reduce VRAM usage for long videos. Chunks overlap by 50%.",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Random seed for reproducible results.",
                }),
                "color_fix": (["adain", "wavelet", "none"], {
                    "default": "adain",
                    "tooltip": "Post-processing color correction. adain: match color stats from input. wavelet: preserve input low-frequency color. none: no correction.",
                }),
                "segment_size": ("INT", {
                    "default": 0, "min": 0, "max": 256,
                    "tooltip": "Process video in segments of this many frames to reduce RAM usage. 0 = process all at once. Recommended: 16-32 for long videos.",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "upscale_video"
    CATEGORY = "STAR"
    DESCRIPTION = "Upscale video frames using STAR diffusion-based super-resolution."

    def upscale_video(
        self,
        star_model,
        images,
        upscale,
        steps,
        guide_scale,
        prompt,
        solver_mode,
        max_chunk_len,
        seed,
        color_fix,
        segment_size=0,
    ):
        kwargs = dict(
            star_model=star_model,
            images=images,
            upscale=upscale,
            steps=steps,
            guide_scale=guide_scale,
            prompt=prompt,
            solver_mode=solver_mode,
            max_chunk_len=max_chunk_len,
            seed=seed,
            color_fix=color_fix,
        )

        if segment_size > 0:
            from .star_pipeline import run_star_inference_segmented
            result = run_star_inference_segmented(segment_size=segment_size, **kwargs)
        else:
            from .star_pipeline import run_star_inference
            result = run_star_inference(**kwargs)

        return (result,)


NODE_CLASS_MAPPINGS = {
    "STARModelLoader": STARModelLoader,
    "STARVideoSuperResolution": STARVideoSuperResolution,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "STARModelLoader": "STAR Model Loader",
    "STARVideoSuperResolution": "STAR Video Super-Resolution",
}
