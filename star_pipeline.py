import torch
import torch.nn.functional as F
import torch.amp
from einops import rearrange

import comfy.utils
import comfy.model_management as mm

from video_to_video.video_to_video_model import pad_to_fit, make_chunks
from video_to_video.utils.config import cfg


# ---------------------------------------------------------------------------
# Tensor format conversions
# ---------------------------------------------------------------------------

def comfyui_to_star_frames(images: torch.Tensor) -> torch.Tensor:
    """Convert ComfyUI IMAGE batch to STAR input format.

    ComfyUI: [N, H, W, 3] float32 in [0, 1]
    STAR:    [N, 3, H, W] float32 in [-1, 1]
    """
    t = images.permute(0, 3, 1, 2)  # [N,3,H,W]
    t = t * 2.0 - 1.0
    return t


def star_output_to_comfyui(video: torch.Tensor) -> torch.Tensor:
    """Convert STAR output to ComfyUI IMAGE batch.

    STAR output: [1, 3, F, H, W] float32 in [-1, 1]
    ComfyUI:     [F, H, W, 3] float32 in [0, 1]
    """
    v = video.squeeze(0)            # [3, F, H, W]
    v = v.permute(1, 2, 3, 0)      # [F, H, W, 3]
    v = (v + 1.0) / 2.0
    v = v.clamp(0.0, 1.0)
    return v


# ---------------------------------------------------------------------------
# VAE helpers (mirror VideoToVideo_sr methods)
# ---------------------------------------------------------------------------

def vae_encode(vae, t, chunk_size=1):
    """Encode [B, F, C, H, W] video tensor to latent space."""
    num_f = t.shape[1]
    t = rearrange(t, "b f c h w -> (b f) c h w")
    z_list = []
    for ind in range(0, t.shape[0], chunk_size):
        z_list.append(vae.encode(t[ind : ind + chunk_size]).latent_dist.sample())
    z = torch.cat(z_list, dim=0)
    z = rearrange(z, "(b f) c h w -> b c f h w", f=num_f)
    return z * vae.config.scaling_factor


def vae_decode_chunk(vae, z, chunk_size=3):
    """Decode latent [B, C, F, H, W] back to pixel frames."""
    z = rearrange(z, "b c f h w -> (b f) c h w")
    video = []
    for ind in range(0, z.shape[0], chunk_size):
        chunk = z[ind : ind + chunk_size]
        num_f = chunk.shape[0]
        decoded = vae.decode(chunk / vae.config.scaling_factor, num_frames=num_f).sample
        video.append(decoded)
    video = torch.cat(video)
    return video


# ---------------------------------------------------------------------------
# Color correction wrappers
# ---------------------------------------------------------------------------

def apply_color_fix(output_frames, input_frames_star, method):
    """Apply colour correction to the upscaled output.

    output_frames:      [F, H, W, 3] float [0, 1]  (ComfyUI format)
    input_frames_star:  [F, 3, H, W] float [-1, 1]  (STAR format)
    method: "adain" | "wavelet" | "none"
    """
    if method == "none":
        return output_frames

    from video_super_resolution.color_fix import adain_color_fix, wavelet_color_fix

    # Resize input to match output spatial size for stats transfer
    _, h_out, w_out, _ = output_frames.shape
    source = F.interpolate(
        input_frames_star, size=(h_out, w_out), mode="bilinear", align_corners=False
    )

    # The color_fix functions expect:
    #   target: [T, H, W, C] in [0, 255]
    #   source: [T, C, H, W] in [-1, 1]
    target = output_frames * 255.0

    if method == "adain":
        result = adain_color_fix(target, source)
    else:
        result = wavelet_color_fix(target, source)

    return (result / 255.0).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Progress-bar integration via trange monkey-patch
# ---------------------------------------------------------------------------

def _make_progress_trange(pbar, total_steps):
    """Return a drop-in replacement for tqdm.auto.trange that drives *pbar*."""
    from tqdm.auto import trange as _real_trange

    def _progress_trange(*args, **kwargs):
        kwargs["disable"] = True                     # silence console output
        for val in _real_trange(*args, **kwargs):
            yield val
            pbar.update(1)

    return _progress_trange


# ---------------------------------------------------------------------------
# Main inference entry point
# ---------------------------------------------------------------------------

def _move(module, device):
    """Move a nn.Module to device and free source memory."""
    module.to(device)
    if device == "cpu" or (isinstance(device, torch.device) and device.type == "cpu"):
        torch.cuda.empty_cache()


def run_star_inference_segmented(
    star_model: dict,
    images: torch.Tensor,
    segment_size: int,
    upscale: int = 4,
    steps: int = 15,
    guide_scale: float = 7.5,
    prompt: str = "",
    solver_mode: str = "fast",
    max_chunk_len: int = 32,
    seed: int = 0,
    color_fix: str = "adain",
) -> torch.Tensor:
    """Run STAR inference in overlapping segments to bound peak RAM usage.

    Each segment of `segment_size` frames is processed independently through
    the full pipeline.  Overlap regions (25% of segment_size, minimum 2 frames)
    are blended with a linear crossfade to avoid temporal seam artifacts.
    """
    total_frames = images.shape[0]

    # Fall back to single-shot if the video fits in one segment.
    if total_frames <= segment_size:
        return run_star_inference(
            star_model=star_model, images=images, upscale=upscale, steps=steps,
            guide_scale=guide_scale, prompt=prompt, solver_mode=solver_mode,
            max_chunk_len=max_chunk_len, seed=seed, color_fix=color_fix,
        )

    overlap = max(2, segment_size // 4)
    stride = segment_size - overlap

    # Build list of (start, end) frame indices for each segment.
    segments = []
    start = 0
    while start < total_frames:
        end = min(start + segment_size, total_frames)
        segments.append((start, end))
        if end == total_frames:
            break
        start += stride

    print(f"[STAR] Segmented processing: {total_frames} frames, "
          f"segment_size={segment_size}, overlap={overlap}, "
          f"{len(segments)} segment(s)")

    result_chunks: list[torch.Tensor] = []  # each on CPU, [F_i, H, W, 3]
    prev_tail: torch.Tensor | None = None   # overlap tail from previous segment

    for seg_idx, (seg_start, seg_end) in enumerate(segments):
        print(f"[STAR] Processing segment {seg_idx + 1}/{len(segments)}: "
              f"frames {seg_start}-{seg_end - 1}")

        seg_images = images[seg_start:seg_end]

        seg_result = run_star_inference(
            star_model=star_model,
            images=seg_images,
            upscale=upscale,
            steps=steps,
            guide_scale=guide_scale,
            prompt=prompt,
            solver_mode=solver_mode,
            max_chunk_len=max_chunk_len,
            seed=seed,
            color_fix=color_fix,
        )
        # seg_result: [F_seg, H, W, 3] float32 on CPU

        if prev_tail is not None:
            # Blend the overlap region between previous segment's tail and
            # this segment's head using a linear ramp.
            n_overlap = prev_tail.shape[0]
            head = seg_result[:n_overlap]

            # Linear ramp: weight for new segment goes from 0→1
            weight = torch.linspace(0, 1, n_overlap, dtype=seg_result.dtype)
            weight = weight.view(n_overlap, 1, 1, 1)  # broadcast over H, W, C

            blended = prev_tail * (1.0 - weight) + head * weight
            result_chunks.append(blended)

            # Append the non-overlapping portion of this segment.
            remainder = seg_result[n_overlap:]
        else:
            remainder = seg_result

        if seg_idx < len(segments) - 1:
            # Save the tail for blending with the next segment, and only
            # append the non-tail portion now.
            prev_tail = remainder[-overlap:].clone()
            result_chunks.append(remainder[:-overlap])
        else:
            # Last segment — append everything remaining.
            result_chunks.append(remainder)
            prev_tail = None

        # Free segment tensors.
        del seg_result, seg_images
        torch.cuda.empty_cache()

    result = torch.cat(result_chunks, dim=0)
    del result_chunks
    return result


def run_star_inference(
    star_model: dict,
    images: torch.Tensor,
    upscale: int = 4,
    steps: int = 15,
    guide_scale: float = 7.5,
    prompt: str = "",
    solver_mode: str = "fast",
    max_chunk_len: int = 32,
    seed: int = 0,
    color_fix: str = "adain",
) -> torch.Tensor:
    """Run STAR video super-resolution and return ComfyUI IMAGE batch."""

    device = star_model["device"]
    dtype = star_model["dtype"]
    text_encoder = star_model["text_encoder"]
    generator = star_model["generator"]
    diffusion = star_model["diffusion"]
    vae = star_model["vae"]
    negative_y = star_model["negative_y"]
    offload = star_model.get("offload", "disabled")

    # In aggressive mode use smaller VAE chunks to cut peak VRAM.
    vae_enc_chunk = 1
    vae_dec_chunk = 3
    if offload == "aggressive":
        vae_dec_chunk = 1

    total_noise_levels = 1000

    # -- Convert ComfyUI frames to STAR format --
    video_data = comfyui_to_star_frames(images)  # [F, 3, H, W]

    # Keep a copy at input resolution (on CPU) for colour correction later
    input_frames_star = video_data.clone().cpu()

    frames_num, _, orig_h, orig_w = video_data.shape
    target_h = orig_h * upscale
    target_w = orig_w * upscale

    # -- Bilinear upscale to target resolution --
    video_data = F.interpolate(video_data, size=(target_h, target_w), mode="bilinear", align_corners=False)
    _, _, h, w = video_data.shape

    # -- Pad to model-friendly resolution --
    padding = pad_to_fit(h, w)
    video_data = F.pad(video_data, padding, "constant", 1)

    video_data = video_data.unsqueeze(0).to(device)  # [1, F, 3, H_pad, W_pad]

    # ---- Stage 1: Text encoding ----
    if offload != "disabled":
        text_encoder.model.to(device)
        text_encoder.device = device
    text = prompt if prompt.strip() else cfg.positive_prompt
    y = text_encoder(text).detach()
    if offload != "disabled":
        text_encoder.model.to("cpu")
        text_encoder.device = "cpu"
        torch.cuda.empty_cache()

    # -- Diffusion sampling (autocast needed for fp16 VAE / UNet) --
    with torch.amp.autocast("cuda"):
        # ---- Stage 2: VAE encode ----
        if offload != "disabled":
            _move(vae, device)
        video_data_feature = vae_encode(vae, video_data, chunk_size=vae_enc_chunk)
        if offload != "disabled":
            _move(vae, "cpu")
        # Free the full-res pixel tensor — only latents needed from here.
        del video_data
        torch.cuda.empty_cache()

        t = torch.LongTensor([total_noise_levels - 1]).to(device)
        noised_lr = diffusion.diffuse(video_data_feature, t)

        model_kwargs = [{"y": y}, {"y": negative_y}, {"hint": video_data_feature}]

        torch.cuda.empty_cache()

        chunk_inds = (
            make_chunks(frames_num, interp_f_num=0, max_chunk_len=max_chunk_len)
            if frames_num > max_chunk_len
            else None
        )
        # Need at least 2 chunks; a single chunk causes IndexError in
        # model_chunk_fn when it accesses chunk_inds[1].
        if chunk_inds is not None and len(chunk_inds) < 2:
            chunk_inds = None

        # Monkey-patch trange for progress reporting
        import video_to_video.diffusion.solvers_sdedit as _solvers_mod
        _orig_trange = _solvers_mod.trange

        # Calculate actual number of sigma steps for progress bar
        # (matches logic inside GaussianDiffusion.sample_sr)
        if solver_mode == "fast":
            num_sigma_steps = 14  # 4 coarse + 11 fine = 15 sigmas, trange iterates len-1 = 14
        else:
            num_sigma_steps = steps
        pbar = comfy.utils.ProgressBar(num_sigma_steps)
        _solvers_mod.trange = _make_progress_trange(pbar, num_sigma_steps)

        # ---- Stage 3: Diffusion (UNet) ----
        if offload != "disabled":
            _move(generator, device)
        try:
            torch.manual_seed(seed)

            gen_vid = diffusion.sample_sr(
                noise=noised_lr,
                model=generator,
                model_kwargs=model_kwargs,
                guide_scale=guide_scale,
                guide_rescale=0.2,
                solver="dpmpp_2m_sde",
                solver_mode=solver_mode,
                return_intermediate=None,
                steps=steps,
                t_max=total_noise_levels - 1,
                t_min=0,
                discretization="trailing",
                chunk_inds=chunk_inds,
            )
        finally:
            _solvers_mod.trange = _orig_trange
        if offload != "disabled":
            _move(generator, "cpu")

        # Free latents that are no longer needed.
        del noised_lr, video_data_feature, model_kwargs
        torch.cuda.empty_cache()

        # ---- Stage 4: VAE decode ----
        if offload != "disabled":
            _move(vae, device)
        vid_tensor_gen = vae_decode_chunk(vae, gen_vid, chunk_size=vae_dec_chunk)
        if offload != "disabled":
            _move(vae, "cpu")

    # -- Remove padding --
    w1, w2, h1, h2 = padding
    vid_tensor_gen = vid_tensor_gen[:, :, h1 : h + h1, w1 : w + w1]

    # -- Reshape to [B, C, F, H, W] then convert to ComfyUI format --
    gen_video = rearrange(vid_tensor_gen, "(b f) c h w -> b c f h w", b=1)
    gen_video = gen_video.float().cpu()

    result = star_output_to_comfyui(gen_video)  # [F, H, W, 3]

    # -- Color correction --
    result = apply_color_fix(result, input_frames_star, color_fix)

    torch.cuda.empty_cache()
    mm.soft_empty_cache()

    return result
