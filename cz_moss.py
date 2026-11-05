"""crispz-moss - the MOSS-Image engine (OpenMOSS-Team, Apache 2.0, vendored pipeline).

MOSS-Image (2026-10-11): a 5B single-stream DiT (42 blocks, 3072 wide, 24 heads of 128),
a frozen Qwen3.5-9B text encoder read at its LAST hidden state, and the FLUX.2 VAE
(AutoencoderKLFlux2, 32 channels, bn-normalized 2x2 patches -> 128-channel tokens on a
16 px grid, FLUX channel order). Flow matching, Euler, 50 steps, CFG 3.0 with the
negative branch skipping block 22 ("uncond skip", two forwards a step), shift 5.0.
Text to image only. Weights and pipeline code are Apache 2.0.

The upstream pipeline is `third_party/moss_image/pipeline_moss_image.py`, vendored (never
`trust_remote_code`) and NOT modified: everything crispz needs is added by subclassing:

  - `MossPipeline(MOSSImagePipeline)`: the same denoise loop plus `callback_on_step_end`
    (latents handed as (B, 128, h, w), the step index and the total), `_interrupt` (Stop:
    the current latents are decoded, a partial image comes back), and `sigmas` (an
    explicit schedule, the tail of the native one for the emulated img2img of lot 3).
    tests/test_moss_engine.py checks it renders the very same image as the vendored class.
  - `MossPipe`: the adapter behind the calling convention cz_pipeline was written against
    (the one cz_flux3.Flux3Pipe follows):

        pipe(prompt=, negative_prompt=, width=, height=, num_inference_steps=, generator=,
             guidance_scale=, callback_on_step_end=, output_type=).images

    plus `encode_prompt` for cz_pipeline's embedding cache, the encoder-only 'model'
    offload (the DiT and the VAE stay on the card, the 9B encoder comes in for the prompt
    and leaves at the first DiT forward), img2img and inpaint EMULATED (lot 3: `image` +
    `strength` runs SDEdit on the tail of MOSS's own schedule, `mask_image` adds latent
    blending, see _prepare_emulated), a named refusal for editing (an `image` alone), and
    the family detection the LoRA and checkpoint guards rely on.

Single files (ComfyUI / CivitAI layout, `load_transformer(single_file=...)` and
`load_text_encoder(...)`): the DiT in bf16, fp8 (plain cast or scaled) or ComfyUI's
'int8_tensorwise' with ConvRot (dequantized to bf16 at load time, the fused `proj.qkv`
split into q / k / v), the
text encoder in bf16 or int8 ConvRot (the `model.` prefix stripped, `lm_head` and the
`mtp` head dropped). The 4-bit `asym_w4a8_int8` (GPTQ) builds are refused by name.
"""
import json
import logging
import os
import struct
import types

import torch

DEFAULT_REPO = "OpenMOSS-Team/MOSS-Image-SFT"
BASE_REPO = "OpenMOSS-Team/MOSS-Image-Base"
MOSS_REPOS = (DEFAULT_REPO, BASE_REPO)
FAMILY = "moss"
MODEL_LABEL = "MOSS-Image"

# One variant for cz_pipeline's "steps advised" lines and the profiles (one place).
DEFAULTS = {"moss": {"steps": 50, "guidance": 3.0, "size": 1024}}

# PRECISION in crispz-moss is the TEXT ENCODER's (the DiT is always bf16, 10.2 GB; the
# Qwen3.5-9B encoder is 16.8 GB in bf16 and ~5.6 GB in nf4 through bitsandbytes). The
# first entry is the default. nf4 on Qwen3.5's hybrid (GDN + attention) layers has NOT
# been measured on the 5090 yet: it is offered, and said at load time.
PRECISIONS = ("bf16", "nf4")

# DiT width (24 heads x 128). Used to NAME things and to tell a MOSS LoRA by its shapes.
HIDDEN_SIZE = 3072
NUM_LAYERS = 42

# Official sampling settings (USERGUIDE.md): 50 steps, CFG 3.0 (3 to 6), shift 5.0,
# negative branch skipping block 22 (zero-based). The constant names below are the ones
# cz_pipeline reads for every engine.
DEFAULT_GUIDANCE = 3.0
SCHEDULE_PEAK = DEFAULT_GUIDANCE          # cz_pipeline's name for "the guidance default"
UNCOND_SKIP_LAYERS = (22,)
# The negative tags the user guide recommends (offered as a preset, never forced).
NEGATIVE_PRESET = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"

# Resolution: multiples of 16; the model is validated from 0.25 to 4 Mpx and recommends
# 1, 1.5 or 2 Mpx; any aspect ratio. Sides are clamped to 256..2048 (the 6:1 banners of
# the design presets need short sides), the area to 4 Mpx.
MIN_SIDE, MAX_SIDE, GRID = 256, 2048, 16
MAX_PIXELS = 2048 * 2048
VALIDATED_MIN_PIXELS = 512 * 512

# The local magic prompt / Describe (lot 2): the text encoder IS a frozen Qwen3.5-9B, so
# its own LM head (one tensor, vocab 248320 x 4096, ~2 GB in bf16) makes it generative
# again. Downloaded once, the tensor alone (an HTTP range on the shard), into cache/.
LM_HEAD_REPO = "Qwen/Qwen3.5-9B"
LM_HEAD_KEY = "lm_head.weight"
LM_HEAD_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "moss",
                             "qwen3.5-9b-lm_head.safetensors")
MAGIC_MAX_NEW_TOKENS = 1536
DESCRIBE_MAX_SIDE = 1024

# Offload 'auto': below this much VRAM, 'none' is turned into 'model' (= the text encoder
# alone off the card, see MossPipe.enable_model_cpu_offload). bf16 everything is 27 GB of
# weights: on a 32 GB card the encoder has to leave between prompts.
AUTO_NONE_MIN_VRAM_GB = 40.0

# Single-file quantization formats (ComfyUI's comfy_quant blobs) this engine dequantizes,
# and the ones it refuses by name.
DEQUANT_FORMATS = {"int8_tensorwise", "int8", "float8_e4m3fn", "float8_e5m2",
                   "fp8_e4m3fn", "fp8_e5m2", "fp8"}
REFUSED_FORMATS = {"asym_w4a8_int8": "4-bit GPTQ (w4a8) weights with a codebook",
                   "w6a8_int8": "6-bit packed weights",
                   "convrot_w4a4": "4-bit ConvRot weights",
                   "nvfp4": "NVFP4 packed weights",
                   "mxfp8": "MXFP8 block-scaled weights"}

_log_handler = logging.getLogger("crispz.moss")
_HERE = os.path.dirname(os.path.abspath(__file__))


def _log(msg):
    """cz_core's logger without importing cz_core (keeps this module standalone)."""
    try:
        from cz_core import _log as core_log
        core_log(msg)
    except Exception:
        _log_handler.info(msg)


def _vendored():
    """The vendored upstream module (imported once, lazily: torch and diffusers)."""
    import importlib
    import sys
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    return importlib.import_module("third_party.moss_image.pipeline_moss_image")


# --- Family detection ----------------------------------------------------------------
# MOSS-Image's DiT has heads no other family in crispz carries: video_embedder (the
# 128 -> 3072 token embedding), text_embedder (the Qwen projection + two text blocks),
# shared_block_modulation, final_linear_video / final_norm_video, video_head_modulation.
# Those names identify a full transformer. A LoRA only touches layers.N.proj.{q,k,v,o,
# ffn_up_gate,ffn_down} (ComfyUI fuses q/k/v into layers.N.proj.qkv): the segment
# '.proj.q.' is MOSS's alone in crispz (Ideogram and Z-Image use '.attention.to_q.',
# FLUX '.attn.to_q.' / '.img_attn.qkv.'); the width 3072 confirms it.
_MOSS_MARKERS = ("video_embedder.", "text_embedder.", "shared_block_modulation.",
                 "final_linear_video.", "final_norm_video.", "video_head_modulation.")
_MOSS_LAYER_MARKERS = (".proj.q.", ".proj.k.", ".proj.v.", ".proj.o.", ".proj.qkv.",
                       ".proj.ffn_up_gate.", ".proj.ffn_down.")
# Foreign families, so a stray file in a shared models folder is refused BY NAME (the
# guard names the studio that runs it): Ideogram 4 by its own heads (crispz-ideogram),
# and its LoRAs by their block names at the 4608 width; FLUX.2 / FLUX.3 through
# cz_flux3's reading (one source).
_IDEOGRAM_MARKERS = ("llm_cond_proj.", "llm_cond_norm.", "embed_image_indicator.",
                     "adaln_proj.", "unconditional_transformer.")
_IDEOGRAM_LAYER_MARKERS = (".attention.to_q.", ".attention.to_k.", ".attention.to_v.",
                           ".attention.qkv.", ".attention.o.", ".attention.to_out.",
                           ".feed_forward.w1.", ".feed_forward.w2.", ".feed_forward.w3.")
_IDEOGRAM_HIDDEN_SIZE = 4608


def _has_marker(key, markers):
    k = "." + key
    return any((m if m.startswith(".") else "." + m) in k for m in markers)


def family_from_keys(keys):
    """'moss', 'ideogram', 'flux3', 'flux2' or None from tensor names alone. The strong
    MOSS markers decide first, then Ideogram 4's heads, then cz_flux3's reading of the
    FLUX families (one source)."""
    keys = [k for k in keys if isinstance(k, str) and k != "__metadata__"]
    if not keys:
        return None
    if any(_has_marker(k, _MOSS_MARKERS) for k in keys):
        return FAMILY
    if any(_has_marker(k, _MOSS_LAYER_MARKERS) for k in keys):
        return FAMILY
    if any(_has_marker(k, _IDEOGRAM_MARKERS) for k in keys):
        return "ideogram"
    try:
        import cz_flux3
        return cz_flux3.family_from_keys(keys)
    except Exception:
        return None


def family_from_header(hdr):
    """The same from a safetensors header {name: {"shape": [...]}}, plus the LoRA case:
    layers.N.proj.* matrices whose input or output width is 3072."""
    if not isinstance(hdr, dict):
        return None
    fam = family_from_keys(hdr.keys())
    if fam:
        return fam
    for k, v in hdr.items():
        if k == "__metadata__" or not isinstance(v, dict):
            continue
        if not any(m in k for m in _IDEOGRAM_LAYER_MARKERS):
            continue
        shape = v.get("shape") or []
        if len(shape) == 2 and _IDEOGRAM_HIDDEN_SIZE in (int(shape[0]), int(shape[1])):
            return "ideogram"
    return None


def is_engine_repo(repo):
    """True for the official repos, a mirror carrying 'moss-image' / 'moss_image' in its
    name, or a local folder whose model_index declares MOSSImagePipeline."""
    r = str(repo or "").strip()
    if not r:
        return False
    low = r.lower().replace("_", "-")
    if r in MOSS_REPOS or "moss-image" in low or "mossimage" in low:
        return True
    idx = os.path.join(r, "model_index.json")
    if os.path.isfile(idx):
        try:
            with open(idx, encoding="utf-8") as f:
                return json.load(f).get("_class_name") == "MOSSImagePipeline"
        except Exception:
            return False
    return False


def is_base_repo(repo):
    """MOSS-Image-Base (the pretrained checkpoint, the one LoRAs are trained on) against
    the SFT. Same architecture, same settings: cz_pipeline treats both as 'base' (real
    CFG, negative prompt); this only names things."""
    return "base" in str(repo or "").lower().split("/")[-1]


def variant_of(repo):
    return "moss"


def variant_from_keys(keys):
    return None


def _bfl_quant_of(path):
    """No BFL fp8r / nvfp4 builds for MOSS: kept for cz_pipeline's call sites."""
    return None


# --- Guidance --------------------------------------------------------------------------
def guidance_schedule(steps, peak=DEFAULT_GUIDANCE):
    """A CONSTANT guidance (MOSS has no per-step schedule); the list form is kept so the
    callers that take a per-step list keep working. peak <= 0 -> the official 3.0."""
    n = max(1, int(steps))
    g = float(peak) if peak and float(peak) > 0 else DEFAULT_GUIDANCE
    return [g] * n


def fit_size(width, height):
    """(w, h) on the 16 grid within 256..2048 a side and 4 Mpx, the closest to the ask.
    The pipeline refuses anything off the grid with a ValueError at call time."""
    w = int(width or DEFAULTS["moss"]["size"])
    h = int(height or DEFAULTS["moss"]["size"])
    w = min(MAX_SIDE, max(MIN_SIDE, w // GRID * GRID))
    h = min(MAX_SIDE, max(MIN_SIDE, h // GRID * GRID))
    if w * h > MAX_PIXELS:
        k = (MAX_PIXELS / float(w * h)) ** 0.5
        w = max(MIN_SIDE, int(w * k) // GRID * GRID)
        h = max(MIN_SIDE, int(h * k) // GRID * GRID)
    return w, h


# --- The pipeline subclass -------------------------------------------------------------
_PIPELINE_CLASS = None


def pipeline_class():
    """MossPipeline, built once (the vendored module is imported on first use)."""
    global _PIPELINE_CLASS
    if _PIPELINE_CLASS is not None:
        return _PIPELINE_CLASS
    V = _vendored()
    import math
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
    from diffusers.utils.torch_utils import randn_tensor

    class MossPipeline(V.MOSSImagePipeline):
        """MOSSImagePipeline + callback_on_step_end, _interrupt and an explicit sigma
        schedule. The loop below is the upstream one, line for line, with the three
        additions marked `# crispz`; the validations are upstream's too."""
        _interrupt = False
        _cz_run_steps = None

        @torch.no_grad()
        def __call__(self, prompt=None, *, height=1024, width=1024, num_inference_steps=50,
                     guidance_scale=3.0, negative_prompt='', num_images_per_prompt=1,
                     generator=None, latents=None, prompt_embeds=None,
                     prompt_attention_mask=None, negative_prompt_embeds=None,
                     negative_prompt_attention_mask=None, max_sequence_length=4096,
                     output_type='pil', return_dict=True, uncond_skip_layers=(22,),
                     slg_layers=None, slg_scale=0.0, slg_start=0.0, slg_stop=1.0,
                     sigmas=None, callback_on_step_end=None,
                     callback_on_step_end_tensor_inputs=None):
            self._interrupt = False                                        # crispz
            uncond_skip_layers = V._validate_skip_layers(
                uncond_skip_layers, len(self.transformer.layers), 'uncond_skip_layers')
            slg_layers = V._validate_skip_layers(slg_layers, len(self.transformer.layers),
                                                 'slg_layers')
            if not math.isfinite(slg_scale) or slg_scale < 0:
                raise ValueError('slg_scale must be finite and nonnegative.')
            if not 0 <= slg_start <= slg_stop <= 1:
                raise ValueError('Require 0 <= slg_start <= slg_stop <= 1.')
            if height <= 0 or width <= 0 or height % 16 or width % 16:
                raise ValueError('height and width must be positive multiples of 16.')
            if output_type not in {'pil', 'np', 'pt', 'latent'}:
                raise ValueError('output_type must be pil, np, pt or latent.')
            if (num_images_per_prompt < 1 or not math.isfinite(guidance_scale)
                    or guidance_scale < 0):
                raise ValueError('Invalid image count or guidance scale.')
            if not 1 <= max_sequence_length <= 4096:
                raise ValueError('max_sequence_length must be between 1 and 4096.')
            if (prompt is None) == (prompt_embeds is None):
                raise ValueError('Provide exactly one of prompt and prompt_embeds.')
            device = self._execution_device
            if prompt_embeds is None:
                prompt_embeds, prompt_attention_mask = self.encode_prompt(
                    prompt, max_sequence_length)
            batch = prompt_embeds.shape[0]

            def prepare(embeds, mask):
                if embeds.ndim != 3 or embeds.shape[0] != batch:
                    raise ValueError('Prompt embeddings must be [B, S, C] with matching '
                                     'batch size.')
                if mask is None:
                    mask = torch.ones(embeds.shape[:2], dtype=torch.bool,
                                      device=embeds.device)
                return (embeds.to(device).repeat_interleave(num_images_per_prompt, 0),
                        mask.to(device).repeat_interleave(num_images_per_prompt, 0))
            positive, positive_mask = prepare(prompt_embeds, prompt_attention_mask)
            do_cfg = guidance_scale != 1.0
            if do_cfg:
                if negative_prompt_embeds is None:
                    negatives = ([negative_prompt] * batch if isinstance(negative_prompt, str)
                                 else negative_prompt)
                    if len(negatives) != batch:
                        raise ValueError('negative_prompt must match prompt batch size.')
                    negative_prompt_embeds, negative_prompt_attention_mask = \
                        self.encode_prompt(negatives, max_sequence_length)
                negative, negative_mask = prepare(negative_prompt_embeds,
                                                  negative_prompt_attention_mask)
            shape = (batch * num_images_per_prompt, self.transformer.config.latent_channels,
                     height // 16, width // 16)
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device,
                                       dtype=torch.float32)
            elif tuple(latents.shape) != shape:
                raise ValueError(f'latents must have shape {shape}.')
            else:
                latents = latents.to(device=device, dtype=torch.float32)
            if sigmas is None:                                             # crispz
                self.scheduler.set_timesteps(num_inference_steps, device=device)
            else:
                set_sigmas(self.scheduler, sigmas, device)
            num_steps = len(self.scheduler.timesteps)
            for step_index, timestep in enumerate(self.progress_bar(self.scheduler.timesteps)):
                if self._interrupt:                                        # crispz
                    break
                skip_active = slg_start <= step_index / num_steps < slg_stop
                prediction = self.transformer(latents, timestep, positive,
                                              positive_mask).sample.float()
                conditional = prediction
                if do_cfg:
                    skip_kwargs = ({'skip_layers': uncond_skip_layers}
                                   if uncond_skip_layers and skip_active else {})
                    uncond = self.transformer(latents, timestep, negative, negative_mask,
                                              **skip_kwargs).sample.float()
                    prediction = uncond + guidance_scale * (prediction - uncond)
                if slg_layers and slg_scale > 0 and skip_active:
                    skipped = self.transformer(latents, timestep, positive, positive_mask,
                                               skip_layers=slg_layers).sample.float()
                    if not do_cfg:
                        prediction = skipped
                    prediction = prediction + slg_scale * (conditional - skipped)
                latents = self.scheduler.step(prediction, timestep, latents,
                                              return_dict=False)[0]
                if callback_on_step_end is not None:                       # crispz
                    out = callback_on_step_end(self, step_index, timestep,
                                               {"latents": latents})
                    if isinstance(out, dict) and out.get("latents") is not None:
                        latents = out["latents"]
            if output_type == 'latent':
                images = latents
            else:
                images = self.decode_latents(latents)
                images = self.image_processor.postprocess(images, output_type=output_type)
            self.maybe_free_model_hooks()
            return ImagePipelineOutput(images=images) if return_dict else (images,)

        @torch.no_grad()
        def decode_latents(self, latents):
            """The upstream decode: bn de-normalization on the 128 token channels, the
            (c//4, 2, 2) unpatchify in FLUX order, the VAE. Returns images in -1..1."""
            if self.vae is None:
                raise ValueError('A FLUX.2 VAE is required for decoded output.')
            z = latents.to(dtype=self.vae.dtype)
            bn = self.vae.bn
            eps = self.vae.config.batch_norm_eps
            z = (z * (bn.running_var + eps).sqrt().to(z).view(1, -1, 1, 1)
                 + bn.running_mean.to(z).view(1, -1, 1, 1))
            b, c, h, w = z.shape
            z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
            z = z.reshape(b, c // 4, 2 * h, 2 * w)
            return self.vae.decode(z, return_dict=False)[0].clamp(-1, 1)

    def set_sigmas(scheduler, sigmas, device):
        """An explicit sigma list on the MOSS scheduler, with no shift applied on top
        (upstream snaps and shifts its own list, then sets _shift to 1.0 for the same
        reason)."""
        scheduler._shift = 1.0
        FlowMatchEulerDiscreteScheduler.set_timesteps(
            scheduler, device=device, sigmas=[float(s) for s in sigmas])

    MossPipeline.set_sigmas = staticmethod(set_sigmas)
    _PIPELINE_CLASS = MossPipeline
    return MossPipeline


def native_sigmas(steps, scheduler=None):
    """The sigma schedule the pipeline would run for `steps` (shifted, snapped to the
    training grid, duplicates removed): what the emulated img2img of lot 3 truncates."""
    V = _vendored()
    sch = scheduler if scheduler is not None else V.MOSSImageScheduler()
    sch.set_timesteps(int(steps))
    s = sch.sigmas.detach().float().cpu().tolist()
    return s[:-1] if s and s[-1] == 0.0 else s


# --- Layout prompts (lot 2) ------------------------------------------------------------
def encode_swap_setting():
    """config `moss_encode_swap` (default on), env MOSS_ENCODE_SWAP=0 to turn it off."""
    env = os.environ.get("MOSS_ENCODE_SWAP")
    if env is not None:
        return env.strip().lower() not in ("0", "false", "no", "off")
    try:
        from cz_core import CONFIG
        return bool(CONFIG.get("moss_encode_swap", True))
    except Exception:
        return True


NO_OTHER_TEXT = "no other text"


def _with_no_other_text(caption):
    """The guard that keeps MOSS from inventing words elsewhere (the lot 1 and lot 3
    finding): appended to a caption that does not say it."""
    c = (caption or "").strip()
    if NO_OTHER_TEXT in c.lower():
        return c
    return (c.rstrip(". ") + ". " if c else "") + "No other text."


def prepare_prompt(prompt, width=None, height=None, fit=False, place=None):
    """What the model gets from a prompt string, and the size to render at:
    (prompt, width, height, note). A layout DRAFT (cz_caption: pixel boxes and its own
    canvas) is compiled to the model form and FORCES the size it was drawn for (a layout
    is worth one size); a COMPILED layout or plain text go through as they are. The note
    (or None) is what the caller logs.
    `fit=True` (img2img / inpaint: the IMAGE sets the canvas): a draft drawn for another
    size is redrawn at width x height instead (its boxes rescaled), said in the note.
    `place=(x0, y0, x1, y1)` (outpaint / reframe: the source image sits in that
    sub-rectangle of the width x height canvas): the draft is PLACED there, its boxes
    scaled into the rectangle, not stretched over the canvas, and its summary says "no
    other text" so the new areas get none; plain text gets the same guard."""
    if not isinstance(prompt, str):
        return prompt, width, height, None
    try:
        import cz_caption
    except Exception:
        return prompt, width, height, None
    if not cz_caption.is_caption(prompt):
        if place is not None and prompt.strip():
            guarded = _with_no_other_text(prompt)
            note = None if guarded == prompt.strip() else "outpaint: \"No other text.\" added to the prompt (plain text invents words in the new areas)"
            return guarded, width, height, note
        return prompt, width, height, None
    d = cz_caption.parse(prompt)
    if place is not None and width and height and not cz_caption.is_compiled(d):
        lay, dw, dh = cz_caption.from_text(d)
        px0, py0, px1, py1 = [int(round(v)) for v in place]
        pw, ph = max(16, px1 - px0), max(16, py1 - py0)
        draft = cz_caption.to_draft(lay, pw, ph)
        for it in draft["items"]:
            b = it.get("bbox")
            if b:
                it["bbox"] = [min(max(0, b[0] + px0), int(width)), min(max(0, b[1] + py0), int(height)),
                              min(max(0, b[2] + px0), int(width)), min(max(0, b[3] + py0), int(height))]
        draft["width"], draft["height"] = int(width), int(height)
        draft["summary"]["caption"] = _with_no_other_text(draft["summary"].get("caption", ""))
        compiled, cw, ch = cz_caption.compile(draft)
        return compiled, cw, ch, (f"layout drawn for {dw}x{dh}: placed on the source image at "
                                  f"({px0}, {py0}) to ({px1}, {py1}) of the {int(width)}x{int(height)} "
                                  f"canvas, \"no other text\" around it")
    if cz_caption.is_compiled(d):
        gw, gh = cz_caption.guess_canvas(d)
        if width and height and (gw > int(width) or gh > int(height)):
            return prompt, width, height, (f"layout: its boxes reach {gw}x{gh}, beyond the "
                                           f"{width}x{height} asked (drawn for a larger canvas?)")
        return prompt, width, height, None
    compiled, dw, dh = cz_caption.compile(d)
    note = None
    if width and height and (int(width), int(height)) != (dw, dh):
        if fit:
            lay, _, _ = cz_caption.from_text(d)
            compiled, dw, dh = cz_caption.compile(cz_caption.to_draft(lay, width, height))
            note = (f"layout drawn for {int(d['width'])}x{int(d['height'])}: boxes rescaled "
                    f"to the image's {dw}x{dh} (img2img: the image sets the size)")
        else:
            note = (f"layout drawn for {dw}x{dh}: rendering at that size, not at "
                    f"{int(width)}x{int(height)} (a layout is worth one size)")
    return compiled, dw, dh, note


# --- The adapter -------------------------------------------------------------------------
class MossPipe:
    """A MossPipeline behind crispz's calling convention. Unknown attributes are forwarded
    to the real pipeline (transformer, vae, scheduler, text_encoder, tokenizer,
    maybe_free_model_hooks...)."""

    _OWN = {"_pipe", "repo", "variant", "_said", "_cfg", "_all_hooks", "_tr_pre",
            "_te_on_gpu", "_offload_device", "released", "_neg_cache", "skip_layers",
            "_generator", "_lm_head", "_vl_processor", "_swap_dit", "_dit_parked", "_loras"}

    def __init__(self, pipe, repo=DEFAULT_REPO):
        object.__setattr__(self, "_pipe", pipe)
        object.__setattr__(self, "repo", repo)
        object.__setattr__(self, "variant", "moss")
        object.__setattr__(self, "_said", set())
        # is_distilled False: a real CFG and a real negative prompt (cz_pipeline reads it).
        object.__setattr__(self, "_cfg", {"is_distilled": False})
        for k in ("_tr_pre", "_offload_device", "_neg_cache", "_generator", "_lm_head",
                  "_vl_processor"):
            object.__setattr__(self, k, None)
        object.__setattr__(self, "_all_hooks", [])
        object.__setattr__(self, "_te_on_gpu", False)
        object.__setattr__(self, "_swap_dit", False)
        object.__setattr__(self, "_dit_parked", False)
        object.__setattr__(self, "_loras", [])
        object.__setattr__(self, "released", False)
        object.__setattr__(self, "skip_layers", tuple(UNCOND_SKIP_LAYERS))

    def release(self):
        """Lets go of the real pipeline, so that a reference to THIS wrapper kept by some
        caller no longer pins 27 GB of weights (cz_pipeline.free_vram calls it). A released
        wrapper refuses any use; cz_pipeline swaps it for a fresh pipe."""
        object.__setattr__(self, "_pipe", None)
        object.__setattr__(self, "released", True)
        object.__setattr__(self, "_all_hooks", [])
        object.__setattr__(self, "_neg_cache", None)
        for k in ("_generator", "_lm_head", "_vl_processor"):
            object.__setattr__(self, k, None)

    def __getattr__(self, name):
        inner = object.__getattribute__(self, "_pipe")
        if inner is None:
            raise RuntimeError("this MOSS pipeline was released (VRAM freed): get a fresh "
                               "one from cz_pipeline.get_pipe()")
        return getattr(inner, name)

    def __setattr__(self, name, value):
        if name in self._OWN:
            object.__setattr__(self, name, value)
        elif name == "_interrupt":
            self._pipe._interrupt = value
        else:
            setattr(self._pipe, name, value)

    @property
    def config(self):
        return types.SimpleNamespace(**self._cfg)

    def register_to_config(self, **kw):
        self._cfg.update(kw)

    def register_modules(self, **kw):
        self._pipe.register_modules(**kw)

    def to(self, *args, **kwargs):
        self._pipe.to(*args, **kwargs)
        return self

    # --- LoRAs (lot 4): crispz-moss's own loader, cz_moss_lora ----------------------------
    def load_lora_weights(self, *a, **k):
        raise NotImplementedError(
            "diffusers' LoRA API does not reach the vendored MOSS pipeline: crispz-moss "
            "applies MOSS-Image LoRAs with its own loader, MossPipe.set_loras([(path, "
            "weight), ...]) (kohya, peft, moss-edit-lora and ComfyUI diff layouts).")

    def set_loras(self, wanted):
        """Applies exactly `wanted` = [(path, weight), ...] on the DiT (the previous set
        removed first; the same files with other weights are re-scaled in place).
        Returns the loader's per-file summaries."""
        import cz_moss_lora
        wanted = [(str(p), float(w)) for p, w in (wanted or [])]
        have = list(self.applied_loras)
        tr = self.transformer
        if wanted and [p for p, _ in have] == [p for p, _ in wanted]:
            if cz_moss_lora.rescale(tr, [w for _, w in wanted]):
                object.__setattr__(self, "_loras", wanted)
                _log("LoRA weights updated in place: "
                     + ", ".join(f"{os.path.basename(p)}@{w:g}" for p, w in wanted))
                return []
        reports = cz_moss_lora.apply(tr, wanted)
        object.__setattr__(self, "_loras", wanted)
        return reports

    @property
    def applied_loras(self):
        return list(getattr(self, "_loras", None) or [])

    # --- 'model' offload = the TEXT ENCODER only -----------------------------------------
    # diffusers' enable_model_cpu_offload chains text_encoder -> transformer -> vae, each
    # model sending the previous one back to the CPU when it runs: the DiT would go to the
    # CPU and back for the VAE decode of every image, and its hook on the encoder is tied
    # to forward() calls the pipeline makes one prompt at a time. Here the DiT and the VAE
    # stay on the card; the encoder is moved in explicitly before a call and sent back as
    # soon as the DiT runs (or at the end of the call, Stop and errors included). Same
    # method name, so every enable_model_cpu_offload() of cz_pipeline gets it. No
    # accelerate hook on any component.
    def enable_model_cpu_offload(self, gpu_id=None, device=None, swap_dit=None):
        """The encoder-only offload: the DiT and the VAE stay on the card, the 9B
        encoder comes in for the prompts and leaves at the first DiT forward.
        `swap_dit` (config `moss_encode_swap`, default on): while the encoder is on the
        card, the DiT steps aside to the RAM, so the two never sit there together (the
        bf16 encoder alone made the peak 27 to 30 GB on a 32 GB card; the swap costs
        about a second each way per new prompt). A quantized encoder never moves, so the
        swap is moot and off."""
        p = self._pipe
        self.remove_all_hooks()
        dev = torch.device(device) if device is not None else torch.device(
            "cuda" if gpu_id is None else f"cuda:{int(gpu_id)}")
        for name in ("transformer", "vae"):
            m = getattr(p, name, None)
            if isinstance(m, torch.nn.Module):
                m.to(dev)
        te = getattr(p, "text_encoder", None)
        if te is None:
            return
        if not _is_quantized(te):
            te.to("cpu")
        if swap_dit is None:
            swap_dit = encode_swap_setting()
        swap_dit = bool(swap_dit) and not _is_quantized(te) and not _is_quantized(p.transformer)
        object.__setattr__(self, "_offload_device", dev)
        object.__setattr__(self, "_all_hooks", ["text_encoder"])   # cz_pipeline counts it
        object.__setattr__(self, "_te_on_gpu", False)
        object.__setattr__(self, "_swap_dit", swap_dit)
        object.__setattr__(self, "_dit_parked", False)
        if swap_dit:
            self._say_once("swap", "offload: the DiT steps aside to the RAM while a prompt is "
                                   "encoded (moss_encode_swap), so the encoder never shares "
                                   "the card with it")

        def tr_pre(module, args):
            if self._te_on_gpu or self._dit_parked:
                self._encoder_out()
        object.__setattr__(self, "_tr_pre", [p.transformer.register_forward_pre_hook(tr_pre)])

    def _encoder_in(self):
        if self._offload_device is not None and not self._te_on_gpu:
            if self._swap_dit and not self._dit_parked:
                self._pipe.transformer.to("cpu")
                object.__setattr__(self, "_dit_parked", True)
            te = self._pipe.text_encoder
            if not _is_quantized(te):
                te.to(self._offload_device)
            object.__setattr__(self, "_te_on_gpu", True)

    def _encoder_out(self):
        if self._te_on_gpu:
            te = self._pipe.text_encoder
            if not _is_quantized(te):
                te.to("cpu")
            object.__setattr__(self, "_te_on_gpu", False)
        if self._dit_parked:
            self._pipe.transformer.to(self._offload_device)
            object.__setattr__(self, "_dit_parked", False)

    def remove_all_hooks(self):
        for h in list(self._tr_pre or []):
            try:
                h.remove()
            except Exception:
                pass
        object.__setattr__(self, "_tr_pre", None)
        object.__setattr__(self, "_all_hooks", [])
        object.__setattr__(self, "_te_on_gpu", False)
        if getattr(self, "_dit_parked", False) and self._offload_device is not None:
            self._pipe.transformer.to(self._offload_device)
        object.__setattr__(self, "_dit_parked", False)
        object.__setattr__(self, "_swap_dit", False)
        object.__setattr__(self, "_offload_device", None)
        self._pipe.remove_all_hooks()

    def maybe_free_model_hooks(self):
        """End of a call / release_vram(offload=True): the encoder back on the CPU (the
        DiT and the VAE stay, that is the point)."""
        if self._offload_device is not None:
            self._encoder_out()
        else:
            self._pipe.maybe_free_model_hooks()

    def _keep_encoder_offload(self):
        """Should diffusers' own model offload come back on the raw pipeline (a loader,
        a merge from a sibling studio), it is replaced, said once."""
        if self._offload_device is not None and getattr(self._pipe, "_all_hooks", None):
            self._say_once("reoffload", "offload: diffusers' model offload had come back; "
                                        "encoder-only offload reapplied")
            self.enable_model_cpu_offload(device=self._offload_device)

    # --- Prompts --------------------------------------------------------------------------
    def encode_prompt(self, prompt, device=None, max_sequence_length=4096, **_ignored):
        """(prompt_embeds, prompt_attention_mask) of one prompt or a list, through the
        encoder (brought onto the card and back under the encoder-only offload). The
        tuple's first element is what cz_pipeline's embedding cache keeps: a single
        unpadded prompt needs no mask (the pipeline builds an all-True one)."""
        self._keep_encoder_offload()
        self._encoder_in()
        try:
            return self._pipe.encode_prompt(prompt, int(max_sequence_length))
        finally:
            if self._offload_device is not None:
                self._encoder_out()

    def _negative_embeds(self, negative, max_sequence_length):
        """The negative prompt's embeddings, remembered for the LAST negative text: the
        same negative on every image costs the encoder one trip, not one per image."""
        key = (negative, int(max_sequence_length), id(self._pipe.text_encoder))
        hit = self._neg_cache
        if hit is not None and hit[0] == key:
            return hit[1]
        self._encoder_in()
        emb = self._pipe.encode_prompt(negative, int(max_sequence_length))
        emb = tuple(t.detach() for t in emb)
        object.__setattr__(self, "_neg_cache", (key, emb))
        return emb

    def _say_once(self, key, msg):
        if key not in self._said:
            self._said.add(key)
            _log(msg)

    def _skip_layers_for(self, asked):
        """The uncond-skip layers that exist in THIS transformer (a tiny test model has
        two blocks; the official 22 is dropped there, said once)."""
        layers = tuple(int(i) for i in (self.skip_layers if asked is None else asked))
        n = len(self.transformer.layers)          # through __getattr__: a released pipe says so
        ok = tuple(i for i in layers if 0 <= i < n)
        if ok != layers:
            self._say_once(("skip", layers, n),
                           f"uncond skip layers {list(layers)} -> {list(ok)}: this "
                           f"transformer has {n} blocks")
        return ok

    # --- Local magic prompt and Describe (lot 2) -------------------------------------------
    def ensure_lm_head(self, token=None, repo=LM_HEAD_REPO, cache=LM_HEAD_CACHE):
        """The LM head tensor (vocab x hidden) of the encoder's own Qwen3.5-9B, loaded
        once: from cache/moss/, else downloaded (the one tensor, ~2 GB, said)."""
        if self._lm_head is not None:
            return self._lm_head
        te = self._pipe.text_encoder
        hidden = int(te.config.text_config.hidden_size)
        vocab = int(te.config.text_config.vocab_size)
        w = load_lm_head(cache) if os.path.isfile(cache) else None
        if w is None:
            _log(f"magic prompt: downloading the LM head of {repo} ({LM_HEAD_KEY}, ~2 GB, "
                 f"once) to {cache}")
            download_lm_head(repo, cache, token=token)
            w = load_lm_head(cache)
        if tuple(w.shape) != (vocab, hidden):
            raise RuntimeError(f"LM head {tuple(w.shape)} does not fit this encoder "
                               f"({vocab} x {hidden}): a replacement encoder of another "
                               f"size? Delete {cache} to download {repo}'s again")
        object.__setattr__(self, "_lm_head", w)
        return w

    def set_lm_head(self, weight):
        """A head handed in directly (tests: a tiny random one)."""
        object.__setattr__(self, "_lm_head", weight)
        object.__setattr__(self, "_generator", None)

    def _get_generator(self):
        """Qwen3_5ForConditionalGeneration sharing the loaded encoder as its `model`
        (no second copy of the 9B), with the LM head on top. Built once."""
        if self._generator is not None:
            return self._generator
        from transformers import Qwen3_5ForConditionalGeneration
        te = self._pipe.text_encoder
        head_w = self.ensure_lm_head()
        with torch.device("meta"):
            gen = Qwen3_5ForConditionalGeneration(te.config)
        gen.model = te
        dt = next(p for p in te.parameters()).dtype
        if dt not in (torch.bfloat16, torch.float16, torch.float32):
            dt = torch.bfloat16                 # a quantized encoder: the head in bf16
        head = torch.nn.Linear(head_w.shape[1], head_w.shape[0], bias=False,
                               device="cpu", dtype=dt)
        with torch.no_grad():
            head.weight.copy_(head_w.to(dt))
        gen.lm_head = head
        gen.eval()
        try:
            gen.generation_config.pad_token_id = self._pipe.tokenizer.pad_token_id
        except Exception:
            pass
        object.__setattr__(self, "_generator", gen)
        return gen

    def _processor(self):
        """Qwen3.5's image + text processor (small config files from LM_HEAD_REPO)."""
        proc = self.__dict__.get("_vl_processor")
        if proc is None:
            from transformers import AutoProcessor
            proc = AutoProcessor.from_pretrained(LM_HEAD_REPO)
            object.__setattr__(self, "_vl_processor", proc)
        return proc

    def _device_for_generation(self, device=None):
        return self._offload_device or device or self._pipe._execution_device

    def _generate_text(self, inputs, max_new_tokens, do_sample=False, temperature=0.7,
                       seed=None, device=None, schema=None):
        """Runs the generator on tokenized `inputs` (a BatchEncoding / dict), the
        encoder and the head brought onto the card for the call under the encoder-only
        offload and sent back after. `schema` (a JSON schema dict) constrains the output
        with outlines when it is installed; said once when it is not."""
        gen = self._get_generator()
        dev = self._device_for_generation(device)
        offloaded = self._offload_device is not None
        self._encoder_in()
        gen.lm_head.to(dev)
        try:
            inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
            if seed is not None and int(seed) >= 0:
                torch.manual_seed(int(seed))
            kw = {"max_new_tokens": int(max_new_tokens), "do_sample": bool(do_sample)}
            if do_sample:
                kw["temperature"] = float(temperature)
            with torch.no_grad(), _no_compile_without_triton():
                processors = (_schema_processors(gen, self._pipe.tokenizer, schema)
                              if schema else None)
                if processors:
                    kw["logits_processor"] = processors
                out = gen.generate(**inputs, **kw)
            new = out[:, inputs["input_ids"].shape[1]:]
            return self._pipe.tokenizer.batch_decode(new, skip_special_tokens=True)[0].strip()
        finally:
            if offloaded:
                gen.lm_head.to("cpu")
                self._encoder_out()

    def magic_prompt(self, idea, width, height, seed=None, temperature=0.7,
                     max_new_tokens=MAGIC_MAX_NEW_TOKENS, device=None, constrained=True):
        """Plain idea -> a layout draft text, locally, with the encoder and its LM head
        (cz_magic cleans and validates it)."""
        import cz_magic
        system, user = cz_magic.messages_for(idea, width, height)
        tok = self._pipe.tokenizer
        text = tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tok(text, return_tensors="pt")
        return self._generate_text(inputs, max_new_tokens, do_sample=temperature > 0,
                                   temperature=temperature, seed=seed, device=device,
                                   schema=cz_magic.OLLAMA_FORMAT if constrained else None)

    def describe_image(self, image, max_new_tokens=MAGIC_MAX_NEW_TOKENS, device=None,
                       constrained=True):
        """An image -> a layout draft text (boxes in pixels of the image), with the
        loaded Qwen3.5-9B: its vision tower and the LM head. Greedy: deterministic.
        Constrained to the draft schema like the magic prompt (an unconstrained
        description once came back as invalid JSON on the 5090)."""
        import cz_magic
        proc = self._processor()
        img = image.convert("RGB")
        if max(img.size) > DESCRIBE_MAX_SIDE:
            k = DESCRIBE_MAX_SIDE / float(max(img.size))
            img = img.resize((max(32, round(img.width * k)), max(32, round(img.height * k))))
        system = (cz_magic.DESCRIBE_SYSTEM.replace("{width}", str(image.width))
                  .replace("{height}", str(image.height)))
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Describe this image as the layout draft."}]}]
        inputs = proc.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                          return_dict=True, return_tensors="pt")
        return self._generate_text(inputs, max_new_tokens, do_sample=False, device=device,
                                   schema=cz_magic.OLLAMA_FORMAT if constrained else None)

    # --- img2img / inpaint, emulated (lot 3) ----------------------------------------------
    # MOSSImagePipeline has neither: it always starts from pure noise on its own schedule.
    # SDEdit on a flow-matching model is x_t = (1 - sigma) * x0 + sigma * noise: the image
    # is encoded to MOSS's latents (the exact inverse of decode_latents: the FLUX.2 VAE,
    # the 2x2 patches packed in FLUX order, bn-normalized), noised to the sigma of the
    # first kept step, and only the tail of the schedule runs. The schedule is MOSS's own
    # (shifted, snapped to the training grid, duplicates removed: native_sigmas), cut at
    # the index the strength gives and handed to the pipeline as `sigmas` (no shift on
    # top). The strength is a FRACTION OF THE STEPS, diffusers' and crispz-flux3's rule
    # (0.35 = the last ~35 % of the steps); at least one step always runs. The guidance is
    # constant on MOSS: nothing to rotate. Inpaint: after each step the latents outside
    # the mask are put back to the init image noised to the sigma just reached (latent
    # blending, the crispz-flux3 technique); the mask is max-pooled to the 16-pixel grid.
    def _encode_latents(self, image, width, height, device):
        """Normalized MOSS latents (1, 128, h/16, w/16) of `image` at width x height: the
        exact inverse of MossPipeline.decode_latents (VAE, (c, p, p) packing, bn)."""
        import numpy as np
        from PIL import Image
        vae = self._pipe.vae
        img = image.convert("RGB")
        if img.size != (width, height):
            img = img.resize((width, height), Image.LANCZOS)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1.0)
        x = x.permute(2, 0, 1)[None].to(device, vae.dtype)
        with torch.no_grad():
            r = vae.encode(x).latent_dist.mode().float()
        return self.pack_latents(r, vae)

    @staticmethod
    def pack_latents(r, vae):
        """(b, 32, 2h, 2w) VAE latents -> (b, 128, h, w) MOSS tokens, FLUX channel order
        (c, p, p), bn-normalized with the stats as stored."""
        b, c, hh, ww = r.shape
        z = r.view(b, c, hh // 2, 2, ww // 2, 2).permute(0, 1, 3, 5, 2, 4)
        z = z.reshape(b, 4 * c, hh // 2, ww // 2)
        bn = vae.bn
        mean = bn.running_mean.float().view(1, -1, 1, 1).to(z.device)
        std = torch.sqrt(bn.running_var.float() + vae.config.batch_norm_eps)
        return (z - mean) / std.view(1, -1, 1, 1).to(z.device)

    @staticmethod
    def _mask_latents(mask_image, width, height, device):
        """(1, 1, h/16, w/16): 1 = repaint, 0 = keep; a token is repainted as soon as ANY
        of its pixels is (max-pool), so a thin stroke never vanishes below the grid."""
        import numpy as np
        from PIL import Image
        m = mask_image.convert("L")
        if m.size != (width, height):
            m = m.resize((width, height), Image.NEAREST)
        t = torch.from_numpy(np.asarray(m, dtype=np.float32) / 255.0)[None, None]
        return torch.nn.functional.max_pool2d(t, kernel_size=GRID, stride=GRID).to(device)

    @staticmethod
    def _is_full_mask(mask_image):
        try:
            return mask_image.convert("L").getextrema()[0] >= 250
        except Exception:
            return False

    def _prepare_emulated(self, kw, init, mask_image, strength, steps, w, h, generator,
                          user_cb):
        """Fills kw (latents, sigmas) for an img2img / inpaint call and returns
        (step callback, restore); restore clears the run-steps counter after the call."""
        p = self._pipe
        dev = self._offload_device or p._execution_device
        s = 1.0 if strength is None else float(min(max(float(strength), 0.0), 1.0))
        full = native_sigmas(steps, p.scheduler)       # what this call would run
        n = len(full)
        start = min(int(max(n - n * s, 0)), n - 1)
        sigma0 = float(full[start])
        npp = int(kw.get("num_images_per_prompt") or 1)
        x0 = self._encode_latents(init, w, h, dev).expand(npp, -1, -1, -1).contiguous()
        g = generator[0] if isinstance(generator, (list, tuple)) else generator
        noise = torch.randn(x0.shape, generator=g, dtype=torch.float32,
                            device=g.device if g is not None else "cpu").to(x0.device)
        kw["latents"] = (1.0 - sigma0) * x0 + sigma0 * noise
        kw["sigmas"] = list(full[start:])
        # The live preview counts the steps that RUN (cz_pipeline reads _cz_run_steps);
        # the progress bar iterates the truncated timesteps and is right by itself.
        run = n - start
        p._cz_run_steps = run

        def restore():
            p._cz_run_steps = None
        mask = None
        if mask_image is not None and not self._is_full_mask(mask_image):
            mask = self._mask_latents(mask_image, w, h, x0.device)
        kind = "inpaint" if mask is not None else "img2img"
        _log(f"{kind} (emulated): strength {s:.2f}, {run} of {n} steps from sigma "
             f"{sigma0:.3f}")
        if mask is None:
            return user_cb, restore

        def blend(pipe, i, t, cb_kwargs):
            lat = cb_kwargs["latents"]
            sig = pipe.scheduler.sigmas
            nxt = float(sig[i + 1]) if i + 1 < len(sig) else 0.0
            known = (1.0 - nxt) * x0 + nxt * noise
            cb_kwargs["latents"] = mask * lat + (1.0 - mask) * known.to(lat.dtype)
            if user_cb is not None:
                out = user_cb(pipe, i, t, cb_kwargs)
                if isinstance(out, dict):
                    cb_kwargs = out
            return cb_kwargs
        return blend, restore

    def __call__(self, prompt=None, image=None, mask_image=None, strength=None,
                 width=None, height=None, num_inference_steps=None, generator=None,
                 guidance_scale=None, num_images_per_prompt=1, negative_prompt=None,
                 prompt_embeds=None, negative_prompt_embeds=None,
                 prompt_attention_mask=None, negative_prompt_attention_mask=None,
                 callback_on_step_end=None, callback_on_step_end_tensor_inputs=None,
                 output_type="pil", return_dict=True, max_sequence_length=4096,
                 uncond_skip_layers=None, sigmas=None, latents=None, **_ignored):
        emulated = strength is not None or mask_image is not None
        if emulated:
            init = image[0] if isinstance(image, (list, tuple)) and image else image
            if init is None:
                raise ValueError("img2img / inpaint need an image to start from")
            # Output size = the asked one, else the image's own (on MOSS's grid).
            width = width or init.size[0]
            height = height or init.size[1]
        elif image is not None:
            raise NotImplementedError(
                "MOSS-Image does not edit images (text to image only). Local editing: send "
                "the image to crispz-flux3 (Edit in crispz-flux3 tab).")
        if isinstance(prompt, str):
            prompt, width, height, note = prepare_prompt(prompt, width, height, fit=emulated)
            if note:
                self._say_once(("layout", note), note)
        steps = int(num_inference_steps or DEFAULTS["moss"]["steps"])
        w, h = fit_size(width, height)
        if (width and height) and (w, h) != (int(width), int(height)):
            self._say_once(("size", int(width), int(height)),
                           f"size {width}x{height} -> {w}x{h} (MOSS-Image: multiples of 16, "
                           f"{MIN_SIDE} to {MAX_SIDE} a side, 4 Mpx at most)")
        if w * h < VALIDATED_MIN_PIXELS:
            self._say_once(("small", w, h),
                           f"{w}x{h} is under the 0.25 Mpx MOSS-Image was validated at "
                           f"(1 to 2 Mpx recommended): the model extrapolates, quality may drop")
        g = float(guidance_scale) if guidance_scale is not None else DEFAULT_GUIDANCE
        if g < 0:
            g = DEFAULT_GUIDANCE
        kw = {"width": w, "height": h, "num_inference_steps": steps, "guidance_scale": g,
              "num_images_per_prompt": int(num_images_per_prompt or 1),
              "generator": generator, "output_type": output_type,
              "max_sequence_length": int(max_sequence_length),
              "uncond_skip_layers": self._skip_layers_for(uncond_skip_layers)}
        if prompt_embeds is not None:
            kw["prompt_embeds"] = prompt_embeds
            kw["prompt_attention_mask"] = prompt_attention_mask
        else:
            kw["prompt"] = prompt if prompt is not None else ""
        neg = negative_prompt if negative_prompt is not None else ""
        if negative_prompt_embeds is not None:
            kw["negative_prompt_embeds"] = negative_prompt_embeds
            kw["negative_prompt_attention_mask"] = negative_prompt_attention_mask
        elif isinstance(neg, str) and g != 1.0:
            # Encoded here (and remembered) rather than by the pipeline on every call;
            # the encoder only comes onto the card on a miss.
            self._keep_encoder_offload()
            try:
                e, m = self._negative_embeds(neg, max_sequence_length)
            except Exception:
                self._encoder_out()
                raise
            kw["negative_prompt_embeds"], kw["negative_prompt_attention_mask"] = e, m
        else:
            kw["negative_prompt"] = neg
        if sigmas is not None:
            kw["sigmas"] = list(sigmas)
        if latents is not None:
            kw["latents"] = latents
        restore = None
        if emulated:
            callback_on_step_end, restore = self._prepare_emulated(
                kw, init, mask_image, strength, steps, w, h, generator, callback_on_step_end)
        if callback_on_step_end is not None:
            kw["callback_on_step_end"] = callback_on_step_end
        self._keep_encoder_offload()
        if "prompt" in kw:
            self._encoder_in()
        try:
            out = self._pipe(**kw)
        finally:
            self._encoder_out()        # stopped before the first DiT forward, or an error
            if restore is not None:
                restore()
        images = out.images if hasattr(out, "images") else out[0]
        return types.SimpleNamespace(images=images) if return_dict else (images,)


def _is_quantized(module):
    """A bitsandbytes-quantized module must not be moved with .to() (it would be
    dequantized or refused); it stays where it was loaded."""
    return bool(getattr(module, "is_loaded_in_4bit", False)
                or getattr(module, "is_loaded_in_8bit", False)
                or getattr(module, "hf_quantizer", None) is not None)


def _no_compile_without_triton():
    """outlines-core wraps its logits-mask kernel in torch.compile, which needs Triton;
    on Windows there is none, so every constrained generation printed two pages of
    TritonMissing traceback before PyTorch fell back to eager (seen on crispz-ideogram).
    Without Triton, dynamo is switched off for the call."""
    import contextlib
    import importlib.util
    if importlib.util.find_spec("triton") is not None:
        return contextlib.nullcontext()
    import torch._dynamo
    return torch._dynamo.config.patch(disable=True)


_OUTLINES_SAID = set()


def _schema_processors(model, tokenizer, schema):
    """A transformers LogitsProcessorList that constrains the output to `schema`
    (outlines 1.x: its outlines-core JSON-schema guide, built for this tokenizer), or
    None when outlines is not installed (said once) or refuses the tokenizer."""
    try:
        import outlines
        from outlines.backends import get_json_schema_logits_processor
        from transformers import LogitsProcessorList
    except Exception as e:
        if "outlines" not in _OUTLINES_SAID:
            _OUTLINES_SAID.add("outlines")
            _log(f"magic prompt: `outlines` is not usable ({type(e).__name__}: {e}), the "
                 f"local engine runs unconstrained and may return invalid JSON")
        return None
    try:
        om = outlines.from_transformers(model, tokenizer)
        proc = get_json_schema_logits_processor(None, om, json.dumps(schema))
        proc.reset()
        return LogitsProcessorList([proc])
    except Exception as e:
        if "guide" not in _OUTLINES_SAID:
            _OUTLINES_SAID.add("guide")
            _log(f"magic prompt: outlines could not build the JSON guide "
                 f"({type(e).__name__}: {e}); unconstrained")
        return None


def load_lm_head(path):
    """The LM head tensor from a one-tensor safetensors (cache), or None."""
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            return f.get_tensor(LM_HEAD_KEY if LM_HEAD_KEY in keys else keys[0])
    except Exception as e:
        _log(f"LM head cache {path} unreadable ({e}): downloading again")
        return None


def download_lm_head(repo=LM_HEAD_REPO, dest=LM_HEAD_CACHE, token=None, key=LM_HEAD_KEY):
    """Fetches ONE tensor of a sharded HF repo with an HTTP range on its shard (the
    header says where it is), and writes it as a one-tensor safetensors at `dest`. Falls
    back to downloading the whole shard (hf_hub_download) when the range is refused."""
    from huggingface_hub import hf_hub_download, hf_hub_url
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    kw = {"token": token} if token else {}
    index = hf_hub_download(repo, "model.safetensors.index.json", **kw)
    with open(index, encoding="utf-8") as f:
        shard = json.load(f)["weight_map"][key]
    url = hf_hub_url(repo, shard)
    try:
        _range_fetch_tensor(url, key, dest, token)
        return dest
    except Exception as e:
        _log(f"LM head: ranged download refused ({type(e).__name__}: {e}); downloading "
             f"the whole shard {shard} instead")
    from safetensors import safe_open
    from safetensors.torch import save_file
    path = hf_hub_download(repo, shard, **kw)
    with safe_open(path, framework="pt", device="cpu") as f:
        save_file({key: f.get_tensor(key).contiguous()}, dest)
    return dest


def _range_fetch_tensor(url, key, dest, token=None, chunk=16 << 20):
    import requests
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    def get(rng, stream=False):
        r = requests.get(url, headers={**headers, "Range": rng}, allow_redirects=True,
                         timeout=120, stream=stream)
        if r.status_code != 206:
            raise RuntimeError(f"HTTP {r.status_code} for range {rng}")
        return r
    n = struct.unpack("<Q", get("bytes=0-7").content[:8])[0]
    hdr = json.loads(get(f"bytes=8-{8 + n - 1}").content)
    meta = hdr[key]
    a, b = meta["data_offsets"]
    out_hdr = json.dumps({key: {"dtype": meta["dtype"], "shape": meta["shape"],
                                "data_offsets": [0, b - a]}}).encode()
    out_hdr += b" " * ((8 - len(out_hdr) % 8) % 8)
    tmp = dest + ".part"
    with open(tmp, "wb") as f, get(f"bytes={8 + n + a}-{8 + n + b - 1}", stream=True) as r:
        f.write(struct.pack("<Q", len(out_hdr)) + out_hdr)
        got = 0
        for piece in r.iter_content(chunk_size=chunk):
            f.write(piece)
            got += len(piece)
    if got != b - a:
        os.remove(tmp)
        raise RuntimeError(f"short read: {got} of {b - a} bytes")
    os.replace(tmp, dest)


# --- Loading -------------------------------------------------------------------------------
def _components_kw(token):
    kw = {}
    if token:
        kw["token"] = token
    return kw


def load_scheduler(repo, token=None):
    V = _vendored()
    try:
        return V.MOSSImageScheduler.from_pretrained(repo, subfolder="scheduler",
                                                    **_components_kw(token))
    except Exception as e:
        _log(f"scheduler config of {repo} not read ({type(e).__name__}: {e}); "
             f"using the built-in MOSS schedule (shift 5.0)")
        return V.MOSSImageScheduler()


def load_transformer(repo, dtype=torch.bfloat16, precision=None, single_file=None,
                     token=None):
    """The DiT: from a diffusers repo / folder (subfolder transformer), or from a single
    .safetensors in the ComfyUI / CivitAI layout (bf16 or int8 ConvRot, fused qkv)."""
    V = _vendored()
    if single_file:
        sd, cfg = read_single_file_transformer(single_file, dtype)
        with torch.device("meta"):
            tr = V.MOSSImageTransformer(**cfg)
        missing, unexpected = tr.load_state_dict(sd, strict=False, assign=True)
        missing = [k for k in missing if k != "rope.bands"]
        if missing or unexpected:
            raise RuntimeError(
                f"{os.path.basename(single_file)}: not a complete MOSS-Image transformer "
                f"({len(missing)} missing, {len(unexpected)} unexpected tensors; first: "
                f"{(missing + list(unexpected))[:3]})")
        tr.rope.bands = tr.rope._default_bands(torch.device("cpu"))
        return tr.to(dtype).eval()
    return V.MOSSImageTransformer.from_pretrained(repo, subfolder="transformer",
                                                  torch_dtype=dtype, **_components_kw(token))


def _text_encoder_quant_config(precision):
    if precision != "nf4":
        return None
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=False)


def load_text_encoder(src, dtype=torch.bfloat16, precision="bf16", token=None,
                      config_repo=DEFAULT_REPO):
    """The Qwen3.5-9B encoder (Qwen3_5Model, vision tower included) from:
      - a diffusers repo / folder: subfolder text_encoder (the default source);
      - a transformers folder / HF repo ('owner/repo' or 'owner/repo/subfolder');
      - a single .safetensors in the ComfyUI / CivitAI layout (bf16 or int8 ConvRot,
        'model.' prefix, lm_head and mtp dropped), architecture from `config_repo`.
    precision 'nf4' quantizes with bitsandbytes at load time (repo / folder sources)."""
    from transformers import AutoModel
    kw = {"dtype": dtype, **_components_kw(token)}
    q = _text_encoder_quant_config(precision)
    if q is not None:
        kw["quantization_config"] = q
    src = str(src)
    if _looks_single_file(src):
        if q is not None:
            _log("text encoder precision nf4 ignored for a single file: dequantized to bf16")
        return load_single_file_text_encoder(src, dtype, config_repo=config_repo, token=token)
    if os.path.isdir(src):
        sub = "text_encoder" if os.path.isdir(os.path.join(src, "text_encoder")) else None
        return AutoModel.from_pretrained(src, subfolder=sub, **kw).eval()
    parts = src.strip("/").split("/")
    if len(parts) >= 3:                         # owner/repo/subfolder
        return AutoModel.from_pretrained("/".join(parts[:2]), subfolder="/".join(parts[2:]),
                                         **kw).eval()
    try:                                        # a diffusers repo (text_encoder/)
        return AutoModel.from_pretrained(src, subfolder="text_encoder", **kw).eval()
    except (OSError, ValueError, EnvironmentError):
        return AutoModel.from_pretrained(src, **kw).eval()   # a plain transformers repo


def load(repo=DEFAULT_REPO, dtype=torch.bfloat16, precision="bf16", transformer=None,
         text_encoder=None, token=None, **_ignored):
    """A ready MossPipe on the CPU (the caller places it). Each component is loaded by
    its class (no model_index / trust_remote_code round trip): the DiT and the scheduler
    from the vendored classes, the encoder and the tokenizer with transformers, the VAE
    with diffusers' AutoencoderKLFlux2 in float32 (upstream's choice)."""
    from diffusers import AutoencoderKLFlux2
    from transformers import AutoTokenizer
    cls = pipeline_class()
    ck = _components_kw(token)
    if transformer is None:
        transformer = load_transformer(repo, dtype, token=token)
    scheduler = load_scheduler(repo, token)
    if text_encoder is None:
        if precision not in PRECISIONS:
            _log(f"text encoder precision '{precision}' unknown -> bf16")
            precision = "bf16"
        if precision == "nf4":
            _log("text encoder in nf4 (bitsandbytes): ~5.6 GB instead of 16.8; not yet "
                 "measured on the 5090 (Qwen3.5's linear-attention layers)")
        text_encoder = load_text_encoder(repo, dtype, precision, token)
    tokenizer = AutoTokenizer.from_pretrained(repo, subfolder="tokenizer", **ck)
    vae = AutoencoderKLFlux2.from_pretrained(repo, subfolder="vae", torch_dtype=torch.float32,
                                             **ck).eval()
    pipe = cls(transformer.eval(), scheduler, text_encoder, tokenizer, vae)
    return MossPipe(pipe, repo=repo)


# --- Single files: ComfyUI / CivitAI layout --------------------------------------------
_COMFY_PREFIXES = ("model.diffusion_model.", "diffusion_model.")


def _looks_single_file(p):
    return isinstance(p, str) and p.lower().endswith((".safetensors", ".sft"))


def safetensors_header(path):
    """The JSON header of a .safetensors (names, dtypes, shapes, metadata), nothing else."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(min(n, 50_000_000)).decode("utf-8", "ignore"))


def _hadamard(n):
    """ComfyUI's ConvRot 'regular hadamard' (NOT Sylvester's): the base H4 below, Kronecker
    products up to n (a power of 4), normalised by 1/sqrt(n). Orthonormal and symmetric,
    so undoing the rotation is one more multiplication by the same matrix."""
    h4 = torch.tensor([[1., 1., 1., -1.], [1., 1., -1., 1.],
                       [1., -1., 1., 1.], [-1., 1., 1., 1.]])
    H = h4
    while H.shape[0] < n:
        H = torch.kron(H, h4)
    if H.shape[0] != n:
        raise ValueError(f"convrot groupsize {n} is not a power of 4")
    return H / float(n) ** 0.5


def rotate_convrot(w, group_size):
    """W -> W_rot (what ComfyUI stores before quantizing) and back: the same function
    both ways (H is its own inverse)."""
    g = int(group_size)
    if w.dim() != 2 or w.shape[1] % g:
        raise ValueError(f"convrot: {tuple(w.shape)} is not divisible by groups of {g}")
    H = _hadamard(g).to(w.device, w.dtype)
    return (w.reshape(w.shape[0], -1, g) @ H).reshape(w.shape[0], -1)


def quant_formats(path):
    """{tensor stem: comfy_quant dict} declared in a single file (per-tensor blobs and the
    central __metadata__._quantization_metadata), read from the file; {} when none."""
    from safetensors import safe_open
    hdr = safetensors_header(path)
    out = {}
    try:
        qm = json.loads((hdr.get("__metadata__") or {}).get("_quantization_metadata") or "{}")
        for lk, lv in (qm.get("layers") or {}).items():
            if isinstance(lv, dict):
                out[_strip_prefix(lk)] = lv
    except Exception:
        pass
    blobs = [k for k in hdr if k.endswith(".comfy_quant")]
    if blobs:
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in blobs:
                try:
                    out[_strip_prefix(k)[:-len(".comfy_quant")]] = json.loads(
                        bytes(f.get_tensor(k).tolist()).decode("utf-8"))
                except Exception:
                    pass
    return out


def single_file_refusal(path):
    """A reason (str) when `path` is not a loadable MOSS single file, else None. Header
    and quant blobs only, never a weight."""
    try:
        hdr = safetensors_header(path)
    except Exception:
        return "its header cannot be read (a truncated download, or not a .safetensors)"
    names = [k for k in hdr if k != "__metadata__"]
    if any(".lora_" in n or ".lora." in n for n in names[:300]):
        return "it is a LoRA, not a checkpoint: put it in the LoRA folder"
    fam = family_from_keys(names)
    if fam != FAMILY:
        return None if fam is None else f"a {fam} file"
    try:
        fmts = {str(v.get("format", "")).lower() for v in quant_formats(path).values()}
    except Exception:
        fmts = set()
    odd = sorted(f for f in fmts if f in REFUSED_FORMATS)
    if odd:
        return (f"ComfyUI '{odd[0]}' build ({REFUSED_FORMATS[odd[0]]}): crispz-moss "
                f"dequantizes the bf16 and INT8 (ConvRot) builds only; take the bf16 or "
                f"int8 file of the same page")
    unknown = sorted(f for f in fmts if f and f not in DEQUANT_FORMATS)
    if unknown:
        return (f"ComfyUI '{unknown[0]}' build: a quantization format crispz-moss has no "
                f"dequantizer for; take the bf16 or int8 file of the same page")
    return None


def _strip_prefix(k):
    for p in _COMFY_PREFIXES:
        if k.startswith(p):
            return k[len(p):]
    return k


def read_single_file_transformer(path, dtype=torch.bfloat16, device=None):
    """(state dict in the vendored layout, transformer config) from a ComfyUI / CivitAI
    single file: prefix stripped, config blobs dropped, int8 / fp8 dequantized (scale, then
    the ConvRot undone), the fused `layers.N.proj.qkv` split into q / k / v. The config
    is read from the shapes (dim, ffn, layers, text_dim, latent channels)."""
    why = single_file_refusal(path)
    if why:
        raise RuntimeError(f"{os.path.basename(path)}: {why}")
    sd = read_single_file(path, dtype, device)
    n = split_fused_qkv(sd)
    cfg = transformer_config_from_sd(sd)
    _log(f"{os.path.basename(path)}: ComfyUI layout read"
         + (f", {n} fused qkv split" if n else "")
         + f" ({cfg['num_layers']} blocks, {cfg['dim']} wide)")
    return sd, cfg


def transformer_config_from_sd(sd):
    V = _vendored()
    q = sd.get("layers.0.proj.q.weight")
    up = sd.get("layers.0.proj.ffn_up_gate.weight")
    te = sd.get("text_embedder.proj.weight")
    ve = sd.get("video_embedder.weight")
    tm = sd.get("time_embedding.0.weight")
    if q is None or up is None or te is None or ve is None:
        raise RuntimeError("not a MOSS-Image transformer (layers.0.proj.q / ffn_up_gate / "
                           "text_embedder.proj / video_embedder missing)")
    dim = int(q.shape[1])
    layers = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("layers."))
    cfg = dict(dim=dim, ffn_dim=int(up.shape[0]) // 2, num_layers=layers,
               text_dim=int(te.shape[1]), latent_channels=int(ve.shape[1]),
               freq_dim=int(tm.shape[1]) if tm is not None else 256)
    # The head count is not in the shapes: 24 heads of 128 at 3072; a tiny file (tests)
    # falls back to head_dim 16.
    cfg["num_heads"] = dim // 128 if dim % 128 == 0 and dim >= 1024 else max(1, dim // 16)
    if dim == HIDDEN_SIZE and layers != NUM_LAYERS:
        _log(f"single file: {layers} blocks where MOSS-Image has {NUM_LAYERS} (a pruned "
             f"or experimental build?)")
    return cfg


def split_fused_qkv(sd):
    """ComfyUI fuses the DiT's attention projections into one 'layers.N.proj.qkv.weight'
    [3 * dim, dim] (q, k, v in that order, checked against the diffusers shards: the three
    [3072, 3072] blocks of the bf16 single file). The text blocks' qkv is fused upstream
    too and stays. Returns how many were split."""
    n = 0
    for k in [k for k in sd if k.endswith(".proj.qkv.weight")]:
        w = sd.pop(k)
        if w.dim() != 2 or w.shape[0] % 3:
            sd[k] = w
            continue
        q, kk, v = torch.chunk(w, 3, dim=0)
        stem = k[:-len("qkv.weight")]
        sd[stem + "q.weight"] = q.contiguous()
        sd[stem + "k.weight"] = kk.contiguous()
        sd[stem + "v.weight"] = v.contiguous()
        n += 1
    return n


def read_single_file(path, dtype=torch.bfloat16, device=None, keep=None):
    """Every tensor of a ComfyUI / CivitAI single file, prefix stripped, in `dtype`,
    dequantized where a comfy_quant blob says so (per-row or per-tensor scale, then the
    ConvRot rotation undone), the blobs and scales consumed. `keep(name)` filters the
    names (after the prefix). Sequential read in the file's physical order."""
    from safetensors import safe_open
    hdr = safetensors_header(path)
    qcfg = quant_formats(path)
    entries = [(k, v) for k, v in hdr.items() if k != "__metadata__" and isinstance(v, dict)]
    entries.sort(key=lambda kv: kv[1].get("data_offsets", [0])[0])
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    raw, sd = {}, {}
    n_dq = n_rot = 0
    with safe_open(path, framework="pt", device="cpu") as f:
        for k, _ in entries:
            kk = _strip_prefix(k)
            if kk.endswith(".comfy_quant") or ".comfy_attention." in kk or kk.endswith(".config"):
                continue
            if keep is not None and not keep(kk):
                continue
            raw[kk] = f.get_tensor(k)
    for k in list(raw):
        if k.endswith((".weight_scale", ".scale_weight", ".input_scale", ".scale_input")):
            continue
        t = raw.pop(k)
        if t.dtype in (torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2):
            stem = k[:-len(".weight")] if k.endswith(".weight") else k
            s = raw.get(k + "_scale", raw.get(stem + ".scale_weight"))
            is_fp8 = t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            if s is None and not is_fp8:
                raise RuntimeError(f"{os.path.basename(path)}: {k} is quantized "
                                   f"({t.dtype}) with no weight_scale next to it")
            cfg = qcfg.get(stem) or {}
            fmt = str(cfg.get("format", "")).lower()
            if fmt in REFUSED_FORMATS:
                raise RuntimeError(f"{os.path.basename(path)}: {k} is a '{fmt}' tensor "
                                   f"({REFUSED_FORMATS[fmt]}), no dequantizer here")
            x = t.to(dev).to(torch.float32)
            if s is None:
                pass                            # a plain fp8 cast (ComfyUI's e4m3fn files)
            else:
                sc = s.to(dev).to(torch.float32)
                if sc.dim() == 2 and sc.shape[0] == x.shape[0] and sc.shape[1] == 1:
                    x = x * sc
                elif sc.numel() == 1:
                    x = x * sc.reshape(())
                elif sc.dim() == 1 and sc.numel() == x.shape[0]:
                    x = x * sc.view(-1, 1)
                else:
                    raise RuntimeError(f"{os.path.basename(path)}: {k}: weight_scale of "
                                       f"shape {tuple(sc.shape)} for a weight {tuple(x.shape)}")
            if cfg.get("convrot") and x.dim() == 2:
                x = rotate_convrot(x, int(cfg.get("convrot_groupsize", 256) or 256))
                n_rot += 1
            t = x.to(dtype).cpu()
            n_dq += 1
        elif t.is_floating_point() and t.dtype != dtype:
            t = t.to(dtype)
        sd[k] = t
    raw.clear()
    if n_dq:
        _log(f"{os.path.basename(path)}: {n_dq} tensors dequantized to "
             f"{str(dtype).replace('torch.', '')} on {dev}"
             + (f", {n_rot} ConvRot rotations undone" if n_rot else "")
             + " (RAM footprint of the full bf16; the saving was disk and download)")
    if dev != "cpu":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    return sd


_TE_PREFIXES = ("model.",)
_TE_DROP = ("lm_head.", "mtp.")


def text_encoder_keys(names):
    """ComfyUI's Qwen3.5 single files keep the ForConditionalGeneration layout: 'model.'
    in front of language_model / visual, plus lm_head and the mtp head. Qwen3_5Model
    wants language_model.* and visual.* only."""
    out = {}
    for n in names:
        if n.startswith(_TE_DROP):
            continue
        k = n
        for p in _TE_PREFIXES:
            if k.startswith(p):
                k = k[len(p):]
        if k.startswith(("language_model.", "visual.")):
            out[n] = k
    return out


def is_text_encoder_file(path):
    try:
        names = [k for k in safetensors_header(path) if k != "__metadata__"]
    except Exception:
        return False
    return any(("language_model." in n and ".layers." in n) for n in names)


def load_single_file_text_encoder(path, dtype=torch.bfloat16, config_repo=DEFAULT_REPO,
                                  token=None, config=None):
    """A Qwen3_5Model from a ComfyUI / CivitAI text-encoder file (bf16 or int8 ConvRot).
    The architecture config is the official repo's text_encoder/config.json unless given."""
    from transformers import AutoConfig, Qwen3_5Model
    why = None
    try:
        fmts = {str(v.get("format", "")).lower() for v in quant_formats(path).values()}
        odd = sorted(f for f in fmts if f in REFUSED_FORMATS)
        if odd:
            why = (f"ComfyUI '{odd[0]}' text encoder ({REFUSED_FORMATS[odd[0]]}): take the "
                   f"bf16 or int8 file of the same page")
    except Exception as e:
        why = f"unreadable ({e})"
    if why:
        raise RuntimeError(f"{os.path.basename(path)}: {why}")
    if config is None:
        config = AutoConfig.from_pretrained(config_repo, subfolder="text_encoder",
                                            **_components_kw(token))
    names = [k for k in safetensors_header(path) if k != "__metadata__"]
    ren = text_encoder_keys([_strip_prefix(n) for n in names])
    if not ren:
        raise RuntimeError(f"{os.path.basename(path)}: no Qwen3.5 language_model tensor in it")
    sd = read_single_file(path, dtype, keep=lambda k: k in ren)
    sd = {ren[k]: v for k, v in sd.items()}
    # transformers builds the model on 'meta', assigns these tensors and rebuilds the
    # buffers the file does not carry (the rotary tables): no second copy in RAM.
    with torch.device("meta"):
        probe = Qwen3_5Model(config)
    want = set(probe.state_dict().keys())
    del probe
    missing, unexpected = sorted(want - set(sd)), sorted(set(sd) - want)
    if missing or unexpected:
        raise RuntimeError(
            f"{os.path.basename(path)}: not a complete Qwen3.5 encoder ({len(missing)} "
            f"missing, {len(unexpected)} unexpected; first: {(missing + unexpected)[:3]})")
    enc = Qwen3_5Model.from_pretrained(None, config=config, state_dict=sd, dtype=dtype)
    _log(f"{os.path.basename(path)}: Qwen3.5 encoder read from the ComfyUI layout "
         f"(lm_head and mtp head dropped)")
    return enc.eval()
