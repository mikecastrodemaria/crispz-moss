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
    (the one cz_ideogram.IdeogramPipe and cz_flux3.Flux3Pipe follow):

        pipe(prompt=, negative_prompt=, width=, height=, num_inference_steps=, generator=,
             guidance_scale=, callback_on_step_end=, output_type=).images

    plus `encode_prompt` for cz_pipeline's embedding cache, the encoder-only 'model'
    offload (the DiT and the VAE stay on the card, the 9B encoder comes in for the prompt
    and leaves at the first DiT forward), named refusals for img2img / inpaint (lot 3) and
    editing, and the family detection the LoRA and checkpoint guards rely on.

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


def _has_marker(key, markers):
    k = "." + key
    return any((m if m.startswith(".") else "." + m) in k for m in markers)


def family_from_keys(keys):
    """'moss', 'ideogram', 'flux3', 'flux2' or None from tensor names alone. The strong
    MOSS markers decide first; the other families are cz_ideogram's reading (one source,
    which itself defers to cz_flux3 for FLUX)."""
    keys = [k for k in keys if isinstance(k, str) and k != "__metadata__"]
    if not keys:
        return None
    if any(_has_marker(k, _MOSS_MARKERS) for k in keys):
        return FAMILY
    if any(_has_marker(k, _MOSS_LAYER_MARKERS) for k in keys):
        return FAMILY
    try:
        import cz_ideogram
        return cz_ideogram.family_from_keys(keys)
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
    try:
        import cz_ideogram
        fam = cz_ideogram.family_from_header(hdr)
        if fam:
            return fam
    except Exception:
        pass
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
    callers written for Ideogram's schedule keep working. peak <= 0 -> the official 3.0."""
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


# --- The adapter -------------------------------------------------------------------------
class MossPipe:
    """A MossPipeline behind crispz's calling convention. Unknown attributes are forwarded
    to the real pipeline (transformer, vae, scheduler, text_encoder, tokenizer,
    maybe_free_model_hooks...)."""

    _OWN = {"_pipe", "repo", "variant", "_said", "_cfg", "_all_hooks", "_tr_pre",
            "_te_on_gpu", "_offload_device", "released", "_neg_cache", "skip_layers"}

    def __init__(self, pipe, repo=DEFAULT_REPO):
        object.__setattr__(self, "_pipe", pipe)
        object.__setattr__(self, "repo", repo)
        object.__setattr__(self, "variant", "moss")
        object.__setattr__(self, "_said", set())
        # is_distilled False: a real CFG and a real negative prompt (cz_pipeline reads it).
        object.__setattr__(self, "_cfg", {"is_distilled": False})
        for k in ("_tr_pre", "_offload_device", "_neg_cache"):
            object.__setattr__(self, k, None)
        object.__setattr__(self, "_all_hooks", [])
        object.__setattr__(self, "_te_on_gpu", False)
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

    # --- LoRAs: lot 4 ---------------------------------------------------------------------
    def load_lora_weights(self, *a, **k):
        raise NotImplementedError(
            "MOSS-Image LoRAs are not loadable yet (lot 4: the vendored pipeline has no "
            "LoRA loader, crispz-moss brings its own, ComfyUI's fused-qkv layout included).")

    # --- 'model' offload = the TEXT ENCODER only -----------------------------------------
    # diffusers' enable_model_cpu_offload chains text_encoder -> transformer -> vae, each
    # model sending the previous one back to the CPU when it runs: the DiT would go to the
    # CPU and back for the VAE decode of every image, and its hook on the encoder is tied
    # to forward() calls the pipeline makes one prompt at a time. Here the DiT and the VAE
    # stay on the card; the encoder is moved in explicitly before a call and sent back as
    # soon as the DiT runs (or at the end of the call, Stop and errors included). Same
    # method name, so every enable_model_cpu_offload() of cz_pipeline gets it. No
    # accelerate hook on any component.
    def enable_model_cpu_offload(self, gpu_id=None, device=None):
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
        object.__setattr__(self, "_offload_device", dev)
        object.__setattr__(self, "_all_hooks", ["text_encoder"])   # cz_pipeline counts it
        object.__setattr__(self, "_te_on_gpu", False)

        def tr_pre(module, args):
            if self._te_on_gpu:
                self._encoder_out()
        object.__setattr__(self, "_tr_pre", [p.transformer.register_forward_pre_hook(tr_pre)])

    def _encoder_in(self):
        if self._offload_device is not None and not self._te_on_gpu:
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

    def remove_all_hooks(self):
        for h in list(self._tr_pre or []):
            try:
                h.remove()
            except Exception:
                pass
        object.__setattr__(self, "_tr_pre", None)
        object.__setattr__(self, "_all_hooks", [])
        object.__setattr__(self, "_te_on_gpu", False)
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
        """Should diffusers' own model offload come back on the raw pipeline (a merge from
        crispz-ideogram, a loader), it is replaced, said once."""
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

    def __call__(self, prompt=None, image=None, mask_image=None, strength=None,
                 width=None, height=None, num_inference_steps=None, generator=None,
                 guidance_scale=None, num_images_per_prompt=1, negative_prompt=None,
                 prompt_embeds=None, negative_prompt_embeds=None,
                 prompt_attention_mask=None, negative_prompt_attention_mask=None,
                 callback_on_step_end=None, callback_on_step_end_tensor_inputs=None,
                 output_type="pil", return_dict=True, max_sequence_length=4096,
                 uncond_skip_layers=None, sigmas=None, latents=None, **_ignored):
        if strength is not None or mask_image is not None:
            raise NotImplementedError(
                "img2img / inpaint on MOSS-Image are emulated in lot 3 (SDEdit on the tail "
                "of the native schedule, latent blending for the mask); not available yet.")
        if image is not None:
            raise NotImplementedError(
                "MOSS-Image does not edit images (text to image only). Local editing: send "
                "the image to crispz-flux3 (Edit in crispz-flux3 tab).")
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
        if callback_on_step_end is not None:
            kw["callback_on_step_end"] = callback_on_step_end
        self._keep_encoder_offload()
        if "prompt" in kw:
            self._encoder_in()
        try:
            out = self._pipe(**kw)
        finally:
            self._encoder_out()        # stopped before the first DiT forward, or an error
        images = out.images if hasattr(out, "images") else out[0]
        return types.SimpleNamespace(images=images) if return_dict else (images,)


def _is_quantized(module):
    """A bitsandbytes-quantized module must not be moved with .to() (it would be
    dequantized or refused); it stays where it was loaded."""
    return bool(getattr(module, "is_loaded_in_4bit", False)
                or getattr(module, "is_loaded_in_8bit", False)
                or getattr(module, "hf_quantizer", None) is not None)


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
