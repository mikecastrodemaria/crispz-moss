"""crispz family - CLI protocol v1: JSON spec in, JSON out.

Contract shared by the whole family (crispz-studio, crispz-qwen-edit, crispz-klein, crispz-flux3, crispz-ideogram,
crispz-krea*, and comics2crispz which drives them):

    czp caps                       -> this tool's capabilities + running instance
    czp gen --spec spec.json       -> generate ONE image, print the JSON result

Principle: the CLI is the CONTRACT, not the process. `gen` routes to the
running Gradio instance (hidden api_name endpoints 'cli_caps' / 'cli_gen',
same HTTP mechanics as the Asset Browser / Comic Studio SPAs): the app's queue
serializes the GPU and the model stays warm. Without an instance, local
execution (cold path, night batch). Heavy imports (torch, pipeline) happen
ONLY on that local path: `caps` and the routing never load anything.

Output: always ONE JSON line on stdout. Exit codes:
    0 ok - 1 run error - 2 invalid spec - 3 unsupported op/protocol -
    4 no route (no instance and no local execution possible).

v1: one image per call (the caller loops); spec.guidance, when present, is
the CFG of THAT call on every op (the instance's slider is restored after it,
said in the log); on the remote route spec.model is ignored
with a warning (we never swap the model of the user's running instance under
their feet).

loras: spec.loras = ["file.safetensors:0.8", ...], hot-swapped per call. The
prompt may also carry <lora:file[:weight]> tags (A1111/Civitai habit): they
are extracted at validation time and merged into spec.loras (explicit entries
win on duplicates), so style/character LoRAs travel INSIDE panel texts too.

refs (v2): spec.refs = LOCAL image paths (the protocol is machine-local).
When an Omni model is configured (zimage_omni_model), generation goes through
generate_omni (multi-reference, character consistency); otherwise refs are
dropped WITH a warning and supports.refs announces it upfront. A ref missing
on disk is an error (code 2), never a silently degraded generation. At most
10 refs (FLUX.3's <ref_image_0> ... <ref_image_9>; the UI offers 4 slots, the protocol
takes the full 10), extras are cut with a warning.
"""

import os
import sys
import json
import time
import argparse
import contextlib
import re
import urllib.request

from cz_core import split_dir_spec  # noqa: E402
from cz_core import CONFIG, APP_VERSION, _log
from prompt_variants import has_variants

PROTOCOL = 1
TOOL = "crispz-moss"
# The op vocabulary of protocol v1: frozen, identical in every family
# tool. czp accepts all of it on the command line so that an op THIS tool
# does not implement still gets a JSON answer with exit code 3 - a machine
# caller cannot read an argparse usage dump on stderr.
PROTOCOL_OPS = ("caps", "gen", "upscale", "edit", "inpaint")
# What this tool implements: a subset of the vocabulary (all of it today).
OPS = ("caps", "gen", "upscale", "edit", "inpaint")

# What this MODEL FAMILY can do at all (diffusers pipelines that exist for
# it). Static on purpose: `caps` must answer without importing the pipeline
# (torch). Krea 2 has no img2img / inpaint pipeline -> variation (upscale
# with factor 1) and inpaint are refused with exit 3, never faked.
# crispz-moss: MOSS-Image is text to image only; img2img and inpaint are EMULATED by
# cz_moss (lot 3: SDEdit on the tail of the native schedule, latent blending for the
# mask, the crispz-ideogram technique). They work for every caller, and `caps` says they
# are emulated (supports.emulated) so a caller can tell them from a native pipeline.
FAMILY_CAPS = {"img2img": True, "inpaint": True}
EMULATED_CAPS = ("img2img", "inpaint")

# Spec fields known to the protocol (v1). An unknown field is a warning,
# never an error: a spec written for another family tool must go through.
SPEC_FIELDS = ("protocol", "op", "prompt", "negative", "width", "height",
               "seed", "steps", "guidance", "refs", "loras", "model", "input", "mask",
               "out_dir", "count", "detail_faces", "detail_hands",
               "factor", "denoise", "fast")

_DEF_URL = "http://127.0.0.1:7860"

MAX_REFS = 4                      # the Moss-Image-Edit LoRA: the image to edit + up to 3 references
# A variation (upscale, factor <= 1) refines the whole image up to this many pixels: MOSS
# extrapolates to about 4 Mpx in one pass. Beyond, the tiled refine (and its rules,
# reported in the answer's warnings).
VARIATION_WHOLE_MAX_PX = 4_200_000

# spec.fast (op edit) -> the mode name on the cz_pipeline.set_edit_speed side
# klein has no speed preset: the model is already distilled to 4 steps and no FLUX.2
# Lightning LoRA exists. Only 'off' stays valid; the other aliases are kept so that a
# spec written for qwen-edit goes through (warning), not an error.
_FAST_ALIASES = {"off": "Off", "none": "Off", "auto": "Auto (model profile)",
                 "lightning-4": "Lightning 4 steps", "lightning4": "Lightning 4 steps",
                 "lightning 4 steps": "Lightning 4 steps",
                 "lightning-8": "Lightning 8 steps", "lightning8": "Lightning 8 steps",
                 "lightning 8 steps": "Lightning 8 steps"}

# Whether this model family has a multi-reference/omni pipeline AT ALL.
# crispz-krea / crispz-krea2 hard-set this to False (their _load_omni raises:
# no instruction-edit model exists for Krea) - config can never turn refs on
# there, and caps must say so instead of promising a capability that raises.
# crispz-moss: reference editing runs in the base pipeline through the Moss-Image-Edit
# LoRA (lot 4b, cz_moss.EditPrefix): supported by the family, AVAILABLE once that LoRA is
# on disk (cz_edit_loras.edit_ready). Native multi-reference editing: crispz-flux3.
OMNI_SUPPORTED = True
# klein serves multi-reference editing from the BASE pipeline (`image` takes a list of
# PIL images): there is no separate edit repo to configure, so refs are ALWAYS available.
OMNI_DEFAULT = (os.environ.get("MOSS_MODEL")
                or "OpenMOSS-Team/MOSS-Image-SFT")

# klein-4B is step-wise distilled: diffusers ignores `guidance_scale` (measured
# bit-identical renders from 1.0 to 8.0, tests/test_klein_guidance.py) and the
# Flux2Klein* pipelines expose no `negative_prompt`. A spec carrying `negative` is
# therefore honoured with a WARNING, never silently - house rule.
# crispz-moss: MOSS-Image runs a real CFG with a real negative prompt (encoded by its
# Qwen3.5 text encoder; empty by default, the user guide suggests a tag preset).
SUPPORTS_NEGATIVE = True


def _omni_configured():
    """Reference editing is available when the Moss-Image-Edit LoRA is on disk (the
    base pipeline edits through its reference prefix; the base weights alone cannot).
    No config key: the Edit tab downloads the preset on first use."""
    if not OMNI_SUPPORTED:
        return False
    try:
        import cz_edit_loras
        return bool(cz_edit_loras.edit_ready())
    except Exception as e:
        _log(f"edit LoRA lookup failed: {e}")
        return False


def instance_url():
    cfg = CONFIG.get("cli_protocol")
    return (cfg or {}).get("instance_url", _DEF_URL) if isinstance(cfg, dict) \
        else _DEF_URL


def _faces_available():
    """Is the face-detection stack (insightface) installed? Light check:
    module spec only, nothing imported - caps must stay fast."""
    import importlib.util
    return importlib.util.find_spec("insightface") is not None


def _models_dir(kind):
    """Model folder ('checkpoints'|'loras'), same priorities as cz_pipeline
    (env > preferences > config > default) but without importing it: caps
    must stay light."""
    from cz_core import HERE, _prefs
    return (os.environ.get(kind.upper() + "_DIR")
            or _prefs.get(f"{kind}_dir") or CONFIG.get(f"{kind}_dir")
            or os.path.join(HERE, kind))


def _list_model_files(kind, exts=(".safetensors", ".gguf")):
    """Available model files (POSIX relative paths, sorted). A plain disk
    listing - each tool stays the SOURCE OF TRUTH for its models, callers
    (the comics2crispz wizard) configure no path at all."""
    dirs_to_scan = [_models_dir(kind)]
    if kind == "loras":
        # extra LoRA folders (loras_extra_dirs), same rule as cz_pipeline:
        # merged, and the main one wins on a duplicate name.
        from cz_core import _prefs
        # explicit env (even empty) > preferences > config
        extra = (os.environ["LORAS_EXTRA_DIRS"] if "LORAS_EXTRA_DIRS" in os.environ
                 else (_prefs.get("loras_extra_dirs") or CONFIG.get("loras_extra_dirs") or []))
        if isinstance(extra, str):
            extra = split_dir_spec(extra)
        for x in extra:
            x = str(x or "").strip()
            if x and x not in dirs_to_scan:
                dirs_to_scan.append(x)
    out, seen = [], set()
    for d in dirs_to_scan:
        if not os.path.isdir(d):
            continue
        for root, dirs, files in os.walk(d):
            dirs[:] = [x for x in dirs
                       if x not in ("_index", ".cache", "recipes")]
            for f in files:
                if f.lower().endswith(exts):
                    rel = (os.path.relpath(os.path.join(root, f), d)
                           .replace("\\", "/"))
                    if rel.lower() not in seen:
                        seen.add(rel.lower())
                        out.append(rel)
    return sorted(out, key=str.lower)


def _model_loaded():
    """The model active in THIS process: filled in on the instance side
    (endpoint cli_caps), '' on the cold czp side (cz_pipeline is not
    imported - and we do NOT import it for this)."""
    mod = sys.modules.get("cz_pipeline")
    if mod is None:
        return ""
    t = getattr(mod, "ZIMAGE_TRANSFORMER", None)
    if t:
        return os.path.basename(str(t))
    return str(getattr(mod, "BASE_REPO", "") or "")


def _hands_available():
    """Can the hand detailer run? A light check, nothing is imported.

    AT RUNTIME it only needs onnxruntime + the .onnx exported once into cache/.
    'ultralytics' serves THAT export only, in a subprocess -- its docstring in
    cz_detailer explains why it must never live in the diffusion process (it
    corrupts the shared weights during offload transfers). Requiring ultralytics
    here therefore declared the feature missing for anyone who had followed that
    advice and exported the .onnx from a throwaway venv: the detector was ready,
    and the app answered 'detail_hands: false'."""
    import importlib.util
    import os
    if importlib.util.find_spec("onnxruntime") is None:
        return False
    if importlib.util.find_spec("ultralytics") is not None:
        return True                     # the export can happen on demand
    try:
        import cz_detailer
        from cz_core import HERE
        stem = os.path.splitext(os.path.basename(cz_detailer._HAND_MODEL))[0]
        return os.path.isfile(os.path.join(HERE, "cache", stem + ".onnx"))
    except Exception:
        return False


def caps_dict():
    """This tool's capabilities. Light: config + disk listing, no model
    loaded. supports.refs = an Omni model is configured (multi-reference
    generation will actually work); supports.faces = the instance can serve
    face detection (cli_faces endpoint). models/loras = the files this tool
    can use (its own dirs - callers never configure model paths);
    model_loaded = what THIS process has active (instance side)."""
    return {"ok": True, "protocol": PROTOCOL, "tool": TOOL,
            "version": APP_VERSION, "ops": list(OPS),
            "supports": {"loras": True, "refs": _omni_configured(),
                         "max_refs": MAX_REFS, "seed": True,
                         "negative": SUPPORTS_NEGATIVE, "arbitrary_size": True,
                         "faces": _faces_available(),
                         "detail_faces": _faces_available(),
                         "detail_hands": _hands_available(),
                         "edit": _omni_configured(),
                         "edit_presets": bool(_edit_lora_catalog()),
                         "inpaint": FAMILY_CAPS["inpaint"],
                         "img2img": FAMILY_CAPS["img2img"],
                         # spec.guidance = the CFG of that call, every op
                         "guidance": True,
                         "emulated": list(EMULATED_CAPS),
                         "cloud": _cloud_caps()},
            "model_loaded": _model_loaded(),
            "models": _list_model_files("checkpoints"),
            "loras": _list_model_files("loras"),
            # Edit presets (HF LoRAs, downloaded on demand): a caller puts
            # their `name` in spec.loras on the edit op. `inputs` = 2 ->
            # input + 1 ref (spec.refs). Light listing, nothing is downloaded.
            "edit_loras": _edit_lora_catalog(),
            # Fast modes spec.fast accepts on the edit op.
            # MOSS-Image editing (the Moss-Image-Edit LoRA) has no Lightning LoRA: the list comes
            # down to 'off' and the UI hides the control. The field stays present
            # (not absent) so that family clients always read the same one.
            "edit_fast": ["off"]}


def _cloud_caps():
    """No cloud on crispz-moss: MOSS-Image has no hosted API and nothing leaves the
    machine. The field stays (family contract), always off."""
    return {"enabled": False, "ops": [], "why": "no cloud function on crispz-moss"}


def _edit_lora_catalog():
    try:
        import cz_edit_loras
        return cz_edit_loras.catalog()
    except Exception as e:
        _log(f"edit LoRA catalog unavailable: {e}")
        return []


# ----------------------------------------------------------------------------
# Spec
# ----------------------------------------------------------------------------
# <lora:file[:weight]> tags INSIDE the prompt (the A1111/Civitai habit).
# Extracted at validation time so every route (remote endpoint, local run)
# and every caller (czp, comics2crispz panel texts) gets them, and the text
# encoder never sees the tag itself. Explicit spec.loras win on duplicates
# (first weight wins, the family rule).
_LORA_TAG = re.compile(r"<lora:([^:>]+?)(?::([0-9.]+))?>", re.IGNORECASE)


def _extract_prompt_loras(prompt, loras):
    """Moves <lora:...> tags from the prompt into `loras` (in place).
    Returns the cleaned prompt."""
    def _sub(m):
        name = m.group(1).strip()
        if name:
            seen = {str(x).split(":", 1)[0].strip().lower() for x in loras}
            if name.lower() not in seen:
                loras.append(f"{name}:{m.group(2)}" if m.group(2) else name)
        return " "
    out = _LORA_TAG.sub(_sub, prompt)
    if out != prompt:
        out = re.sub(r"\s{2,}", " ", out)
        out = re.sub(r"\s+,", ",", out).strip(" ,")
    return out


def _expand_spec(spec, seed):
    """Variant groups {a|b|c} + __wildcards__ of the prompt AND the negative,
    bound to `seed` (index 0: one image per call), THEN the <lora:...> tags
    of the EXPANDED prompt are moved into spec['loras'] - a LoRA written in
    an option that was not picked is never applied. Returns a new dict (the
    caller's spec is untouched). A prompt without group nor placeholder makes
    no random draw: same text, same image as before."""
    from cz_prompt import expand_prompt_pair
    out = dict(spec)
    loras = list(out.get("loras") or [])
    prompt, negative = expand_prompt_pair(out.get("prompt", ""),
                                          out.get("negative", ""), seed,
                                          index=0)
    out["prompt"] = _extract_prompt_loras(prompt or "", loras)
    # No strip_lora_tags in this cz_prompt: same regex as the extraction.
    out["negative"] = _LORA_TAG.sub("", negative or "").strip()
    out["loras"] = loras
    return out


class SpecError(Exception):
    def __init__(self, msg, code=2):
        super().__init__(msg)
        self.code = code


def validate_spec(spec, op="gen"):
    """Normalize a 'gen' spec. Returns (normalized_spec, warnings).
    Raises SpecError(code=2) when invalid, SpecError(code=3) when the
    protocol/op is out of contract. Never truncates silently."""
    if not isinstance(spec, dict):
        raise SpecError("spec must be a JSON object")
    try:
        proto = int(spec.get("protocol"))
    except (TypeError, ValueError):
        raise SpecError("missing/invalid 'protocol' (expected 1)")
    if proto != PROTOCOL:
        raise SpecError(f"protocol {proto} not supported (this tool speaks "
                        f"{PROTOCOL})", code=3)
    sop = spec.get("op") or op
    if sop != op:
        raise SpecError(f"spec op '{sop}' does not match command '{op}'")
    if op not in OPS:
        raise SpecError(f"op '{op}' not supported by {TOOL} (v1: {OPS})",
                        code=3)
    warnings = [f"unknown field '{k}' ignored" for k in spec
                if k not in SPEC_FIELDS]
    out = {"protocol": PROTOCOL, "op": op}
    out["prompt"] = str(spec.get("prompt") or "").strip()
    if op == "gen" and not out["prompt"]:
        raise SpecError("empty 'prompt'")
    if op == "edit":
        # edit = image + INSTRUCTION -> image, through the tool's omni/edit
        # pipeline. On klein the base pipeline edits by itself, so no flat refusal
        # (code 3) - and never a BRAND NEW image quietly made instead.
        if not out["prompt"]:
            raise SpecError("edit requires 'prompt' (the edit instruction)")
        inp = str(spec.get("input") or "").strip()
        if not inp:
            raise SpecError("edit requires 'input' (image path)")
        if not os.path.isfile(inp):
            raise SpecError(f"input not found on disk: {inp}")
        if not _omni_configured():
            raise SpecError(f"edit not available on {TOOL}: the Moss-Image-Edit LoRA is "
                            f"not on disk (pick the preset once in the Edit tab, it is "
                            f"downloaded; or put moss_image_edit_lora_rank_64_bf16"
                            f".safetensors in loras/)", code=3)
        out["input"] = inp
        spec = dict(spec)
        spec["refs"] = [inp]
        # size_explicit is now computed in the common trunk (further down), for
        # EVERY op: it only served the 'edit' op, so a 'gen' WITH refs -- which takes
        # the same omni route -- came back with the size of the REFERENCE instead of
        # the one asked for.
        # fast = the edit speed mode: "off" | "auto" (the model's profile,
        # Rapid-AIO / Lightning merge) | "lightning-4" | "lightning-8" (stacked
        # Lightning LoRA). Absent -> the instance's own setting stands.
        if spec.get("fast") is not None:
            f = str(spec["fast"]).strip().lower()
            if f not in _FAST_ALIASES:
                raise SpecError(f"invalid 'fast' {spec['fast']!r} (expected one of "
                                f"{', '.join(sorted(_FAST_ALIASES))})")
            out["fast"] = _FAST_ALIASES[f]
        if not spec.get("width") or not spec.get("height"):
            # default size = the image's own (aligned to 32, clamped)
            try:
                from PIL import Image
                with Image.open(inp) as im:
                    iw, ih = im.size
                spec.setdefault("width", max(64, min(2048, iw // 32 * 32)))
                spec.setdefault("height", max(64, min(2048, ih // 32 * 32)))
            except Exception as e:
                raise SpecError(f"input unreadable: {e}")
    if op == "inpaint":
        # inpaint = image + MASK (white = to redraw) + local prompt -> image.
        # Flat refusal (code 3) when the family has no such pipeline.
        if not FAMILY_CAPS["inpaint"]:
            raise SpecError(f"inpaint not supported by {TOOL}: this model "
                            f"family has no inpaint pipeline", code=3)
        inp = str(spec.get("input") or "").strip()
        if not inp:
            raise SpecError("inpaint requires 'input' (image path)")
        if not os.path.isfile(inp):
            raise SpecError(f"input not found on disk: {inp}")
        msk = str(spec.get("mask") or "").strip()
        if not msk:
            raise SpecError("inpaint requires 'mask' (PNG path, white = "
                            "area to redraw)")
        if not os.path.isfile(msk):
            raise SpecError(f"mask not found on disk: {msk}")
        out["input"], out["mask"] = inp, msk
        den = spec.get("denoise")
        if den is not None:
            try:
                den = float(den)
            except (TypeError, ValueError):
                raise SpecError("invalid 'denoise'")
            if not 0.0 <= den <= 1.0:
                raise SpecError("'denoise' out of range (0-1)")
        out["denoise"] = den
    if op == "upscale":
        inp = str(spec.get("input") or "").strip()
        if not inp:
            raise SpecError("upscale requires 'input' (image path)")
        if not os.path.isfile(inp):
            raise SpecError(f"input not found on disk: {inp} (paths are "
                            f"LOCAL - the caller resolves them)")
        out["input"] = inp
        try:
            out["factor"] = float(spec.get("factor") or 2.0)
        except (TypeError, ValueError):
            raise SpecError("invalid 'factor'")
        if not 1.0 <= out["factor"] <= 8.0:
            raise SpecError(f"'factor' out of range (1-8): {out['factor']}")
        if out["factor"] <= 1.0 and not FAMILY_CAPS["img2img"]:
            raise SpecError(f"variation (upscale with factor 1 = img2img "
                            f"refine) not supported by {TOOL}: this model "
                            f"family has no img2img pipeline", code=3)
        den = spec.get("denoise")
        if den is not None:
            try:
                den = float(den)
            except (TypeError, ValueError):
                raise SpecError("invalid 'denoise'")
            if not 0.0 <= den <= 1.0:
                raise SpecError("'denoise' out of range (0-1)")
        out["denoise"] = den
    # The size the caller ASKED FOR. True as soon as the spec carries width AND
    # height, whatever the op: the omni route (edit, or gen with refs) then passes
    # those dimensions to the pipe instead of letting the edit keep the input
    # image's own. Without it an 800x1312 panel rendered from a 1280x832 reference
    # came back in landscape and got cropped at layout time.
    out["size_explicit"] = bool(spec.get("width") and spec.get("height"))
    out["negative"] = str(spec.get("negative") or "").strip()
    if out["negative"] and not SUPPORTS_NEGATIVE:
        # klein is step-wise distilled: no CFG -> no negative prompt. We say so
        # instead of letting the caller believe it acts (degradation announced, never silent).
        warnings.append("'negative' ignored by this model - see supports.negative")
        out["negative"] = ""
    for key, default in (("width", 1024), ("height", 1024)):
        try:
            v = int(spec.get(key) or default)
        except (TypeError, ValueError):
            raise SpecError(f"invalid '{key}'")
        if not 64 <= v <= 4096:
            raise SpecError(f"'{key}' out of range (64-4096): {v}")
        out[key] = v
    try:
        out["seed"] = int(spec.get("seed", -1))
    except (TypeError, ValueError):
        raise SpecError("invalid 'seed'")
    steps = spec.get("steps")
    if steps is not None:
        try:
            steps = int(steps)
        except (TypeError, ValueError):
            raise SpecError("invalid 'steps'")
    out["steps"] = steps
    if spec.get("guidance") is not None:
        # Applied on EVERY op as a per-call override (edit since lot 4b; gen,
        # upscale and inpaint since 2026-11-09, for crispz-motion): a distilled
        # preset (Turbo 4 steps, DMD 8 steps) wants CFG 1.0 while the instance's
        # slider stays at 3.0. The instance's own value is restored after the call.
        try:
            out["guidance"] = float(spec["guidance"])
        except (TypeError, ValueError):
            raise SpecError("invalid 'guidance'")
        if not 0.0 <= out["guidance"] <= 20.0:
            raise SpecError(f"'guidance' out of range (0-20): {out['guidance']}")
    # (crispz-klein warned here that guidance had no effect: true of a distilled model,
    # false of MOSS-Image, a real CFG. Removed.)
    refs = [str(r) for r in (spec.get("refs") or []) if str(r).strip()]
    # Without the edit LoRA on disk, refs are dropped WITH a warning that says how to
    # get it (and where native reference editing lives: crispz-flux3).
    if refs and not _omni_configured():
        warnings.append(f"{len(refs)} ref(s) ignored: MOSS-Image edits from references "
                        f"only with the Moss-Image-Edit LoRA, not on disk (Edit tab "
                        f"preset); native reference editing: crispz-flux3")
        refs = []
    if len(refs) > MAX_REFS:
        warnings.append(f"{len(refs)} refs, keeping the first {MAX_REFS} "
                        f"(the Moss-Image-Edit LoRA was trained for up to {MAX_REFS})")
        refs = refs[:MAX_REFS]
    for r in refs:
        if not os.path.isfile(r):
            raise SpecError(f"ref not found on disk: {r} (refs are LOCAL "
                            f"file paths - the caller resolves them)")
    out["refs"] = refs
    out["loras"] = [str(x) for x in (spec.get("loras") or [])]
    # A prompt with {a|b|c} groups keeps its <lora:...> tags until run time
    # (_expand_spec): only the option actually picked may bring its LoRA.
    if not has_variants(out["prompt"]):
        out["prompt"] = _extract_prompt_loras(out["prompt"], out["loras"])
        if op == "gen" and not out["prompt"]:
            raise SpecError("prompt contained only <lora:...> tags - describe "
                            "the image too")
    out["model"] = (str(spec["model"]).strip()
                    if spec.get("model") else None)
    out["out_dir"] = (str(spec["out_dir"]).strip()
                      if spec.get("out_dir") else None)
    count = spec.get("count")
    if count is not None and int(count) != 1:
        warnings.append("count forced to 1 (v1: one image per call, loop on "
                        "the caller side)")
    # ADetailer-style detailer: true/false forces it, absent (None) = the
    # tool's current setting (config face_detailer) for faces, OFF for hands
    # (ultralytics is optional - never implicit).
    for key in ("detail_faces", "detail_hands"):
        v = spec.get(key)
        out[key] = None if v is None else bool(v)
    return out, warnings


# ----------------------------------------------------------------------------
# Local execution (heavy imports HERE only)
# ----------------------------------------------------------------------------
@contextlib.contextmanager
def _call_guidance(spec, warnings):
    """spec.guidance as the pipeline's CFG for the duration of ONE call, then
    the instance's own value back (UI slider, config). Every MOSS render reads
    cz_pipeline.GUIDANCE through _qwen_call, so swapping it covers gen, the
    detailer, img2img / upscale and inpaint alike. The Gradio queue runs one
    job at a time: no UI render can see the override. Yields the value
    applied (None when the spec carries none)."""
    g = spec.get("guidance")
    if g is None:
        yield None
        return
    import cz_pipeline
    if not hasattr(cz_pipeline, "GUIDANCE"):
        warnings.append("'guidance' not applied: this pipeline exposes no "
                        "GUIDANCE setting")
        yield None
        return
    old = cz_pipeline.GUIDANCE
    cz_pipeline.GUIDANCE = float(g)
    if float(old) != float(g):
        _log(f"protocol: cfg {float(g):g} for this call (instance setting "
            f"{float(old):g}, restored after)")
    try:
        yield float(g)
    finally:
        cz_pipeline.GUIDANCE = old


def _resolve_draft(name):
    """A draft preset name ('moss-turbo-4-steps', 'Draft (4 steps, Turbo LoRA)',
    'moss-dmd-8-steps') -> its LoRA file, found on disk or downloaded once
    (licence in the log), exactly what the UI's Performance preset does. Any
    other name is returned as is (resolved later by cz_pipeline)."""
    try:
        import cz_edit_loras
    except Exception:
        return name
    if not hasattr(cz_edit_loras, "draft_spec") or \
            cz_edit_loras.draft_spec(name) is None:
        return name
    return cz_edit_loras.resolve_draft(name)


def _lora_slots(spec, resolve_drafts=True):
    """spec.loras -> [(name_or_path, weight)]. 'name:0.8' carries its weight,
    a bare name takes the default; a Windows path's drive colon is never read
    as a weight."""
    import cz_pipeline
    slots = []
    for s in spec.get("loras") or []:
        head, _, tail = str(s).rpartition(":")
        try:
            name, weight = head, float(tail)
            if not name:
                raise ValueError
        except ValueError:
            name, weight = str(s), float(cz_pipeline.LORA_WEIGHT)
        slots.append((_resolve_draft(name) if resolve_drafts else name, weight))
    return slots


def _apply_spec_loras(spec):
    """spec.loras on the base pipe for THIS call (gen, upscale / img2img,
    inpaint). No loras in the spec = the app's own set, untouched. The app's
    set comes back after the call (_keep_app_loras)."""
    import cz_pipeline
    if spec.get("loras") and hasattr(cz_pipeline, "set_loras"):
        cz_pipeline.set_loras(_lora_slots(spec))


@contextlib.contextmanager
def _keep_app_loras():
    """The app's LoRA set (its UI slots) back after a protocol call that set
    its own: before 2026-11-09 a gen with spec.loras left them loaded in the
    app, and the user's next render inherited them. Restored even when the
    render fails; hot-swapped back on the app's next run."""
    import cz_pipeline
    if not hasattr(cz_pipeline, "LORAS"):
        yield
        return
    old = list(cz_pipeline.LORAS)
    try:
        yield
    finally:
        if list(cz_pipeline.LORAS) != old:
            cz_pipeline.LORAS = old
            _log("protocol: the app's LoRAs are back ("
                 + (", ".join(f"{os.path.basename(str(p))}@{w}" for p, w in old)
                    or "none") + "), applied on its next run")


def run_gen(spec, warnings=None, route="local"):
    """Generate ONE image from a VALIDATED spec, with spec.guidance (if any)
    as the CFG and spec.loras (if any) as the LoRAs of this call only. See
    _run_gen."""
    warnings = list(warnings or [])
    with _call_guidance(spec, warnings) as g, _keep_app_loras():
        res = _run_gen(spec, warnings, route)
    if g is not None:
        res["guidance"] = g
    return res


def _run_gen(spec, warnings=None, route="local"):
    """Generate ONE image from a VALIDATED spec. Loads the pipeline when
    needed (cold path) - never call this while an instance is rendering on
    the same GPU: that is what the remote route is for."""
    import cz_pipeline
    from cz_imageio import build_output_path, save_image

    warnings = list(warnings or [])
    t0 = time.time()
    if spec.get("model"):
        cz_pipeline.set_zimage_model(spec["model"])
    # The seed is resolved HERE, the same way the UI does (cz_ui.run): a -1
    # becomes a concrete value BEFORE generating -> seed_used is always exact
    # and replayable. (Do NOT read cz_pipeline._LAST_SEED afterwards: only the
    # UI path sets it, it would report the instance's last UI render.)
    import random
    seed_used = int(spec.get("seed", -1))
    if seed_used < 0:
        seed_used = random.randint(0, 2**31 - 1)
    cz_pipeline._LAST_SEED = seed_used          # the UI's 'Reuse last seed' sees it
    # {a|b|c} + wildcards bound to that seed, then <lora:...> of the result.
    spec = _expand_spec(spec, seed_used)
    if not spec["prompt"]:
        raise SpecError("prompt contained only <lora:...> tags - describe "
                        "the image too")
    refs = spec.get("refs") or []
    if refs and hasattr(cz_pipeline, "set_edit_loras"):
        # omni/edit route: the LoRAs go on the EDIT pipe (a set separate from
        # the base one), and a cz_edit_loras preset name is resolved there
        # (downloaded on demand). Without loras -> the edit set is empty for
        # this call (the spec stands alone: the UI's presets do not reach it).
        cz_pipeline.set_edit_loras(_lora_slots(spec, resolve_drafts=False))
        cz_pipeline.set_edit_loras_enabled(True)
    else:
        # Base, or the pipeline of a family tool with no edit set
        # (cz_protocol.py is shared): the old behaviour, with a warning.
        if refs and spec.get("loras"):
            warnings.append("loras applied on the base pipe (this pipeline "
                            "has no edit LoRA set)")
        _apply_spec_loras(spec)
    steps = spec.get("steps") or int(CONFIG.get("default_gen_steps", 8))
    refs = spec.get("refs") or []
    if refs:
        # Omni route (multi-reference): refs were already validated by
        # validate_spec (files exist, omni configured).
        from PIL import Image
        t_o = time.time()
        imgs = [Image.open(r) for r in refs]
        try:
            # kwargs only when asked for: generate_omni's historical signature
            # (family tools) stays valid otherwise.
            extra = {}
            if spec.get("guidance") is not None:
                extra["guidance"] = float(spec["guidance"])
            if spec.get("size_explicit"):
                extra["honor_size"] = True
            if spec.get("steps") is not None:
                extra["steps_explicit"] = True     # the fast mode does not overwrite them
            if spec.get("fast") and hasattr(cz_pipeline, "set_edit_speed"):
                cz_pipeline.set_edit_speed(spec["fast"])
            img = cz_pipeline.generate_omni(
                imgs, spec["prompt"], spec.get("negative", ""),
                spec["width"], spec["height"], steps, seed_used, **extra)
        finally:
            for im in imgs:
                try:
                    im.close()
                except Exception:
                    pass
        timings = {"omni": time.time() - t_o}
    else:
        img, timings = cz_pipeline.txt2img_run(
            spec["prompt"], spec["width"], spec["height"], steps,
            seed_used, spec.get("negative", ""))
    # Detailer pass (ADetailer-style faces, YOLO hands) - the same one as the
    # CLI's --detail-faces/--detail-hands. A failure (optional package missing,
    # detector down) DEGRADES to a warning, never to a lost panel.
    timings = dict(timings or {})
    nf = nh = 0
    df, dh = spec.get("detail_faces"), spec.get("detail_hands")
    if df or dh or df is None:
        try:
            import cz_detailer
            if df is None:
                df = bool(getattr(cz_detailer, "DETAILER_ENABLED", False))
            if df:
                t_d = time.time()
                img, nf = cz_detailer.detail_faces(img, spec["prompt"],
                                                   seed_used, steps)
                timings["detail_faces"] = time.time() - t_d
            if dh:
                t_d = time.time()
                img, nh = cz_detailer.detail_hands(img, spec["prompt"],
                                                   seed_used, steps)
                timings["detail_hands"] = time.time() - t_d
        except Exception as e:
            warnings.append(f"detailer skipped: {e}")
    path = build_output_path(None, "local", spec.get("out_dir"), "png",
                             tag=("czp_omni" if refs else "czp_txt2img"),
                             seed=seed_used, size=img.size)
    save_image(img, path, "png",
               meta={"mode": "omni" if refs else "txt2img",
                     "prompt": spec["prompt"],
                     "negative": spec.get("negative", ""), "seed": seed_used,
                     "steps": steps, "size": list(img.size),
                     "loras": spec.get("loras") or [],
                     "refs": [os.path.basename(r) for r in refs]})
    return {"ok": True, "protocol": PROTOCOL, "tool": TOOL,
            "version": APP_VERSION, "route": route,
            "images": [os.path.abspath(path)], "seed_used": seed_used,
            "loras": list(spec.get("loras") or []),
            "refs_used": len(refs),
            "faces_refined": nf, "hands_refined": nh,
            "timings": {"total_s": round(time.time() - t0, 2),
                        **{k: round(v, 2) for k, v in (timings or {}).items()}},
            "warnings": warnings}


def _pick_esrgan(models, factor):
    """Default ESRGAN model for a factor when the spec names none: the config
    default if present, else the first model whose name announces the closest
    scale (4x for factor 4, 2x for 2...), else the first 4x/2x, else the first
    one - never a 16x by accident."""
    if not models:
        return None
    cfg = str(CONFIG.get("default_esrgan_model") or "").strip()
    if cfg in models:
        return cfg
    want = max(2, min(8, int(round(factor))))
    for scale in (want, 4, 2, 8):
        for m in models:
            if m.lower().startswith(f"{scale}x"):
                return m
    return models[0]


def run_upscale(spec, warnings=None, route="local"):
    """Upscale / img2img ONE image, with spec.guidance and spec.loras (if any)
    for this call only. See _run_upscale."""
    warnings = list(warnings or [])
    with _call_guidance(spec, warnings) as g, _keep_app_loras():
        res = _run_upscale(spec, warnings, route)
    if g is not None:
        res["guidance"] = g
    return res


def _run_upscale(spec, warnings=None, route="local"):
    """Upscale ONE image (ESRGAN + refine, the tool's pipeline) from a VALID
    spec. The family's print output: comic panels generated at ~1 MP come up
    to print resolution through here. The prompt (optional) guides the refine
    pass - NEVER the scene prompt on a crop, house rule: the caller sends a
    LOCAL description or nothing."""
    import cz_pipeline
    import cz_esrgan
    from PIL import Image
    from cz_imageio import build_output_path, save_image

    warnings = list(warnings or [])
    t0 = time.time()
    factor = float(spec.get("factor", 2.0) or 2.0)
    # factor <= 1 = pure img2img (variation): NO ESRGAN stage at all. Picking
    # the first model of the folder for a 1x pass once ran a 16x upscaler on a
    # 1280x832 panel (10+ min, 32 GB VRAM) just to shrink it back.
    do_esrgan = factor > 1.0
    models = cz_esrgan.list_esrgan_models() if do_esrgan else []
    model = None
    if do_esrgan:
        if spec.get("model"):
            model = spec["model"] if spec["model"] in models else None
            if model is None:
                warnings.append(f"ESRGAN model '{spec['model']}' not found - "
                                f"using {_pick_esrgan(models, factor) or 'none'}")
        if model is None:
            model = _pick_esrgan(models, factor)
        if model is None:
            raise RuntimeError(f"no ESRGAN model in {cz_esrgan.ESRGAN_DIR}")
    denoise = spec.get("denoise")
    if denoise is None:
        denoise = float(CONFIG.get("default_denoise", 0.30))
    steps = spec.get("steps") or int(CONFIG.get("default_refine_steps",
                                                CONFIG.get("default_steps",
                                                           12)))
    # Seed concrete (the refine pass draws from it) + {a|b|c} / wildcards
    # bound to it; returned as seed_used so a variation can be replayed.
    from cz_prompt import resolve_seed
    seed_used = resolve_seed(spec.get("seed", -1))
    spec = _expand_spec(spec, seed_used)
    _apply_spec_loras(spec)           # the refine pass runs on the base pipe
    with Image.open(spec["input"]) as im:
        iw, ih = im.size
        # A variation (factor <= 1) is an img2img at the image's own size: it runs on
        # the WHOLE image with the requested denoise and prompt, up to the size MOSS
        # renders in one pass. The tile rules (per-tile denoise cap, empty prompt per
        # tile) exist for upscales to 4K+; applied to a 1088x1920 variation they made
        # every denoise 0.40 and dropped the prompt, in silence (crispz-motion lot 2).
        whole = not do_esrgan and iw * ih <= VARIATION_WHOLE_MAX_PX
        kw = {"auto_tile": False, "refine_tile": 0} if whole else {}
        rw, rh = (iw, ih) if not do_esrgan else (round(iw * factor), round(ih * factor))
        notes = getattr(cz_pipeline, "refine_tiling_notes", None)
        if notes is not None and not whole:
            warnings += notes(rw, rh, denoise, spec.get("prompt", ""))
        img, timings = cz_pipeline.process_one(
            im.convert("RGB"), model, factor, denoise,
            steps, spec.get("prompt", ""), seed_used,
            int(CONFIG.get("default_tile", 0)),
            int(CONFIG.get("default_overlap", 16)), do_esrgan=do_esrgan, **kw)
    path = build_output_path(None, "local", spec.get("out_dir"), "png",
                             tag="czp_upscale", seed=seed_used,
                             size=img.size)
    save_image(img, path, "png",
               meta={"mode": "upscale", "source": spec["input"],
                     "factor": spec.get("factor"), "denoise": denoise,
                     "model": model, "size": list(img.size)})
    return {"ok": True, "protocol": PROTOCOL, "tool": TOOL,
            "version": APP_VERSION, "route": route,
            "images": [os.path.abspath(path)], "seed_used": seed_used,
            "size": list(img.size), "esrgan_model": model,
            "timings": {"total_s": round(time.time() - t0, 2),
                        **{k: round(v, 2) for k, v in (timings or {}).items()
                           if isinstance(v, (int, float))}},
            "warnings": warnings}


def handle_upscale_json(spec_json):
    """INSTANCE side (endpoint api_name='cli_upscale')."""
    try:
        spec, warnings = validate_spec(json.loads(spec_json or "{}"),
                                       op="upscale")
        return run_upscale(spec, warnings, route="remote")
    except SpecError as e:
        return {"ok": False, "protocol": PROTOCOL, "tool": TOOL,
                "error": str(e), "exit_code": e.code}
    except Exception as e:
        return {"ok": False, "protocol": PROTOCOL, "tool": TOOL,
                "error": f"{type(e).__name__}: {e}", "exit_code": 1}


def handle_edit_json(spec_json):
    """INSTANCE side (endpoint api_name='cli_edit'): image + instruction ->
    image through the omni/edit pipeline (refs = [input])."""
    try:
        spec, warnings = validate_spec(json.loads(spec_json or "{}"),
                                       op="edit")
        res = run_gen(spec, warnings, route="remote")
        res["op"] = "edit"
        return res
    except SpecError as e:
        return {"ok": False, "protocol": PROTOCOL, "tool": TOOL,
                "error": str(e), "exit_code": e.code}
    except Exception as e:
        return {"ok": False, "protocol": PROTOCOL, "tool": TOOL,
                "error": f"{type(e).__name__}: {e}", "exit_code": 1}


def run_inpaint(spec, warnings=None, route="local"):
    """Inpaint ONE image, with spec.guidance and spec.loras (if any) for this
    call only. See _run_inpaint."""
    warnings = list(warnings or [])
    with _call_guidance(spec, warnings) as g, _keep_app_loras():
        res = _run_inpaint(spec, warnings, route)
    if g is not None:
        res["guidance"] = g
    return res


def _run_inpaint(spec, warnings=None, route="local"):
    """Inpaint ONE image from a VALID spec: the WHITE area of the mask is
    redrawn from the prompt (the tool's inpaint pipeline, the same code as the
    Inpaint tab). The prompt describes what must appear IN the area (a local
    description - house rule: never the scene prompt on a fragment). Outside
    the mask: original pixels, feathered seam."""
    import random
    import cz_pipeline
    from PIL import Image
    from cz_imageio import build_output_path, save_image

    warnings = list(warnings or [])
    t0 = time.time()
    denoise = spec.get("denoise")
    if denoise is None:
        denoise = float(CONFIG.get("default_inpaint_strength", 0.9))
    steps = spec.get("steps") or int(CONFIG.get("default_refine_steps",
                                                CONFIG.get("default_steps",
                                                           12)))
    seed_used = int(spec.get("seed", -1))
    if seed_used < 0:
        seed_used = random.randint(0, 2**31 - 1)
    cz_pipeline._LAST_SEED = seed_used
    spec = _expand_spec(spec, seed_used)        # {a|b|c} + wildcards, then LoRA tags
    _apply_spec_loras(spec)
    with Image.open(spec["input"]) as im, Image.open(spec["mask"]) as mk:
        bg = im.convert("RGB")
        mask = mk.convert("L")
        if mask.size != bg.size:
            warnings.append(f"mask {mask.size[0]}x{mask.size[1]} resized to "
                            f"the image {bg.size[0]}x{bg.size[1]}")
            mask = mask.resize(bg.size, Image.NEAREST)
        if mask.getbbox() is None:
            raise SpecError("mask is empty (nothing painted white): nothing "
                            "to inpaint")
        img = cz_pipeline.inpaint_run(bg, mask, spec.get("prompt", ""),
                                      int(steps), float(denoise), seed_used)
    path = build_output_path(None, "local", spec.get("out_dir"), "png",
                             tag="czp_inpaint", seed=seed_used, size=img.size)
    save_image(img, path, "png",
               meta={"mode": "inpaint", "source": spec["input"],
                     "mask": spec["mask"], "prompt": spec.get("prompt", ""),
                     "denoise": denoise, "steps": int(steps),
                     "seed": seed_used, "size": list(img.size)})
    return {"ok": True, "protocol": PROTOCOL, "tool": TOOL,
            "version": APP_VERSION, "route": route, "op": "inpaint",
            "images": [os.path.abspath(path)], "seed_used": seed_used,
            "size": list(img.size), "denoise": denoise, "steps": int(steps),
            "timings": {"total_s": round(time.time() - t0, 2)},
            "warnings": warnings}


def handle_inpaint_json(spec_json):
    """INSTANCE side (endpoint api_name='cli_inpaint'): image + mask +
    prompt -> image (the white area is redrawn)."""
    try:
        spec, warnings = validate_spec(json.loads(spec_json or "{}"),
                                       op="inpaint")
        return run_inpaint(spec, warnings, route="remote")
    except SpecError as e:
        return {"ok": False, "protocol": PROTOCOL, "tool": TOOL,
                "error": str(e), "exit_code": e.code}
    except Exception as e:
        return {"ok": False, "protocol": PROTOCOL, "tool": TOOL,
                "error": f"{type(e).__name__}: {e}", "exit_code": 1}


def handle_gen_json(spec_json):
    """INSTANCE side (api_name='cli_gen' endpoint): validate + generate.
    Always returns a dict {ok: ...} - the czp caller has a single error
    path."""
    try:
        spec, warnings = validate_spec(json.loads(spec_json or "{}"))
        if spec.get("model"):
            warnings.append("model override ignored on the remote route (the "
                            "running instance keeps its model)")
            spec["model"] = None
        return run_gen(spec, warnings, route="remote")
    except SpecError as e:
        return {"ok": False, "protocol": PROTOCOL, "tool": TOOL,
                "error": str(e), "exit_code": e.code}
    except Exception as e:
        return {"ok": False, "protocol": PROTOCOL, "tool": TOOL,
                "error": f"{type(e).__name__}: {e}", "exit_code": 1}


# ----------------------------------------------------------------------------
# Remote route (direct HTTP on the Gradio endpoints, like the SPAs)
# ----------------------------------------------------------------------------
def _gradio_call(url, name, data, post_timeout=10, get_timeout=3600):
    """POST /gradio_api/call/<name> -> event_id, then GET the stream and
    extract the first output. Returns the string, or raises
    urllib.error/ValueError."""
    req = urllib.request.Request(
        f"{url.rstrip('/')}/gradio_api/call/{name}",
        data=json.dumps({"data": data}).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=post_timeout) as r:
        j = json.loads(r.read().decode("utf-8"))
    eid = j.get("event_id") or j.get("hash")
    if not eid:
        raise ValueError("no event id (endpoint missing? app too old?)")
    with urllib.request.urlopen(
            f"{url.rstrip('/')}/gradio_api/call/{name}/{eid}",
            timeout=get_timeout) as r:
        text = r.read().decode("utf-8")
    m = re.search(r"data:\s*(\[[\s\S]*?\])\s*(?:\n|$)", text)
    if not m:
        raise ValueError("empty API response")
    return json.loads(m.group(1))[0]


def probe_instance(url, timeout=4):
    """caps of the instance running at `url`, or None if nothing answers.
    Short: never delays a local call by more than a few seconds."""
    try:
        raw = _gradio_call(url, "cli_caps", [], post_timeout=timeout,
                           get_timeout=max(timeout, 8))
        caps = json.loads(raw) if isinstance(raw, str) else raw
        return caps if isinstance(caps, dict) and caps.get("ok") else None
    except Exception:
        return None


# ----------------------------------------------------------------------------
# CLI entry (czp.bat / python cz_protocol.py)
# ----------------------------------------------------------------------------
def _emit(payload, code):
    print(json.dumps(payload, ensure_ascii=False))
    return code


def _read_spec(path):
    # utf-8-sig: PowerShell 5.1 (Set-Content/Out-File -Encoding utf8) writes
    # a BOM - refusing it would break the first test anyone runs on Windows.
    # The extra strip covers a BOM arriving through stdin.
    raw = sys.stdin.read() if path == "-" \
        else open(path, encoding="utf-8-sig").read()
    return json.loads(raw.lstrip("\ufeff"))


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="czp", description="crispz family CLI protocol v1 (JSON in/out)")
    parser.add_argument("op", choices=list(PROTOCOL_OPS))
    parser.add_argument("--spec", metavar="FILE",
                        help="spec JSON ('-' = stdin); required for gen")
    parser.add_argument("--local", action="store_true",
                        help="force local execution (night batch), never route")
    parser.add_argument("--remote", metavar="URL",
                        help="force this instance (error 4 if unreachable)")
    args = parser.parse_args(argv)

    if args.op == "caps":
        caps = caps_dict()
        url = args.remote or instance_url()
        inst = probe_instance(url)
        caps["instance"] = ({"running": True, "url": url,
                             "tool": inst.get("tool"),
                             "version": inst.get("version")} if inst
                            else {"running": False, "url": url})
        return _emit(caps, 0)

    if args.op not in OPS:
        return _emit({"ok": False, "tool": TOOL,
                      "error": f"op '{args.op}' not supported by {TOOL} "
                               f"(v1: {', '.join(OPS)})"}, 3)

    if not args.spec:
        return _emit({"ok": False, "tool": TOOL,
                      "error": "gen requires --spec <file|->"}, 2)
    try:
        spec, warnings = validate_spec(_read_spec(args.spec), op=args.op)
    except SpecError as e:
        return _emit({"ok": False, "tool": TOOL, "error": str(e)}, e.code)
    except Exception as e:
        return _emit({"ok": False, "tool": TOOL,
                      "error": f"unreadable spec: {e}"}, 2)

    # Routing: remote when an instance answers (its queue serializes the
    # GPU), local otherwise. --local / --remote bypass the detection.
    endpoint = {"upscale": "cli_upscale", "edit": "cli_edit",
                "inpaint": "cli_inpaint"}.get(
        args.op, "cli_gen")
    if not args.local:
        url = args.remote or instance_url()
        inst = probe_instance(url)
        if inst:
            if args.op == "gen" and spec.get("model"):
                # (upscale: spec.model = the ESRGAN model, harmless remote)
                warnings.append("model override ignored on the remote route "
                                "(the running instance keeps its model)")
                spec["model"] = None
            try:
                raw = _gradio_call(url, endpoint, [json.dumps(spec)])
                res = json.loads(raw) if isinstance(raw, str) else raw
            except Exception as e:
                return _emit({"ok": False, "tool": TOOL,
                              "error": f"remote call failed: {e}"}, 1)
            res.setdefault("warnings", [])
            res["warnings"] = warnings + [w for w in res["warnings"]
                                          if w not in warnings]
            return _emit(res, 0 if res.get("ok") else
                         int(res.get("exit_code", 1)))
        if args.remote:
            return _emit({"ok": False, "tool": TOOL,
                          "error": f"no instance at {args.remote}"}, 4)

    try:
        runner = {"upscale": run_upscale,
                  "inpaint": run_inpaint}.get(args.op, run_gen)
        res = runner(spec, warnings)
        if args.op == "edit":
            res["op"] = "edit"
        return _emit(res, 0)
    except Exception as e:
        return _emit({"ok": False, "tool": TOOL,
                      "error": f"{type(e).__name__}: {e}"}, 1)


if __name__ == "__main__":
    sys.exit(main())
