"""crispz-moss - magic prompt: a plain idea -> a MOSS-Image layout draft (lot 2).

Two engines, both on this machine (`magic_prompt_engine`):

  ollama   (default) a local Ollama model (`magic_prompt_ollama_model`, else the Improve
           model, else the first installed) with a system prompt adapted from the official
           MOSS-Image prompt-enhancer skill, and the draft schema passed as Ollama's
           structured-output `format` (Ollama >= 0.5), so the answer is JSON of the right
           shape. Nothing leaves the machine (the Ollama URL is logged on every call).
  local    the Qwen3.5-9B text encoder already loaded for MOSS-Image, made generative
           with its own LM head (downloaded once from Qwen/Qwen3.5-9B, ~2 GB, the one
           tensor only). No second model in VRAM. JSON constrained by `outlines` when
           installed, said once when it is not.

Every answer goes through the same cleaning: thinking blocks and code fences dropped,
the outer {...} kept, a draft rebuilt at the asked size (boxes clamped to the canvas,
labels normalized), then cz_caption.validate. Returns (draft_text, warnings) where
draft_text is the one-line DRAFT the prompt box takes (cz_caption.to_prompt).

`describe` does the same from an IMAGE with the local engine only (the loaded Qwen3.5's
vision tower reads the image): summary, meta, every visible text verbatim and a pixel
box per element.
"""

import json
import re

import cz_caption
from cz_core import CONFIG, _log, _prefs

ENGINES = ("ollama", "local")
ENGINE_LABELS = {"ollama": "Ollama (local LLM)",
                 "local": "Local (the loaded Qwen3.5-9B + its LM head, ~2 GB once)"}


class MagicPromptError(RuntimeError):
    """A magic prompt that could not produce a layout, with a message to show as is.
    `raw` keeps the model's whole answer for the log and the bench reports."""

    def __init__(self, message, raw=None):
        super().__init__(message)
        self.raw = raw


def _setting(key, default=None):
    """The UI preference when set, else `default`, which every caller passes as
    CONFIG.get("<key>", ...) (literal, so tools/check_config_keys.py sees the read)."""
    v = _prefs.get(key)
    return default if v is None else v


def engine():
    e = _setting("magic_prompt_engine", CONFIG.get("magic_prompt_engine", "ollama"))
    e = str(e or "ollama").strip().lower()
    return e if e in ENGINES else "ollama"


def visible_engines():
    return ENGINES


# --- The brief the LLM gets (adapted from skills/moss-image-prompt-enhancer, Apache 2.0) -
SYSTEM = """You are a prompt enhancer for MOSS-Image, a text-to-image model that renders text exactly and places elements where a layout says. Turn the user's idea into a LAYOUT DRAFT: one JSON object and nothing else.

Schema (every field below, no other):
{"width": W, "height": H, "bbox_coord_space": "abs", "bbox_format": "xyxy",
 "summary": {"meta": ["poster"], "canvas": "single", "caption": "...", "palette": ["#RRGGBB"]},
 "items": [{"label": "text", "caption": "...", "text": "EXACT WORDS", "bbox": [x1, y1, x2, y2]},
           {"label": "object", "caption": "...", "bbox": [x1, y1, x2, y2]}]}

Rules:
- width and height are the ones given; bbox = [x_min, y_min, x_max, y_max] in PIXELS from the top-left, inside the canvas, x_min < x_max, y_min < y_max. Never normalized values.
- summary.caption: cohesive prose describing the whole image: subject, setting, medium, style, lighting, materials, palette, mood, and the global typographic treatment. Say "no other text" when the design should carry only the listed texts.
- summary.meta: one or two genre tags (poster, logo, business card, book cover, slide, packaging, thumbnail, social media, banner, product campaign, illustration...). summary.canvas is "single" unless the idea asks for panels or a collage.
- items: one per coherent visible unit. label is "text", "object" or "character". Each caption describes that element's appearance; for a text item also its typography (weight, case, colour, alignment). A text item's "text" holds the exact visible characters, in the user's language, punctuation and capitalization kept, "\\n" for an intended line break. Write the real copy the design needs, never placeholders.
- Preserve the user's subjects, counts, wording and exclusions. Do not invent event dates, prices or addresses the user did not give. Keep boxes large enough for the text to be readable at the canvas size, and leave margins.
- Describe in English unless the user writes in another language for the visible text; the visible text stays in its language.
Output the JSON object only: no markdown fence, no explanation."""

USER_TEMPLATE = ("Canvas: {width} x {height} pixels (aspect ratio {aspect}).\n"
                 "Idea: {idea}")

# The shape Ollama is asked to produce (its structured-output `format`).
OLLAMA_FORMAT = {
    "type": "object",
    "properties": {
        "width": {"type": "integer"}, "height": {"type": "integer"},
        "bbox_coord_space": {"type": "string", "enum": ["abs"]},
        "bbox_format": {"type": "string", "enum": ["xyxy"]},
        "summary": {"type": "object",
                    "properties": {"meta": {"type": "array", "items": {"type": "string"}},
                                   "canvas": {"type": "string",
                                              "enum": list(cz_caption.CANVASES)},
                                   "caption": {"type": "string"},
                                   "palette": {"type": "array", "items": {"type": "string"}}},
                    "required": ["caption"]},
        "items": {"type": "array", "minItems": 1, "items": {
            "type": "object",
            "properties": {"label": {"type": "string", "enum": list(cz_caption.LABELS)},
                           "caption": {"type": "string"},
                           "text": {"type": "string"},
                           "bbox": {"type": "array", "minItems": 4, "maxItems": 4,
                                    "items": {"type": "number"}}},
            "required": ["label", "caption", "bbox"]}}},
    "required": ["summary", "items"]}


# --- Cleaning ------------------------------------------------------------------------
_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```$")


def _repair_json(text):
    """A best-effort repair of the JSON an LLM returns: trailing commas dropped, an
    answer cut short (token budget, or a model that stopped mid-way) closed: the open
    string, a dangling key, then the open arrays and objects. Returns the repaired text
    (to parse again), or None when nothing changed."""
    s = re.sub(r",\s*([\]}])", r"\1", text.rstrip())
    stack, in_str, esc = [], False, False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "[{":
            stack.append("]" if ch == "[" else "}")
        elif ch in "]}" and stack:
            stack.pop()
    if in_str:
        s += '"'
    # a key without its value (`"bbox":` or `"bbox"`) cannot be closed: dropped
    s = re.sub(r'(,|\{)\s*"(?:[^"\\]|\\.)*"\s*(:\s*)?$', r"\1", s)
    s = re.sub(r",\s*$", "", s)
    s += "".join(reversed(stack))
    return s if s != text else None


def clean(raw, width, height, validate=True):
    """Raw model text -> (draft_text, warnings). The JSON object is read out of the
    answer (thinking, fences and prose around it dropped), the boxes are taken as
    PIXELS of a width x height canvas (an answer that drew them for another canvas, or
    in 0..1000, is rescaled to fit: said in the warnings), labels and keys normalized.
    Raises MagicPromptError when no JSON object can be read."""
    import prompt_improve
    text = prompt_improve.strip_thinking(raw or "").strip()
    text = _FENCE_RE.sub("", text).strip()
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        raise MagicPromptError(f"the magic prompt returned no JSON object: {text[:160]!r}",
                               raw=raw)
    repaired = None
    try:
        d = cz_caption.parse(text[a:b + 1])
    except ValueError as e:
        # A cut answer has no closing brace to trust: repaired from the first one on.
        fixed = _repair_json(text[a:])
        try:
            d = cz_caption.parse(fixed) if fixed else None
        except ValueError:
            d = None
        if d is None:
            _log(f"magic prompt: invalid JSON ({e}); the whole answer "
                 f"({len(text)} chars): {text[:2000]!r}")
            raise MagicPromptError(f"the magic prompt returned invalid JSON ({e}): "
                                   f"{text[a:a + 160]!r}", raw=raw)
        repaired = str(e)
    if "items" not in d:
        raise MagicPromptError("the magic prompt returned no 'items': "
                               f"{text[a:a + 160]!r}", raw=raw)
    d.setdefault("summary", {"caption": ""})
    w, h = int(width), int(height)
    warnings = []
    if repaired:
        warnings.append(f"JSON repaired (the answer was cut or malformed: {repaired}); "
                        f"check the last items")
    # The canvas the answer drew for: the one asked, unless it says otherwise or its
    # boxes overflow (an LLM that ignored the size, or answered in 0..1000).
    try:
        dw, dh = int(d.get("width") or w), int(d.get("height") or h)
    except (TypeError, ValueError):
        dw, dh = w, h
    boxes = [it.get("bbox") for it in d["items"] if isinstance(it, dict)
             and isinstance(it.get("bbox"), list) and len(it["bbox"]) == 4]
    mx = max((float(v) for bx in boxes for v in bx[0::2] if isinstance(v, (int, float))),
             default=0)
    my = max((float(v) for bx in boxes for v in bx[1::2] if isinstance(v, (int, float))),
             default=0)
    if d.get("bbox_coord_space") == "abs_centralized":
        lay, _, _ = cz_caption.from_text(d, w, h)
        warnings.append("the answer came compiled (centred boxes): read back as such")
    else:
        if (dw, dh) != (w, h):
            warnings.append(f"the answer drew its boxes for {dw}x{dh}: rescaled to {w}x{h}")
        elif boxes and "width" not in d and mx <= 1000 and my <= 1000 and (mx > w or my > h):
            dw, dh = 1000, 1000
            warnings.append("the answer used 0..1000 boxes: read as thousandths")
        d["width"], d["height"] = dw, dh
        d["bbox_coord_space"], d["bbox_format"] = "abs", "xyxy"
        lay, _, _ = cz_caption.from_text(d, dw, dh)
    text_out = cz_caption.to_prompt(lay, w, h)
    return text_out, warnings + (cz_caption.validate(text_out) if validate else [])


# --- Engines -------------------------------------------------------------------------
def _ollama(idea, width, height, seed, model=None):
    import cz_ollama
    import prompt_improve
    aspect = cz_caption.aspect_ratio(width, height)
    user = USER_TEMPLATE.format(width=int(width), height=int(height), aspect=aspect, idea=idea)
    model = model or _setting("magic_prompt_ollama_model",
                              CONFIG.get("magic_prompt_ollama_model", "")) or None
    base = cz_ollama._improve_base()
    if not model:
        models = prompt_improve.list_models(base=base)
        if not models:
            raise MagicPromptError("Ollama: no model installed (ollama pull <model>)")
        model = models[0]
    payload = {"model": model, "system": SYSTEM, "prompt": user,
               **cz_ollama._ollama_gen_opts(temperature=_setting(
                   "magic_prompt_temperature", CONFIG.get("magic_prompt_temperature", 0.7)))}
    opts = payload.setdefault("options", {})
    opts["num_predict"] = max(int(opts.get("num_predict") or 0), 2048)
    if seed is not None and int(seed) >= 0:
        opts["seed"] = int(seed)
    payload["format"] = OLLAMA_FORMAT
    _log(f"magic prompt (ollama, {model}, {base}): sending the idea, {int(width)}x{int(height)}")
    try:
        out = prompt_improve.http("/api/generate", payload, base=base, timeout=300)
    except Exception as e:
        raise MagicPromptError(f"Ollama ({model}): {e}")
    return out.get("response") or ""


def _local(idea, width, height, seed):
    import cz_pipeline
    return cz_pipeline.magic_prompt_local(idea, width, height, seed=seed)


def messages_for(idea, width, height):
    """(system, user) for the local engine (the same brief as Ollama's)."""
    aspect = cz_caption.aspect_ratio(width, height)
    return SYSTEM, USER_TEMPLATE.format(width=int(width), height=int(height),
                                        aspect=aspect, idea=idea)


def expand(idea, width, height, seed=None, which=None, model=None):
    """A plain idea -> (draft_text, warnings, engine_used). Raises MagicPromptError."""
    idea = (idea or "").strip()
    if not idea:
        raise MagicPromptError("nothing to expand: the prompt is empty")
    if cz_caption.is_caption(idea):
        raise MagicPromptError("the prompt is already a layout JSON")
    which = which or engine()
    if which == "local":
        raw = _local(idea, width, height, seed)
    else:
        which = "ollama"
        raw = _ollama(idea, width, height, seed, model=model)
    text, warnings = clean(raw, width, height)
    if warnings:
        _log(f"magic prompt ({which}): {len(warnings)} note(s): " + "; ".join(warnings[:3]))
    return text, warnings, which


DESCRIBE_SYSTEM = """You describe an image as a MOSS-Image LAYOUT DRAFT that a renderer could use to recreate it. Output ONE JSON object and nothing else:
{"summary": {"meta": [...], "canvas": "single", "caption": "...", "palette": ["#RRGGBB"]},
 "items": [{"label": "text", "caption": "...", "text": "VERBATIM", "bbox": [x1, y1, x2, y2]},
           {"label": "object", "caption": "...", "bbox": [x1, y1, x2, y2]}]}
Rules: bbox in PIXELS of the image ({width} x {height}), [x_min, y_min, x_max, y_max] from the top-left. Transcribe EVERY visible text exactly, one text item per block, line breaks as \\n, and describe its typography in the caption. One object or character item per main visible thing, 10 to 40 words each, concrete (shape, colour, material, position). summary.caption describes the whole image: subject, medium, style, lighting, palette, mood. palette: 3 to 8 dominant colours as #RRGGBB. Valid JSON only: double quotes, no trailing commas, no markdown."""


DESCRIBE_USER = "Describe this image as the layout draft."


def ollama_vision_model(model=None, base=None):
    """The Ollama model that will read an image: the magic prompt's model when Ollama
    reports the 'vision' capability for it, else the first vision model installed, else
    None (the caller falls back on the local engine and says so)."""
    import cz_ollama
    base = base or cz_ollama._improve_base()
    model = model or _setting("magic_prompt_ollama_model",
                              CONFIG.get("magic_prompt_ollama_model", "")) or None
    if model:
        try:
            info = cz_ollama._ollama_http("/api/show", {"model": model}, base=base, timeout=8)
            if "vision" in [c.lower() for c in (info.get("capabilities") or [])]:
                return model
        except Exception as e:
            _log(f"describe (ollama): /api/show failed for {model} ({e})")
    try:
        models = cz_ollama._ollama_vision_models(base=base)
    except Exception as e:
        _log(f"describe (ollama): cannot list the models at {base} ({e})")
        return None
    return models[0] if models else None


def _ollama_describe(image, model, base=None):
    """The image (JPEG, 1024 px max, base64) and the Describe brief to a local Ollama
    vision model, the draft schema as structured output; the raw answer."""
    import cz_ollama
    from cz_core import _pil_to_b64_jpeg
    base = base or cz_ollama._improve_base()
    system = (DESCRIBE_SYSTEM.replace("{width}", str(image.width))
              .replace("{height}", str(image.height)))
    payload = {"model": model, "system": system, "prompt": DESCRIBE_USER,
               "images": [_pil_to_b64_jpeg(image.convert("RGB"), max_side=1024)],
               **cz_ollama._ollama_gen_opts(temperature=0.0)}
    opts = payload.setdefault("options", {})
    opts["num_predict"] = max(int(opts.get("num_predict") or 0), 2048)
    payload["format"] = OLLAMA_FORMAT
    _log(f"describe (ollama, {model}, {base}): sending the image, "
         f"{image.width}x{image.height}")
    try:
        import prompt_improve
        out = prompt_improve.http("/api/generate", payload, base=base, timeout=300)
    except Exception as e:
        raise MagicPromptError(f"Ollama ({model}): {e}")
    return out.get("response") or ""


def describe(image, which=None):
    """An image -> (draft_text, warnings, engine_used), on this machine: through an
    Ollama VISION model when the engine is Ollama and one is installed (seconds), else
    with the loaded Qwen3.5-9B (vision tower and LM head; a minute or so, nothing else
    to install). The draft's canvas is the image's size on the 16 grid."""
    if image is None:
        raise MagicPromptError("no image to describe")
    which = (which or engine()) if which != "local" else "local"
    if which == "ollama":
        model = ollama_vision_model()
        if model:
            raw = _ollama_describe(image, model)
        else:
            which = "local"
            _log("describe: no Ollama vision model found (set magic_prompt_ollama_model to "
                 "one with the 'vision' capability); describing with the local Qwen3.5-9B")
    if which == "local":
        import cz_pipeline
        raw = cz_pipeline.describe_local(image)
    # The boxes come in pixels of the image as shown to the model; the draft is written
    # on the 16 grid (what MOSS renders), the boxes rescaled with it.
    text, warnings = clean(raw, image.width, image.height, validate=False)
    w, h = cz_caption.fit_grid(image.width, image.height)
    if (w, h) != (image.width, image.height):
        lay, _, _ = cz_caption.from_text(text)
        text = cz_caption.to_prompt(lay, w, h)
    warnings += cz_caption.validate(text)
    if warnings:
        _log(f"describe ({which}): {len(warnings)} note(s): " + "; ".join(warnings[:3]))
    return text, warnings, which
