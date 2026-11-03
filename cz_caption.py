"""crispz-moss - structured layout prompts for MOSS-Image (lot 2).

MOSS-Image takes three prompt forms: plain text, tag sequences, and a LAYOUT JSON that
names the scene, every visible text VERBATIM and where each thing goes. The A/B of lot 1
showed why the layout matters beyond placement: in plain text the model may add words it
infers from the prompt ("COFFEE SHOP" on a coffee logo, misspelt), in a layout it renders
the listed texts and nothing else.

Three shapes of the same layout live here (USERGUIDE.md and the official prompt-enhancer
skill, skills/moss-image-prompt-enhancer/references/layout.md):

  EDITOR (this module's dict, what the form and the board handle):
    {"summary": {"caption", "meta": [...], "canvas", "palette": [...]},
     "items": [{"label": "text" | "object" | "character", "caption", "text"?, "bbox"?,
                "palette"?, "tags"?}]}
    with bbox = [y_min, x_min, y_max, x_max] in 0..1000 of the canvas (resolution-free:
    a preset or a box drawn on the board does not depend on the size yet).

  DRAFT (the skill's readable form, what "Use as prompt" puts in the prompt box):
    the same plus "width", "height", "bbox_coord_space": "abs", "bbox_format": "xyxy",
    bbox = [x1, y1, x2, y2] in PIXELS from the top-left. A draft carries its canvas, so
    the engine (cz_moss) compiles it and renders at THAT size: a layout is drawn for one
    size, and the size follows the layout, never the opposite.

  COMPILED (what the model reads, built by compile()):
    summary + items, bbox = [y_min, x_min, y_max, x_max] in pixels from the canvas
    CENTRE (y down), "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx", one
    minified UTF-8 line, no canvas metadata. Same arithmetic as the skill's
    compile_prompt.py (floor the near edges, ceil the far ones, then subtract W/2, H/2).

A compiled layout pasted by hand is accepted as a prompt too (it goes through as is).
Validation is this module's own (the model publishes no verifier): the checks of the
official compiler plus a few of ours. No torch, no gradio.
"""

import json
import math
import re

BBOX_SCALE = 1000
GRID = 16
LABELS = ("text", "object", "character")
CANVASES = ("single", "multi_panel", "collage")
# Common meta tags (the USERGUIDE's examples are "poster"; free text is accepted).
METAS = ("poster", "logo", "business card", "book cover", "slide", "packaging",
         "thumbnail", "social media", "banner", "product campaign", "studio photography",
         "illustration", "infographic")
SUMMARY_KEYS = ("meta", "canvas", "caption", "palette")
ITEM_KEYS = ("label", "caption", "text", "bbox", "tags", "palette")

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


# --- Text forms ----------------------------------------------------------------------
def to_prompt(layout, width, height):
    """The DRAFT as one minified UTF-8 line: what goes in the prompt box (the engine
    compiles it at render time and renders at width x height)."""
    return json.dumps(to_draft(layout, width, height), separators=(",", ":"),
                      ensure_ascii=False)


def to_pretty(layout, width, height):
    """The draft, indented, for the raw-JSON editor."""
    return json.dumps(to_draft(layout, width, height), indent=2, ensure_ascii=False)


def to_draft(layout, width, height):
    """EDITOR dict -> DRAFT dict (pixel boxes [x1, y1, x2, y2], the canvas written in)."""
    w, h = int(width), int(height)
    out = {"width": w, "height": h, "bbox_coord_space": "abs", "bbox_format": "xyxy",
           "summary": _summary_of(layout), "items": []}
    for it in layout.get("items") or []:
        d = _item_of(it)
        b = parse_bbox(it.get("bbox"))
        if b:
            x0, y0, x1, y1 = bbox_to_pixels(b, w, h)
            d["bbox"] = [x0, y0, x1, y1]
        out["items"].append(d)
    return out


def _summary_of(layout):
    s = layout.get("summary") or {}
    out = {}
    meta = [str(m).strip() for m in (s.get("meta") or []) if str(m).strip()]
    if meta:
        out["meta"] = meta
    canvas = str(s.get("canvas") or "").strip()
    if canvas:
        out["canvas"] = canvas
    out["caption"] = str(s.get("caption") or "").strip()
    pal, _ = parse_palette(s.get("palette") or [], 16)
    if pal:
        out["palette"] = pal
    return out


def _item_of(it):
    label = str(it.get("label") or "object").strip().lower()
    if label == "obj":
        label = "object"
    d = {"label": label, "caption": str(it.get("caption") or "").strip()}
    if label == "text" or it.get("text"):
        d["text"] = str(it.get("text") or "")
    tags = [str(t).strip() for t in (it.get("tags") or []) if str(t).strip()]
    if tags:
        d["tags"] = tags
    pal, _ = parse_palette(it.get("palette") or [], 5)
    if pal:
        d["palette"] = pal
    return d


def compile(draft):
    """DRAFT (dict or JSON text) -> (compiled_text, width, height): the one-line JSON the
    model reads. Raises ValueError with a readable message on an invalid draft. A
    COMPILED input is returned as is (width and height then None)."""
    d = parse(draft) if isinstance(draft, str) else draft
    if is_compiled(d):
        return json.dumps(d, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False), None, None
    problems = validate_draft(d)
    if problems:
        raise ValueError("; ".join(problems[:3]))
    w, h = int(d["width"]), int(d["height"])
    items = []
    for it in d["items"]:
        c = dict(it)
        x1, y1, x2, y2 = it["bbox"]
        c["bbox"] = [math.floor(round(y1, 9)) - h // 2, math.floor(round(x1, 9)) - w // 2,
                     math.ceil(round(y2, 9)) - h // 2, math.ceil(round(x2, 9)) - w // 2]
        items.append({k: c[k] for k in ITEM_KEYS if k in c})
    out = {"summary": {k: d["summary"][k] for k in SUMMARY_KEYS if k in d["summary"]},
           "items": items, "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}
    return json.dumps(out, separators=(",", ":"), ensure_ascii=False, allow_nan=False), w, h


def parse(text):
    """A dict from JSON text. Raises ValueError with a readable message."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON at line {e.lineno} column {e.colno}: {e.msg}")
    if not isinstance(data, dict):
        raise ValueError("a layout is a JSON object ({...})")
    return data


def is_caption(text):
    """True when `text` is a layout JSON (draft or compiled): such a prompt bypasses the
    styles, the {a|b} variants and anything else that rewrites a plain prompt."""
    if not isinstance(text, str):
        return False
    t = text.strip()
    if not (t.startswith("{") and t.endswith("}")):
        return False
    try:
        data = json.loads(t)
    except ValueError:
        return False
    return isinstance(data, dict) and "items" in data and "summary" in data


def is_draft(d):
    return (isinstance(d, dict) and "items" in d and "summary" in d
            and str(d.get("bbox_coord_space", "abs")) == "abs")


def is_compiled(d):
    return (isinstance(d, dict) and "items" in d and "summary" in d
            and d.get("bbox_coord_space") == "abs_centralized")


def from_text(text, width=None, height=None):
    """DRAFT or COMPILED text (or dict) -> (EDITOR dict, width, height). A compiled
    layout has no canvas: `width` / `height` (the sliders) say which one it was drawn
    for; without them the canvas is guessed from the farthest box, rounded to 16."""
    d = parse(text) if isinstance(text, str) else text
    if not (isinstance(d, dict) and "items" in d):
        raise ValueError("not a MOSS layout (no 'items')")
    summary = d.get("summary") if isinstance(d.get("summary"), dict) else {}
    items_in = d.get("items") if isinstance(d.get("items"), list) else []
    compiled = is_compiled(d)
    if compiled:
        if width and height:
            w, h = int(width), int(height)
        else:
            w, h = guess_canvas(d)
    else:
        try:
            w, h = int(d.get("width") or width or 1024), int(d.get("height") or height or 1024)
        except (TypeError, ValueError):
            w, h = int(width or 1024), int(height or 1024)
    items = []
    for it in items_in:
        if not isinstance(it, dict):
            continue
        e = {"label": it.get("label", "object"), "caption": it.get("caption", ""),
             "text": it.get("text", ""), "tags": list(it.get("tags") or []),
             "palette": list(it.get("palette") or []), "bbox": None}
        b = it.get("bbox")
        if isinstance(b, list) and len(b) == 4:
            try:
                vals = [float(v) for v in b]
            except (TypeError, ValueError):
                vals = None
            if vals:
                if compiled:
                    y0, x0, y1, x1 = vals
                    e["bbox"] = bbox_from_pixels(x0 + w / 2, y0 + h / 2, x1 + w / 2,
                                                 y1 + h / 2, w, h)
                else:
                    x0, y0, x1, y1 = vals
                    e["bbox"] = bbox_from_pixels(x0, y0, x1, y1, w, h)
        items.append(e)
    layout = {"summary": {"caption": summary.get("caption", ""),
                          "meta": list(summary.get("meta") or []),
                          "canvas": summary.get("canvas", ""),
                          "palette": list(summary.get("palette") or [])},
              "items": items}
    return layout, w, h


def guess_canvas(compiled):
    """The smallest 16-multiple canvas that holds every centred box (a compiled layout
    carries no size)."""
    mx = my = 0.0
    for it in compiled.get("items") or []:
        b = it.get("bbox") if isinstance(it, dict) else None
        if isinstance(b, list) and len(b) == 4:
            try:
                y0, x0, y1, x1 = (float(v) for v in b)
            except (TypeError, ValueError):
                continue
            my = max(my, abs(y0), abs(y1))
            mx = max(mx, abs(x0), abs(x1))
    w = max(GRID, int(math.ceil(2 * mx / GRID) * GRID)) if mx else 1024
    h = max(GRID, int(math.ceil(2 * my / GRID) * GRID)) if my else 1024
    return w, h


# --- Colours and boxes ---------------------------------------------------------------
def normalize_hex(value):
    """'#abc' / 'aabbcc' / '#AaBbCc' -> '#AABBCC'; None when it is not a colour."""
    if not isinstance(value, str):
        return None
    m = _HEX_RE.match(value.strip())
    if not m:
        return None
    h = m.group(1)
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return "#" + h.upper()


def parse_palette(value, limit):
    """A palette from a list or from free text ('#fff, 1b1b2f #E43F5A'): normalized,
    duplicates dropped, cut at `limit`. Returns (palette, rejected_tokens)."""
    if isinstance(value, str):
        tokens = [t for t in re.split(r"[\s,;]+", value) if t]
    else:
        tokens = list(value or [])
    out, bad = [], []
    for t in tokens:
        h = normalize_hex(t)
        if h is None:
            bad.append(str(t))
        elif h not in out:
            out.append(h)
    return out[:limit], bad + [f"(over {limit}) {h}" for h in out[limit:]]


def bbox_from_pixels(x0, y0, x1, y1, width, height):
    """A box in pixels on a width x height canvas -> [y_min, x_min, y_max, x_max] in
    0..1000, ordered and clamped."""
    def n(v, size):
        return max(0, min(BBOX_SCALE, int(round(float(v) * BBOX_SCALE / float(size)))))
    xa, xb = sorted((n(x0, width), n(x1, width)))
    ya, yb = sorted((n(y0, height), n(y1, height)))
    return [ya, xa, yb, xb]


def bbox_to_pixels(bbox, width, height):
    """[y_min, x_min, y_max, x_max] in 0..1000 -> (x0, y0, x1, y1) in pixels."""
    y0, x0, y1, x1 = bbox
    return (round(x0 * width / BBOX_SCALE), round(y0 * height / BBOX_SCALE),
            round(x1 * width / BBOX_SCALE), round(y1 * height / BBOX_SCALE))


def parse_bbox(value):
    """A 0..1000 bbox from a list or from text ('120, 80, 300, 920'); None when
    empty / invalid."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        parts = [p for p in re.split(r"[\s,;\[\]]+", value) if p]
    else:
        parts = list(value)
    try:
        vals = [int(round(float(p))) for p in parts]
    except (TypeError, ValueError):
        return None
    if len(vals) != 4:
        return None
    return [max(0, min(BBOX_SCALE, v)) for v in vals]


def aspect_ratio(width, height):
    """'W:H' reduced."""
    g = math.gcd(int(width), int(height)) or 1
    return f"{int(width) // g}:{int(height) // g}"


def fit_grid(width, height):
    """The nearest 16-multiples (what MOSS-Image renders)."""
    return max(GRID, int(width) // GRID * GRID), max(GRID, int(height) // GRID * GRID)


# --- Build from editor fields / read back --------------------------------------------
def build(caption="", meta=(), canvas="single", palette=None, items=()):
    """An EDITOR dict from the form's fields; empty optionals left out of the text forms
    (to_draft) but present here so the form round-trips.

    meta: a list or comma-separated text. items: dicts {"label", "caption", "text",
    "bbox" (0..1000 list or text), "palette", "tags"}; an empty row is skipped."""
    if isinstance(meta, str):
        meta = [m.strip() for m in meta.split(",")]
    pal, _ = parse_palette(palette or [], 16)
    out_items = []
    for it in items or ():
        label = str(it.get("label") or "object").strip().lower()
        if label == "obj":
            label = "object"
        if label not in LABELS:
            label = "object"
        text = str(it.get("text") or "")
        cap = str(it.get("caption") or "").strip()
        if not cap and not (label == "text" and text):
            continue                                  # an empty editor row
        e = {"label": label, "caption": cap, "text": text if label == "text" else "",
             "bbox": parse_bbox(it.get("bbox")),
             "palette": parse_palette(it.get("palette") or [], 5)[0]}
        tags = it.get("tags") or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",")]
        e["tags"] = [t for t in tags if t]
        out_items.append(e)
    return {"summary": {"caption": (caption or "").strip(),
                        "meta": [m for m in meta if m],
                        "canvas": (canvas or "").strip(), "palette": pal},
            "items": out_items}


def fields_of(layout):
    """The editor fields of an EDITOR dict (the inverse of build)."""
    s = layout.get("summary") or {}
    items = []
    for it in layout.get("items") or []:
        items.append({"label": it.get("label", "object"), "caption": it.get("caption", ""),
                      "text": it.get("text", ""), "bbox": it.get("bbox"),
                      "palette": list(it.get("palette") or []),
                      "tags": list(it.get("tags") or [])})
    return {"caption": s.get("caption", ""), "meta": list(s.get("meta") or []),
            "canvas": s.get("canvas", "single"), "palette": list(s.get("palette") or []),
            "items": items}


# --- Validation ----------------------------------------------------------------------
def validate_draft(d):
    """Problems of a DRAFT dict (the official compiler's checks): [] when it compiles."""
    p = []
    if not isinstance(d, dict):
        return ["the draft is not a JSON object"]
    root = {"width", "height", "bbox_coord_space", "bbox_format", "summary", "items"}
    unknown = sorted(set(d) - root)
    missing = sorted(root - set(d))
    if unknown:
        p.append(f"unknown top-level fields {unknown}")
    if missing:
        p.append(f"missing top-level fields {missing}")
    w, h = d.get("width"), d.get("height")
    for name, v in (("width", w), ("height", h)):
        if not (type(v) is int and v > 0 and v % GRID == 0):
            p.append(f"{name} must be a positive multiple of 16 (got {v!r})")
    if d.get("bbox_coord_space") != "abs":
        p.append("bbox_coord_space must be 'abs' in a draft")
    if d.get("bbox_format") != "xyxy":
        p.append("bbox_format must be 'xyxy' in a draft")
    p += _validate_summary(d.get("summary"))
    items = d.get("items")
    if not isinstance(items, list) or not items:
        p.append("items must be a nonempty array")
        return p
    ok_canvas = (type(w) is int and type(h) is int and w > 0 and h > 0)
    for i, it in enumerate(items):
        loc = f"items[{i}]"
        p += _validate_item(it, loc)
        b = it.get("bbox") if isinstance(it, dict) else None
        if not (isinstance(b, list) and len(b) == 4):
            p.append(f"{loc}.bbox must be [x1, y1, x2, y2]")
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in b)
        except (TypeError, ValueError):
            p.append(f"{loc}.bbox must hold four numbers")
            continue
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            p.append(f"{loc}.bbox must hold finite numbers")
        elif not (x1 < x2 and y1 < y2):
            p.append(f"{loc}.bbox has no area {b}")
        elif ok_canvas and not (0 <= x1 and x2 <= w and 0 <= y1 and y2 <= h):
            p.append(f"{loc}.bbox exceeds the {w}x{h} canvas {b}")
    return p


def _validate_summary(s):
    p = []
    if not isinstance(s, dict):
        return ["summary must be an object"]
    unknown = sorted(set(s) - set(SUMMARY_KEYS))
    if unknown:
        p.append(f"summary: unknown fields {unknown}")
    if not isinstance(s.get("caption"), str) or not s.get("caption", "").strip():
        p.append("summary.caption: empty (the global description of the image)")
    if "canvas" in s and s["canvas"] not in CANVASES:
        p.append(f"summary.canvas must be one of {list(CANVASES)}")
    for k in ("meta", "palette"):
        if k in s and not (isinstance(s[k], list) and all(isinstance(x, str) and x.strip()
                                                            for x in s[k])):
            p.append(f"summary.{k} must be an array of non-empty strings")
    return p


def _validate_item(it, loc):
    p = []
    if not isinstance(it, dict):
        return [f"{loc} must be an object"]
    unknown = sorted(set(it) - set(ITEM_KEYS))
    if unknown:
        p.append(f"{loc}: unknown fields {unknown}")
    if it.get("label") not in LABELS:
        p.append(f"{loc}.label must be one of {list(LABELS)}")
    if not isinstance(it.get("caption"), str) or not it.get("caption", "").strip():
        p.append(f"{loc}.caption: empty (what this element looks like)")
    if it.get("label") == "text" and not (isinstance(it.get("text"), str) and it["text"]):
        p.append(f"{loc}.text: empty (a text item renders these exact characters)")
    for k in ("tags", "palette"):
        if k in it and not (isinstance(it[k], list) and all(isinstance(x, str) and x.strip()
                                                              for x in it[k])):
            p.append(f"{loc}.{k} must be an array of non-empty strings")
    return p


def validate(layout_or_text, width=None, height=None):
    """Warnings for a layout in any shape (EDITOR dict with width / height, DRAFT or
    COMPILED dict or text); [] means it compiles. Our own checks on top of the official
    compiler's: a box too thin to render (under 16 px a side), a text item whose text is
    only whitespace, a 0..1000 box with no area."""
    if isinstance(layout_or_text, str):
        try:
            d = parse(layout_or_text)
        except ValueError as e:
            return [str(e)]
    else:
        d = layout_or_text
    if not isinstance(d, dict):
        return ["not a JSON object"]
    if is_compiled(d):
        p = _validate_summary(d.get("summary"))
        items = d.get("items")
        if not isinstance(items, list) or not items:
            return p + ["items must be a nonempty array"]
        for i, it in enumerate(items):
            p += _validate_item(it, f"items[{i}]")
            b = it.get("bbox") if isinstance(it, dict) else None
            if isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
                if not (b[0] < b[2] and b[1] < b[3]):
                    p.append(f"items[{i}].bbox has no area {b}")
            else:
                p.append(f"items[{i}].bbox must be [y_min, x_min, y_max, x_max]")
        return p
    if "width" in d or "bbox_coord_space" in d:
        p = validate_draft(d)
        w, h = d.get("width"), d.get("height")
    else:
        if not (width and height):
            width, height = 1024, 1024
        p = validate_draft(to_draft(d, width, height))
        w, h = int(width), int(height)
        for i, it in enumerate(d.get("items") or []):
            b = parse_bbox(it.get("bbox")) if isinstance(it, dict) else None
            if b and (b[0] >= b[2] or b[1] >= b[3]):
                p.append(f"items[{i}].bbox: no area {b}")
    items = d.get("items") if isinstance(d.get("items"), list) else []
    draft_items = (to_draft(d, w, h)["items"] if "width" not in d and w and h else items)
    for i, it in enumerate(draft_items):
        b = it.get("bbox") if isinstance(it, dict) else None
        if isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
            if b[2] - b[0] < GRID or b[3] - b[1] < GRID:
                p.append(f"items[{i}].bbox: under 16 px a side, nothing renders there")
        if isinstance(it, dict) and it.get("label") == "text" and isinstance(it.get("text"), str) \
                and it["text"] and not it["text"].strip():
            p.append(f"items[{i}].text: whitespace only")
    return p
