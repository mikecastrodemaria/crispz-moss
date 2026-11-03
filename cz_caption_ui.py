"""crispz-moss - the layout editor of the Generate tab (lot 2).

A form for MOSS-Image's layout (summary caption, meta tags, canvas, palette, items), a
board where each item's box is drawn with the mouse at the canvas ratio, the design
presets, the magic prompt, a raw-JSON mode, and "Use as prompt" which puts the layout
DRAFT (cz_caption: pixel boxes, the canvas written in) in the main prompt box. The
engine compiles it at render time and renders at the draft's size: a layout is worth one
size, so "Use as prompt" also sets the width / height sliders.

Layout: build_panel() creates the components where it is called; wire() connects them
once the prompt box and the width / height sliders exist (they are built later in
cz_ui). The pure helpers (rows <-> items, form <-> layout, the board's HTML) are module
functions, tested without a browser.

The board: the boxes are drawn by Python as an SVG over a div at the canvas ratio. The
only JavaScript (HEAD_JS) is the mouse drag: on release it writes the box, in 0..1000
[y0, x0, y1, x1] of the canvas, into a hidden textbox and fires its input event; Python
then puts it on the selected item.
"""

import html
import json
import time

import gradio as gr

import cz_caption
import cz_design
import cz_magic
from cz_core import _log

COLS = ["label", "text", "caption", "bbox", "palette"]
BRIDGE_ID = "cz_bbox_bridge"
BOARD_ID = "cz_board"
BOARD_MAX_PX = 520            # the board's longest side on screen


# --- rows <-> items ------------------------------------------------------------------
def _rows(value):
    """Gradio Dataframe value (list of lists, or a pandas frame) -> list of 5-str lists."""
    if value is None:
        return []
    if hasattr(value, "values") and hasattr(value, "columns"):
        value = value.values.tolist()
    out = []
    for r in value or []:
        r = ["" if v is None else str(v) for v in list(r)[:len(COLS)]]
        r += [""] * (len(COLS) - len(r))
        out.append(r)
    return out


def items_to_rows(items):
    rows = []
    for e in items or []:
        bbox = e.get("bbox")
        rows.append([e.get("label", "object"),
                     e.get("text", "") if e.get("label") == "text" else "",
                     e.get("caption", ""), ", ".join(str(v) for v in bbox) if bbox else "",
                     " ".join(e.get("palette") or [])])
    return rows


def rows_to_items(rows):
    out = []
    for label, text, caption, bbox, pal in _rows(rows):
        out.append({"label": label.strip().lower() or "object", "text": text,
                    "caption": caption, "bbox": bbox, "palette": pal})
    return out


def element_choices(rows):
    """'1 · text · GRAND OPENING' labels for the box target dropdown."""
    labels = []
    for i, (label, text, caption, bbox, _) in enumerate(_rows(rows)):
        name = text if label.strip().lower() == "text" and text else caption
        name = (name or "(empty)").replace("\n", " ")
        labels.append(f"{i + 1} · {label or 'object'} · {name[:40]}"
                      f"{' [box]' if bbox.strip() else ''}")
    return labels


def _index_of(label):
    try:
        return int(str(label).split("·")[0].strip()) - 1
    except (ValueError, AttributeError):
        return -1


def apply_bbox(rows, label, bbox):
    """Puts `bbox` ([y0, x0, y1, x1] in 0..1000, or None to clear) on the item named by
    `label`."""
    rows = _rows(rows)
    i = _index_of(label)
    if 0 <= i < len(rows):
        rows[i][3] = ", ".join(str(v) for v in bbox) if bbox else ""
    return rows


# --- form <-> layout -----------------------------------------------------------------
def form_to_layout(caption, meta, canvas, palette, rows):
    return cz_caption.build(caption, meta or "", canvas or "single", palette or "",
                            rows_to_items(rows))


def layout_to_form(lay):
    f = cz_caption.fields_of(lay)
    return (f["caption"], ", ".join(f["meta"]), f["canvas"] or "single",
            " ".join(f["palette"]), items_to_rows(f["items"]))


def warnings_md(warnings, extra=""):
    if not warnings:
        return f"Layout: valid. {extra}".strip()
    items = "\n".join(f"- {html.escape(w)}" for w in warnings[:12])
    more = f"\n- ... {len(warnings) - 12} more" if len(warnings) > 12 else ""
    return f"Layout: {len(warnings)} warning(s). {extra}\n{items}{more}".strip()


# --- the board -----------------------------------------------------------------------
def _swatches(colors, size=18):
    return "".join(f'<span title="{c}" style="display:inline-block;width:{size}px;'
                   f'height:{size}px;margin-right:3px;border:1px solid #555;'
                   f'background:{c}"></span>' for c in colors)


def render_board(lay, width, height, selected=None):
    """The board's HTML: a div at the canvas ratio, the items' boxes as SVG (text =
    orange, object / character = blue, the selected one thicker), the palette swatches."""
    w, h = max(1, int(width or 1024)), max(1, int(height or 1024))
    k = BOARD_MAX_PX / float(max(w, h))
    bw, bh = max(60, int(w * k)), max(30, int(h * k))
    pal = (lay.get("summary") or {}).get("palette") or []
    bg = pal[0] if pal else "#2b2b2b"
    sel = _index_of(selected)
    shapes = []
    for i, e in enumerate(lay.get("items") or []):
        b = e.get("bbox")
        if not (isinstance(b, list) and len(b) == 4):
            continue
        y0, x0, y1, x1 = b
        col = "#f59e0b" if e.get("label") == "text" else "#3b82f6"
        sw = 3 if i == sel else 1.5
        label = html.escape((e.get("text") if e.get("label") == "text" else e.get("caption")) or "")[:30]
        shapes.append(
            f'<rect x="{x0 / 10}%" y="{y0 / 10}%" width="{(x1 - x0) / 10}%" '
            f'height="{(y1 - y0) / 10}%" fill="{col}" fill-opacity="0.15" stroke="{col}" '
            f'stroke-width="{sw}"/>'
            f'<text x="{x0 / 10 + 0.8}%" y="{y0 / 10 + 4}%" fill="{col}" '
            f'font-size="12" font-family="sans-serif">{i + 1} {label}</text>')
    svg = (f'<svg width="100%" height="100%" style="position:absolute;left:0;top:0;'
           f'pointer-events:none">{"".join(shapes)}</svg>')
    return (f'<div style="font-size:12px;opacity:.8;margin-bottom:4px">Canvas {w} x {h} '
            f'({cz_caption.aspect_ratio(w, h)}). Pick the item to place above, then '
            f'drag its box here.</div>'
            f'<div id="{BOARD_ID}" class="cz-board-surface" style="position:relative;'
            f'width:{bw}px;height:{bh}px;background:{bg};border:1px solid #666;'
            f'cursor:crosshair;user-select:none">{svg}</div>'
            f'<div style="margin-top:6px">{_swatches(pal)}</div>')


HEAD_JS = """
<style>#%(bridge)s{display:none !important}</style>
<script>
(function () {
  let drag = null;
  function surface(t) { return t && t.closest ? t.closest('#%(board)s') : null; }
  document.addEventListener('mousedown', function (ev) {
    const s = surface(ev.target);
    if (!s || ev.button !== 0) return;
    ev.preventDefault();
    const r = s.getBoundingClientRect();
    const tmp = document.createElement('div');
    tmp.style.cssText = 'position:absolute;border:2px dashed #fff;pointer-events:none';
    s.appendChild(tmp);
    drag = {s: s, r: r, x0: ev.clientX - r.left, y0: ev.clientY - r.top, tmp: tmp};
  });
  document.addEventListener('mousemove', function (ev) {
    if (!drag) return;
    const x = Math.max(0, Math.min(drag.r.width, ev.clientX - drag.r.left));
    const y = Math.max(0, Math.min(drag.r.height, ev.clientY - drag.r.top));
    const t = drag.tmp;
    t.style.left = Math.min(x, drag.x0) + 'px'; t.style.top = Math.min(y, drag.y0) + 'px';
    t.style.width = Math.abs(x - drag.x0) + 'px'; t.style.height = Math.abs(y - drag.y0) + 'px';
  });
  document.addEventListener('mouseup', function (ev) {
    if (!drag) return;
    const d = drag; drag = null;
    const x = Math.max(0, Math.min(d.r.width, ev.clientX - d.r.left));
    const y = Math.max(0, Math.min(d.r.height, ev.clientY - d.r.top));
    d.tmp.remove();
    if (Math.abs(x - d.x0) < 4 || Math.abs(y - d.y0) < 4) return;   // a click, not a box
    const n = (v, size) => Math.round(v * 1000 / size);
    const box = [n(Math.min(y, d.y0), d.r.height), n(Math.min(x, d.x0), d.r.width),
                 n(Math.max(y, d.y0), d.r.height), n(Math.max(x, d.x0), d.r.width)];
    const ta = document.querySelector('#%(bridge)s textarea, #%(bridge)s input');
    if (!ta) return;
    ta.value = JSON.stringify(box) + '|' + Date.now();
    ta.dispatchEvent(new Event('input', {bubbles: true}));
  });
})();
</script>
""" % {"bridge": BRIDGE_ID, "board": BOARD_ID}


def parse_bridge(value):
    """'[y0, x0, y1, x1]|timestamp' from the board's JavaScript -> the bbox, or None."""
    if not value:
        return None
    return cz_caption.parse_bbox(str(value).split("|")[0])


# --- the panel -----------------------------------------------------------------------
def build_panel():
    """The components, created where it is called (inside the Generate column)."""
    p = {}
    with gr.Accordion("MOSS layout (JSON): presets, magic prompt, boxes", open=False):
        gr.Markdown("A layout tells MOSS-Image the scene, every visible text VERBATIM and "
                    "where each thing goes; the model then renders those texts and nothing "
                    "else (in plain text it may add words it infers from the brief). "
                    "**Use as prompt** puts the layout in the prompt box and sets the size "
                    "it was drawn for; Generate does the rest.")
        with gr.Row():
            p["preset"] = gr.Dropdown(list(cz_design.PRESETS), label="Design preset",
                                      value=None, scale=3)
            p["preset_btn"] = gr.Button("Load preset (sets the size)", scale=1)
        with gr.Row():
            engines = [(cz_magic.ENGINE_LABELS[e], e) for e in cz_magic.visible_engines()]
            p["engine"] = gr.Radio(engines, value=cz_magic.engine(), label="Magic prompt engine",
                                   info="Both engines run on this machine: nothing leaves. "
                                        "Ollama needs a local model; Local reuses the "
                                        "loaded Qwen3.5-9B (its LM head, ~2 GB, downloaded "
                                        "once).", scale=3)
            p["magic_btn"] = gr.Button("Magic prompt: idea -> layout", variant="primary",
                                       scale=1)
        with gr.Accordion("Describe an image -> layout (local Qwen3.5-9B)", open=False):
            with gr.Row():
                p["describe_img"] = gr.Image(type="pil", label="Image to describe",
                                             height=220, scale=2)
                with gr.Column(scale=1):
                    gr.Markdown("The loaded text encoder (Qwen3.5-9B, vision included) "
                                "reads the image: summary, palette, every visible text "
                                "verbatim, and a box per element. Nothing leaves the "
                                "machine. The first use downloads the LM head (~2 GB).")
                    p["describe_btn"] = gr.Button("Describe -> layout", variant="primary")
        p["caption"] = gr.Textbox(label="Summary caption (the whole image: subject, medium, "
                                        "style, lighting, palette, mood)", lines=3)
        with gr.Row():
            p["meta"] = gr.Textbox(label="Meta tags (comma separated: poster, logo...)",
                                   lines=1, scale=2)
            p["canvas"] = gr.Dropdown(list(cz_caption.CANVASES), value="single",
                                      label="Canvas", scale=1)
        p["palette"] = gr.Textbox(label="Palette (hex: #1B1B2F #E43F5A ...)", lines=1)
        p["rows"] = gr.Dataframe(headers=COLS, datatype=["str"] * len(COLS),
                                 col_count=(len(COLS), "fixed"), row_count=(1, "dynamic"),
                                 type="array", interactive=True, wrap=True,
                                 label="Items: label = text, object or character; text = "
                                       "the exact characters; caption = what it looks "
                                       "like; bbox = y0, x0, y1, x1 in 0..1000 of the "
                                       "canvas; palette = up to 5 hex")
        with gr.Row():
            p["target"] = gr.Dropdown([], label="Item to place with the mouse", scale=3)
            p["clear_box"] = gr.Button("Clear its box", scale=1)
        p["board"] = gr.HTML(render_board(cz_caption.build(""), 1024, 1024))
        p["bridge"] = gr.Textbox(elem_id=BRIDGE_ID, show_label=False, container=False)
        p["status"] = gr.Markdown("")
        with gr.Accordion("Raw JSON (the draft: pixel boxes, the canvas written in)", open=False):
            p["raw"] = gr.Code(language="json", label="Layout draft (editable; a compiled "
                                                      "layout is accepted too)",
                               interactive=True)
            p["raw_apply"] = gr.Button("Apply this JSON to the form")
        p["use_btn"] = gr.Button("Use as prompt (and set the size)", variant="primary")
    return p


def _form_inputs(p):
    return [p["caption"], p["meta"], p["canvas"], p["palette"], p["rows"]]


def wire(p, prompt, width, height):
    """Connects the panel once `prompt`, `width` and `height` exist."""
    form = _form_inputs(p)

    def refresh(caption, meta, canvas, palette, rows, w, h, target):
        lay = form_to_layout(caption, meta, canvas, palette, rows)
        _, bad = cz_caption.parse_palette(palette or "", 16)
        extra = f"Palette entries ignored: {', '.join(bad)}." if bad else ""
        choices = element_choices(rows)
        tgt = target if target in choices else (choices[0] if choices else None)
        gw, gh = cz_caption.fit_grid(w or 1024, h or 1024)
        return (cz_caption.to_pretty(lay, gw, gh),
                warnings_md(cz_caption.validate(lay, gw, gh), extra),
                render_board(lay, gw, gh, tgt), gr.update(choices=choices, value=tgt))

    outs = [p["raw"], p["status"], p["board"], p["target"]]
    ins = form + [width, height, p["target"]]
    for c in form + [width, height]:
        c.change(refresh, ins, outs, show_progress="hidden")

    def board_only(*vals):
        *fields, w, h, target = vals
        return render_board(form_to_layout(*fields), w, h, target)
    # The target's own change redraws the board only (it must not rewrite the target).
    p["target"].change(board_only, ins, [p["board"]], show_progress="hidden")

    def load_preset(name):
        if not name:
            return [gr.update()] * (len(form) + 2)
        w, h = cz_design.size_of(name)
        return (*layout_to_form(cz_design.layout_of(name)), w, h)
    p["preset_btn"].click(load_preset, [p["preset"]], form + [width, height])

    def magic(idea, engine, w, h):
        t0 = time.time()
        gw, gh = cz_caption.fit_grid(w or 1024, h or 1024)
        try:
            text, warnings, used = cz_magic.expand(idea, gw, gh, which=engine)
        except cz_magic.MagicPromptError as e:
            _log(f"magic prompt failed: {e}")
            return (*[gr.update()] * len(form), f"Magic prompt failed: {e}")
        except Exception as e:                       # the model itself (OOM, download...)
            _log(f"magic prompt failed: {type(e).__name__}: {e}")
            return (*[gr.update()] * len(form), f"Magic prompt failed: {type(e).__name__}: {e}")
        lay, _, _ = cz_caption.from_text(text)
        note = f"Magic prompt ({cz_magic.ENGINE_LABELS[used]}) in {time.time() - t0:.0f} s."
        return (*layout_to_form(lay), warnings_md(warnings, note))
    p["magic_btn"].click(magic, [prompt, p["engine"], width, height], form + [p["status"]])

    def describe(img):
        t0 = time.time()
        try:
            text, warnings = cz_magic.describe(img)
        except cz_magic.MagicPromptError as e:
            return (*[gr.update()] * len(form), f"Describe failed: {e}", gr.update(), gr.update())
        except Exception as e:
            _log(f"describe failed: {type(e).__name__}: {e}")
            return (*[gr.update()] * len(form), f"Describe failed: {type(e).__name__}: {e}",
                    gr.update(), gr.update())
        lay, w, h = cz_caption.from_text(text)
        return (*layout_to_form(lay),
                warnings_md(warnings, f"Described locally in {time.time() - t0:.0f} s; "
                                      f"size set to the image's ({w}x{h})."), w, h)
    p["describe_btn"].click(describe, [p["describe_img"]], form + [p["status"], width, height])

    def from_raw(raw, w, h):
        try:
            lay, dw, dh = cz_caption.from_text(raw or "", w, h)
        except ValueError as e:
            return (*[gr.update()] * len(form), f"Raw JSON not applied: {e}", gr.update(),
                    gr.update())
        warnings = cz_caption.validate(raw or "")
        return (*layout_to_form(lay), warnings_md(
            warnings, f"Applied; size set to the draft's ({dw}x{dh})."), dw, dh)
    p["raw_apply"].click(from_raw, [p["raw"], width, height],
                         form + [p["status"], width, height])

    def on_box(value, rows, target):
        bbox = parse_bridge(value)
        if bbox is None or _index_of(target) < 0:
            return gr.update()
        return apply_bbox(rows, target, bbox)
    p["bridge"].input(on_box, [p["bridge"], p["rows"], p["target"]], [p["rows"]])
    p["clear_box"].click(lambda rows, t: apply_bbox(rows, t, None),
                         [p["rows"], p["target"]], [p["rows"]])

    def use(caption, meta, canvas, palette, rows, w, h):
        lay = form_to_layout(caption, meta, canvas, palette, rows)
        gw, gh = cz_caption.fit_grid(w or 1024, h or 1024)
        text = cz_caption.to_prompt(lay, gw, gh)
        warnings = cz_caption.validate(text)
        if warnings:
            _log(f"layout used as prompt with {len(warnings)} warning(s): "
                 + "; ".join(warnings[:3]))
        return (text, warnings_md(warnings, f"Layout put in the prompt box, drawn for "
                                            f"{gw}x{gh}."), gw, gh)
    p["use_btn"].click(use, form + [width, height], [prompt, p["status"], width, height])
