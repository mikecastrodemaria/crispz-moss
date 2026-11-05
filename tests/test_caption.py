"""cz_caption: the MOSS-Image layout in its three shapes (editor 0..1000, draft in pixels,
compiled centred), the presets, the editor's pure half.

No model, no GPU. The fixture is the USERGUIDE's own layout example (1024x1024 poster,
"EXPLORE") and the official skill's compile example (1536x1024, Chinese copy): the
compiler here must give the very numbers those documents print.

Run:  .venv/Scripts/python -m pytest tests/test_caption.py -q
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_caption as C  # noqa: E402

# USERGUIDE.md: a 1024x1024 canvas, boxes already centred (the compiled form).
GUIDE_COMPILED = {
    "summary": {"meta": ["poster"], "canvas": "single",
                "caption": "A minimalist travel poster with a cream background, a blue "
                           "mountain illustration, and the title EXPLORE at the top."},
    "items": [
        {"label": "text", "caption": "Large dark-blue uppercase title centered at the top.",
         "text": "EXPLORE", "bbox": [-416, -384, -288, 384]},
        {"label": "object", "caption": "A flat blue mountain illustration in the lower center.",
         "bbox": [-192, -288, 384, 288]}],
    "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}

# skills/.../references/layout.md: the draft (pixels, top-left) and what it compiles to.
SKILL_DRAFT = {
    "width": 1536, "height": 1024, "bbox_coord_space": "abs", "bbox_format": "xyxy",
    "summary": {"meta": ["product campaign", "studio photography"], "canvas": "single",
                "caption": "A warm studio campaign for a handmade ceramic cup."},
    "items": [
        {"label": "object", "caption": "An ivory ceramic cup with a rounded handle.",
         "bbox": [176, 224, 800, 864]},
        {"label": "text", "caption": "Large dark-green Chinese title.",
         "text": "把日常，过成诗。", "bbox": [928, 272, 1408, 448]},
        {"label": "text", "caption": "Smaller supporting copy on two lines.",
         "text": "一杯温暖\n一刻从容", "bbox": [928, 512, 1376, 656]}]}


def test_the_skill_example_compiles_to_the_documented_numbers():
    text, w, h = C.compile(SKILL_DRAFT)
    assert (w, h) == (1536, 1024)
    d = json.loads(text)
    assert d["items"][0]["bbox"] == [-288, -592, 352, 32]          # the number in layout.md
    assert d["bbox_coord_space"] == "abs_centralized" and d["bbox_format"] == "yxyx"
    assert "width" not in d and "\n" not in text and "一杯温暖\\n一刻从容" in text
    assert list(d) == ["summary", "items", "bbox_coord_space", "bbox_format"]
    assert list(d["items"][1]) == ["label", "caption", "text", "bbox"]
    assert C.validate(SKILL_DRAFT) == [] and C.validate(text) == []


def test_fractional_edges_are_enclosed_like_the_official_compiler():
    d = dict(SKILL_DRAFT, items=[{"label": "object", "caption": "c",
                                  "bbox": [10.4, 20.6, 100.2, 200.7]}])
    out = json.loads(C.compile(d)[0])["items"][0]["bbox"]
    assert out == [20 - 512, 10 - 768, 201 - 512, 101 - 768]


def test_the_guide_example_reads_back_and_recompiles():
    lay, w, h = C.from_text(json.dumps(GUIDE_COMPILED), 1024, 1024)
    assert (w, h) == (1024, 1024)
    assert lay["items"][0]["bbox"] == [94, 125, 219, 875]           # 0..1000 of the canvas
    again = json.loads(C.compile(C.to_draft(lay, 1024, 1024))[0])
    # the editor keeps thousandths of the canvas: a pixel of play at most on the way back
    for got, want in ((again["items"][0]["bbox"], [-416, -384, -288, 384]),
                      (again["items"][1]["bbox"], [-192, -288, 384, 288])):
        assert all(abs(a - b) <= 1 for a, b in zip(got, want)), (got, want)
    assert again["summary"] == GUIDE_COMPILED["summary"]
    # without a size the canvas is guessed from the farthest box (384 -> 768, 16-grid)
    _, gw, gh = C.from_text(json.dumps(GUIDE_COMPILED))
    assert (gw, gh) == (768, 832)
    assert C.validate(GUIDE_COMPILED) == []


def test_build_and_fields_round_trip():
    lay = C.build("A poster.", "poster, event", "single", "#fff, 1b1b2f",
                  [{"label": "text", "text": "GRAND OPENING", "caption": "Huge headline.",
                    "bbox": "100, 80, 300, 920", "palette": ["#ffffff"]},
                   {"label": "obj", "caption": "", "text": ""},                 # empty row
                   {"label": "obj", "caption": "A croissant basket."}])
    assert lay["summary"]["meta"] == ["poster", "event"]
    assert lay["summary"]["palette"] == ["#FFFFFF", "#1B1B2F"]
    assert [it["label"] for it in lay["items"]] == ["text", "object"]
    assert lay["items"][0]["bbox"] == [100, 80, 300, 920]
    f = C.fields_of(lay)
    assert C.build(f["caption"], f["meta"], f["canvas"], f["palette"], f["items"]) == lay
    d = C.to_draft(lay, 720, 1024)
    assert d["items"][0]["bbox"] == [58, 102, 662, 307]
    assert "bbox" not in d["items"][1] and "palette" not in d["items"][1]
    assert C.validate(C.to_prompt(lay, 720, 1024)) == ["items[1].bbox must be [x1, y1, x2, y2]"]


def test_serialization_is_one_utf8_line():
    lay = C.build("Café « crème »", [], "single", [], [{"label": "text", "text": "日本",
                                                        "caption": "x", "bbox": [0, 0, 500, 500]}])
    t = C.to_prompt(lay, 512, 512)
    assert "\n" not in t and ", " not in t and "Café" in t and "日本" in t and "\\u" not in t
    assert C.is_caption(t) and C.is_caption(C.compile(t)[0])


def test_is_caption():
    assert not C.is_caption("a poster that says GRAND OPENING")
    assert not C.is_caption("{red|blue} car")
    assert not C.is_caption('{"a": 1}')
    assert not C.is_caption('{"high_level_description": "an Ideogram caption"}')
    assert C.is_caption('  {"summary": {"caption": "x"}, "items": []}  ')


def test_colours_and_boxes():
    assert C.normalize_hex("#abc") == "#AABBCC" and C.normalize_hex("red") is None
    pal, bad = C.parse_palette("#fff red #FFF 000000", 16)
    assert pal == ["#FFFFFF", "#000000"] and bad == ["red"]
    b = C.bbox_from_pixels(1536, 768, 512, 256, 2048, 1024)
    assert b == [250, 250, 750, 750]
    assert C.bbox_to_pixels(b, 2048, 1024) == (512, 256, 1536, 768)
    assert C.parse_bbox("[10, 20, 30, 40]") == [10, 20, 30, 40]
    assert C.parse_bbox("10 20 30") is None and C.parse_bbox("") is None
    assert C.aspect_ratio(1920, 320) == "6:1" and C.fit_grid(1000, 1000) == (992, 992)


def test_validate_catches_what_matters():
    bad = {"width": 1000, "height": 1024, "bbox_coord_space": "abs", "bbox_format": "xyxy",
           "summary": {"caption": "", "canvas": "round", "extra": 1},
           "items": [{"label": "thing", "caption": "", "bbox": [10, 10, 5, 5]},
                     {"label": "text", "caption": "c", "bbox": [0, 0, 2000, 10]},
                     {"label": "object", "caption": "c", "bbox": [0, 0, 8, 8], "tags": "x"}]}
    w = "\n".join(C.validate(bad))
    for needle in ("width must be a positive multiple of 16", "summary.caption: empty",
                   "summary.canvas must be one of", "unknown fields ['extra']",
                   "items[0].label must be one of", "items[0].caption: empty",
                   "items[0].bbox has no area", "items[1].text: empty",
                   "items[1].bbox exceeds the 1000x1024 canvas",
                   "items[2].tags must be an array", "under 16 px a side"):
        assert needle in w, (needle, w)
    assert C.validate("{not json") and "invalid JSON" in C.validate("{not json")[0]
    assert C.validate({"summary": {"caption": "x"}, "items": []}, 512, 512) == \
        ["items must be a nonempty array"]
    try:
        C.compile(bad)
    except ValueError as e:
        assert "width" in str(e)
    else:
        raise AssertionError("compile must refuse an invalid draft")


def test_prompt_pipeline_leaves_a_layout_alone():
    """Styles and {a|b} variants rewrite plain prompts; on a layout JSON they would
    produce invalid JSON, so the layout goes through untouched (and the log says so)."""
    import cz_prompt
    lay = C.build("A sign that says {OPEN|CLOSED}.", [], "single", [],
                  [{"label": "text", "text": "OPEN|CLOSED", "caption": "sign",
                    "bbox": [0, 0, 500, 500]}])
    t = C.to_prompt(lay, 512, 512)
    p, _ = cz_prompt.expand_prompt_pair(t, "", 123)
    assert p == t
    style = next(iter(cz_prompt.STYLES))
    p2, _ = cz_prompt._apply_styles(t, "", [style])
    assert p2 == t and C.validate(p2) == []
    p3, _ = cz_prompt.expand_prompt_pair("a {red|red} car", "", 1)
    assert p3 == "a red car"


def test_design_presets_are_valid_at_native_sizes():
    """Every preset: a valid draft at a size MOSS-Image takes as is (16 grid, 0.6 to 2
    Mpx), a meta tag, at least one text item, placeholders in capitals."""
    import cz_design
    import cz_moss
    assert len(cz_design.PRESETS) == 9
    for name in cz_design.PRESETS:
        w, h = cz_design.size_of(name)
        assert cz_moss.fit_size(w, h) == (w, h) and 0.55e6 <= w * h <= 2.0e6, (name, w, h)
        text = cz_design.prompt_of(name)
        assert C.validate(text) == [], (name, C.validate(text))
        lay = cz_design.layout_of(name)
        assert lay["summary"]["meta"] and any(it["label"] == "text" for it in lay["items"])
        compiled, cw, ch = C.compile(text)
        assert (cw, ch) == (w, h) and C.validate(compiled) == []


def test_engine_compiles_a_draft_and_forces_its_size():
    """cz_moss.prepare_prompt: a draft becomes the compiled form at the draft's size; a
    compiled layout goes through (with a note when its boxes overflow); plain text is
    untouched."""
    import cz_moss
    import cz_design
    text = cz_design.prompt_of("Banner (6:1)")
    out, w, h, note = cz_moss.prepare_prompt(text, 1024, 1024)
    assert (w, h) == (1920, 320) and json.loads(out)["bbox_coord_space"] == "abs_centralized"
    assert note and "1920x320" in note and "1024x1024" in note
    out2, w2, h2, note2 = cz_moss.prepare_prompt(text, 1920, 320)
    assert (out2, w2, h2, note2) == (out, 1920, 320, None)
    out3, w3, h3, note3 = cz_moss.prepare_prompt(out, 512, 512)
    assert out3 == out and (w3, h3) == (512, 512) and "beyond" in note3
    assert cz_moss.prepare_prompt("a plain prompt", 640, 480) == ("a plain prompt", 640, 480, None)


def test_editor_form_round_trip_and_boxes():
    """cz_caption_ui's pure half: form <-> layout, rows <-> items, the box bridge."""
    import cz_caption_ui as U
    import cz_design
    lay = cz_design.layout_of("Banner (6:1)")
    form = U.layout_to_form(lay)
    assert U.form_to_layout(*form) == lay
    rows = form[-1]
    labels = U.element_choices(rows)
    assert labels[0].startswith("1 · text · HEADLINE") and labels[0].endswith("[box]")
    rows2 = U.apply_bbox(rows, labels[1], [10, 20, 30, 40])
    assert rows2[1][3] == "10, 20, 30, 40" and rows2[0] == rows[0]
    assert U.apply_bbox(rows, labels[1], None)[1][3] == ""
    assert U.apply_bbox(rows, "nonsense", [1, 2, 3, 4]) == U._rows(rows)
    assert U.parse_bridge("[100, 50, 300, 950]|1730000000") == [100, 50, 300, 950]
    assert U.parse_bridge("") is None and U.parse_bridge("junk|1") is None
    assert U.rows_to_items([["text", "HI", None, None]])[0]["text"] == "HI"
    html = U.render_board(lay, 1920, 320, labels[0])
    assert 'id="cz_board"' in html and "<rect" in html and "6:1" in html
    assert 'stroke-width="3"' in html                     # the selected item
    assert "#cz_bbox_bridge" in U.HEAD_JS
    w = U.warnings_md(["a", "b"])
    assert "2 warning(s)" in w and U.warnings_md([]).startswith("Layout: valid")


def test_prepare_prompt_places_a_draft_on_the_source_image():
    """Outpaint / reframe (lot 3b): a draft is PLACED in the rectangle where the source
    image sits, its boxes scaled into it (never stretched over the new canvas), and the
    summary says "no other text" for the rest; plain text gets the same guard."""
    import cz_design
    import cz_moss
    draft = cz_design.prompt_of("Logo (square)")                   # drawn for 1024x1024
    # the source (1024 square) pasted at (0, 0) of a 1536x1024 canvas (outpaint right)
    text, w, h, note = cz_moss.prepare_prompt(draft, 1536, 1024, fit=True, place=(0, 0, 1024, 1024))
    assert (w, h) == (1536, 1024) and "placed on the source image" in note
    d = C.parse(text)
    assert C.is_compiled(d)
    lay, lw, lh = C.from_text(text, 1536, 1024)
    assert (lw, lh) == (1536, 1024)
    own = [it for it in lay["items"] if it.get("caption") != cz_moss.MARGIN_CAPTION]
    bands = [it for it in lay["items"] if it.get("caption") == cz_moss.MARGIN_CAPTION]
    for it in own:
        x0, y0, x1, y1 = C.bbox_to_pixels(it["bbox"], 1536, 1024)
        assert x1 <= 1024 + 2 and y1 <= 1024 + 2, (it["text"], x0, y0, x1, y1)   # inside the source
    assert "no other text" in d["summary"]["caption"].lower()
    # the new band is DECLARED as an empty-background item covering it (lot 3b, second pass)
    assert len(bands) == 1 and bands[0]["label"] == "object" and not bands[0].get("text")
    bx = C.bbox_to_pixels(bands[0]["bbox"], 1536, 1024)
    assert abs(bx[0] - 1024) <= 1 and bx[1:] == (0, 1536, 1024), bx      # a pixel of play
    assert "1 empty-background item(s) around it (right)" in note
    # reframe contain: the source reduced to 768 square, centred in 1376x768
    text, w, h, note = cz_moss.prepare_prompt(draft, 1376, 768, fit=True, place=(304, 0, 1072, 768))
    lay, _, _ = C.from_text(text, 1376, 768)
    own = [it for it in lay["items"] if it.get("caption") != cz_moss.MARGIN_CAPTION]
    bands = [C.bbox_to_pixels(it["bbox"], 1376, 768) for it in lay["items"]
             if it.get("caption") == cz_moss.MARGIN_CAPTION]
    xs = [C.bbox_to_pixels(it["bbox"], 1376, 768) for it in own]
    assert all(304 - 2 <= b[0] and b[2] <= 1072 + 2 for b in xs), xs
    assert len(bands) == 2 and "(left, right)" in note
    assert abs(bands[0][2] - 304) <= 1 and bands[0][0] == 0 and abs(bands[1][0] - 1072) <= 1
    assert cz_moss.margins_of((0, 0, 1376, 768), 1376, 768) == []          # nothing new: no item
    assert cz_moss.margins_of((0, 100, 1376, 668), 1376, 768) == [("top", [0, 0, 1376, 100]),
                                                                  ("bottom", [0, 668, 1376, 768])]
    # plain text: the guard and the bands, appended once
    t, _, _, note = cz_moss.prepare_prompt("a bakery poster", 1376, 768, fit=True, place=(0, 0, 1024, 768))
    assert t == ("a bakery poster. No other text. The added area on the right is plain empty "
                 "background, no text, no letters, no objects.") and "No other text" in note
    t2, _, _, note2 = cz_moss.prepare_prompt(t, 1376, 768, fit=True, place=(0, 0, 1024, 768))
    assert t2 == t and note2 is None
    # without place, fit still stretches (img2img of the whole image)
    text, w, h, note = cz_moss.prepare_prompt(draft, 1536, 1024, fit=True)
    assert "rescaled" in note and (w, h) == (1536, 1024)

