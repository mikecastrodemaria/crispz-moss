"""crispz-moss - design presets (lot 2): a size and a layout skeleton per format.

Each preset is what a designer would set up before the first word: the canvas (real
proportions, multiples of 16, 0.6 to 1.9 Mpx, inside MOSS-Image's recommended 1 to 2
Mpx where the format allows), a summary caption, the meta tag, and placeholder items with
boxes where that format puts them. Loading one fills the layout editor and sets the
size; the placeholders (in CAPITALS) are meant to be replaced. Boxes are
[y_min, x_min, y_max, x_max] in 0..1000 (cz_caption's EDITOR form).

The A/B of lot 1 (CLAUDE.md): a layout lists the visible texts one by one, and the
model renders those and nothing else; in plain text it may add words it infers from
the brief (a misspelt "COFFEE SHOP" on a coffee logo). So the Logo preset, and every
other preset here, goes through the layout.

No torch, no gradio: cz_caption_ui reads PRESETS, cz_caption builds the JSON.
"""

import cz_caption

# name -> {size, hint, meta, caption, items}
PRESETS = {
    "Poster (A-series, portrait)": {
        "size": (1152, 1632),
        "hint": "A-series proportions (1:1.41), printable at A4/A3; 1.9 Mpx",
        "meta": ["poster"],
        "caption": "A bold event poster, flat graphic design with a large headline across "
                   "the top, a central illustration and the event details below it, on a "
                   "solid colour field with a subtle paper grain, no other text.",
        "items": [
            {"label": "text", "text": "HEADLINE", "bbox": [60, 80, 230, 920],
             "caption": "Very large bold condensed sans-serif headline across the top, "
                        "one line, centered."},
            {"label": "object", "bbox": [260, 120, 760, 880],
             "caption": "A central flat illustration of THE SUBJECT, simple bold shapes."},
            {"label": "text", "text": "DATE · PLACE", "bbox": [800, 120, 880, 880],
             "caption": "Medium uppercase details line, wide letter spacing, centered."},
        ],
    },
    "Logo (square)": {
        "size": (1024, 1024),
        "hint": "1:1, a mark and a wordmark on a plain field",
        "meta": ["logo"],
        "caption": "A minimalist logo: a simple geometric mark above a clean wordmark, "
                   "centered on a plain solid background, flat vector, two colours, no "
                   "gradients, no other text.",
        "items": [
            {"label": "object", "bbox": [200, 330, 560, 670],
             "caption": "A simple geometric symbol representing THE BRAND IDEA."},
            {"label": "text", "text": "BRAND", "bbox": [620, 200, 760, 800],
             "caption": "Clean bold sans-serif wordmark centered below the symbol."},
        ],
    },
    "Business card (85 x 55 mm)": {
        "size": (1536, 992),
        "hint": "European card proportions (85 x 55 mm), front side; 1.5 Mpx",
        "meta": ["business card"],
        "caption": "The front of a clean modern business card, flat graphic design on a "
                   "solid off-white card surface with a subtle paper texture, generous "
                   "whitespace, sans-serif typography, no other text.",
        "items": [
            {"label": "text", "text": "FIRST LAST", "bbox": [330, 90, 470, 700],
             "caption": "Bold dark sans-serif name, left aligned."},
            {"label": "text", "text": "Job title", "bbox": [490, 90, 560, 700],
             "caption": "Smaller grey job title under the name."},
            {"label": "text", "text": "name@company.com", "bbox": [780, 90, 860, 700],
             "caption": "Small contact line near the bottom left."},
        ],
    },
    "Book cover (2:3)": {
        "size": (1088, 1632),
        "hint": "2:3, trade paperback / ebook cover; 1.8 Mpx",
        "meta": ["book cover"],
        "caption": "A book cover with a striking central painterly illustration, an "
                   "atmospheric backdrop filling the whole cover, the title at the top in "
                   "refined serif typography and the author's name at the bottom.",
        "items": [
            {"label": "text", "text": "TITLE", "bbox": [70, 100, 250, 900],
             "caption": "Large elegant serif title across the top third."},
            {"label": "object", "bbox": [280, 120, 800, 880],
             "caption": "The central image: THE STORY'S KEY SUBJECT."},
            {"label": "text", "text": "AUTHOR NAME", "bbox": [850, 200, 930, 800],
             "caption": "Author name in small capitals near the bottom edge."},
        ],
    },
    "Slide (16:9)": {
        "size": (1600, 896),
        "hint": "16:9 presentation slide; 1.4 Mpx",
        "meta": ["slide"],
        "caption": "A clean corporate presentation slide: a light solid background with a "
                   "thin accent band along the left edge, a bold title and a short "
                   "subtitle on the left, a simple flat illustration on the right, modern "
                   "sans-serif typography, no other text.",
        "items": [
            {"label": "text", "text": "SLIDE TITLE", "bbox": [180, 70, 330, 600],
             "caption": "Large bold dark title, left aligned."},
            {"label": "text", "text": "One-line key message", "bbox": [360, 70, 440, 600],
             "caption": "Medium grey subtitle under the title."},
            {"label": "object", "bbox": [200, 640, 820, 950],
             "caption": "A simple flat illustration of THE TOPIC on the right."},
        ],
    },
    "Packaging (front, square)": {
        "size": (1216, 1216),
        "hint": "1:1 front face of a box, a can label or a pouch; 1.5 Mpx",
        "meta": ["packaging"],
        "caption": "The front face of a product package, flat packaging design on a solid "
                   "brand colour, bold display typography, appetizing and premium, no "
                   "shelf, no photo context, no other text.",
        "items": [
            {"label": "text", "text": "BRAND", "bbox": [60, 250, 180, 750],
             "caption": "Brand wordmark centered at the top."},
            {"label": "object", "bbox": [220, 180, 700, 820],
             "caption": "Hero illustration of THE PRODUCT in the center."},
            {"label": "text", "text": "PRODUCT NAME", "bbox": [740, 120, 860, 880],
             "caption": "Large product name below the illustration."},
        ],
    },
    "YouTube thumbnail (16:9)": {
        "size": (1280, 720),
        "hint": "YouTube's own 1280 x 720; 0.9 Mpx",
        "meta": ["thumbnail"],
        "caption": "An eye-catching YouTube thumbnail: a saturated colour gradient, a huge "
                   "short headline with a thick dark outline on the left half, one strong "
                   "cut-out subject on the right half, punchy and high contrast.",
        "items": [
            {"label": "text", "text": "BIG HOOK", "bbox": [120, 50, 520, 560],
             "caption": "Huge heavy uppercase headline with a thick dark outline, left half."},
            {"label": "object", "bbox": [80, 560, 1000, 980],
             "caption": "THE SUBJECT, large, cut out, on the right half."},
        ],
    },
    "Vertical story (9:16)": {
        "size": (1024, 1824),
        "hint": "9:16 Instagram / TikTok story; 1.9 Mpx",
        "meta": ["social media"],
        "caption": "A vertical social media story on a colour gradient: a bold headline in "
                   "the upper part, a central visual, and a call to action in a rounded "
                   "pill near the bottom, modern and energetic, no other text.",
        "items": [
            {"label": "text", "text": "HEADLINE", "bbox": [120, 80, 260, 920],
             "caption": "Bold headline in the upper part, centered."},
            {"label": "object", "bbox": [300, 100, 760, 900],
             "caption": "The central visual: THE SUBJECT."},
            {"label": "text", "text": "SWIPE UP", "bbox": [820, 250, 900, 750],
             "caption": "Call to action in a rounded pill shape near the bottom."},
        ],
    },
    "Banner (6:1)": {
        "size": (1920, 320),
        "hint": "6:1 web header or roll-up top; 0.6 Mpx",
        "meta": ["banner"],
        "caption": "A wide horizontal web banner on a solid colour band: a bold headline on "
                   "one line at the left, a small flat illustration at the right end, clean "
                   "and uncluttered, no other text.",
        "items": [
            {"label": "text", "text": "HEADLINE", "bbox": [200, 40, 800, 600],
             "caption": "Bold headline on one line, left aligned, vertically centered."},
            {"label": "object", "bbox": [100, 700, 900, 960],
             "caption": "A small flat illustration of THE SUBJECT at the right end."},
        ],
    },
}


def layout_of(name):
    """The preset's EDITOR dict."""
    p = PRESETS[name]
    return cz_caption.build(p["caption"], p["meta"], "single", [], p["items"])


def size_of(name):
    return PRESETS[name]["size"]


def prompt_of(name):
    """The preset as a draft prompt at its native size."""
    w, h = size_of(name)
    return cz_caption.to_prompt(layout_of(name), w, h)
