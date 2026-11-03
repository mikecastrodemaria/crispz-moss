"""tools/gpu_lot2_check.py: the hand layouts and the box overlay (no GPU)."""
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from PIL import Image  # noqa: E402

import gpu_lot2_check as G  # noqa: E402


def test_hand_layouts_are_valid_with_real_texts():
    import cz_caption
    for name in G.HAND:
        text, size = G.hand_layout(name)
        assert cz_caption.validate(text) == [], (name, cz_caption.validate(text))
        d = json.loads(text)
        assert (d["width"], d["height"]) == size
        texts = [it.get("text") for it in d["items"] if it.get("text")]
        assert "HEADLINE" not in texts and "BRAND" not in texts
        assert not any("THE SUBJECT" in it["caption"] or "THE BRAND IDEA" in it["caption"]
                       for it in d["items"])
    text, _ = G.hand_layout("Logo (square)")
    assert "BEAN THERE" in [it.get("text") for it in json.loads(text)["items"]]
    text, _ = G.hand_layout("Poster (A-series, portrait)")
    assert "SATURDAY 9 AM · RUE DU PAIN" in text


def test_box_overlay_draws_every_box():
    text, (w, h) = G.hand_layout("Logo (square)")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "boxes.png")
        G.draw_boxes(Image.new("RGB", (w, h), (250, 240, 220)), text, p)
        out = Image.open(p)
        assert out.size == (w, h)
        # an orange text box and a blue object box were drawn
        px = out.convert("RGB").getdata()
        assert (245, 158, 11) in px and (59, 130, 246) in px
