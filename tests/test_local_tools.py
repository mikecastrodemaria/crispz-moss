"""Lot 5: the local alternatives to the cloud functions. Remove BG (rembg) saves a
transparent PNG with its metadata; nothing leaves the machine."""
import json
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_ui  # noqa: E402


def test_remove_bg_saves_a_transparent_png_with_metadata(tmp_path, monkeypatch):
    import cz_imageio
    monkeypatch.setitem(cz_imageio.CONFIG, "date_subfolders", False)
    monkeypatch.setattr(cz_ui, "_remove_bg",
                        lambda im: Image.new("RGBA", im.size, (0, 0, 0, 0)))
    imgs, msg, hist, _ = cz_ui._ui_remove_bg(Image.new("RGB", (40, 30)), [], "custom",
                                             str(tmp_path))
    assert imgs[0].mode == "RGBA" and "locally" in msg and hist[0] is imgs[0]
    pngs = [p for p in os.listdir(tmp_path) if p.endswith(".png")]
    assert len(pngs) == 1
    meta = json.load(open(tmp_path / (pngs[0] + ".json")))
    assert meta["local"] is True and meta["tool"].startswith("rembg")


def test_remove_bg_failure_is_said(monkeypatch):
    def boom(im):
        raise RuntimeError("rembg not installed. pip install rembg")
    monkeypatch.setattr(cz_ui, "_remove_bg", boom)
    imgs, msg, _, _ = cz_ui._ui_remove_bg(Image.new("RGB", (8, 8)), [], "display", "")
    assert imgs == [] and "rembg not installed" in msg


def test_the_edit_tab_and_input_mode_are_offered():
    """Lot 4b: the 'Edit (references)' tab and input mode are offered (the Moss-Image-Edit
    LoRA path), next to the Edit in crispz-flux3 hand-over; the tab says what the LoRA
    is and that its weights carry no stated licence."""
    import gradio as gr
    demo = cz_ui.build_ui()
    tabs = {b.label: b.visible for b in demo.blocks.values() if isinstance(b, gr.Tab)}
    assert tabs.get(cz_ui.EDIT_MODE) is True
    assert tabs.get("Edit in crispz-flux3") is True and tabs.get("Vectorize (SVG)") is True
    radios = [b for b in demo.blocks.values()
              if isinstance(b, gr.Radio) and b.label == "Input mode"]
    assert radios and radios[0].visible is True and cz_ui.EDIT_MODE in radios[0].choices[1]
    texts = " ".join(str(getattr(b, "value", "")) for b in demo.blocks.values()
                     if isinstance(b, gr.Markdown))
    assert "Moss-Image-Edit LoRA" in texts and "no licence" in texts


def test_no_visible_flux3_or_ideogram_wording_in_the_ui():
    """The inherited FLUX.3 and Ideogram wording stays only in hidden components and
    comments (a visible mention of Ideogram is allowed only to name crispz-ideogram as
    the studio that runs a refused file, Ideogram LoRAs as refused, or Ideogram 4 as the
    model the Compare in crispz-ideogram tab renders with, lot 5)."""
    import gradio as gr
    demo = cz_ui.build_ui()
    parents = {}
    for b in demo.blocks.values():
        for c in getattr(b, "children", []) or []:
            parents[c._id] = b

    def shown(b):
        while b is not None:
            if getattr(b, "visible", True) is False:
                return False
            b = parents.get(b._id)
        return True
    bad = []
    for b in demo.blocks.values():
        if not shown(b):
            continue
        for attr in ("label", "info", "placeholder", "value"):
            v = getattr(b, attr, None)
            if not isinstance(v, str):
                continue
            plain = (v.replace("crispz-ideogram", "").replace("Ideogram LoRAs", "")
                     .replace("Ideogram 4", ""))
            if "FLUX.3" in v or "FLUX 3" in v or "ideogram" in plain.lower():
                bad.append(f"{type(b).__name__}.{attr}: {v[:80]}")
    assert not bad, bad
