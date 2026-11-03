"""Lot 2 validation on the GPU: layout drafts, the magic prompt, the local Describe.

The acceptance of lot 2 (CLAUDE.md): a layout built by hand and a layout from the magic
prompt both validate and both give an image whose texts sit in the boxes asked. On top,
the "no other text" question raised by the lot 1 A/B.

  1. HAND        the "Logo (square)" and "Poster (A-series)" design presets with real
                 texts -> validate -> compile -> render at the preset's size. The boxes
                 are drawn on a copy of each render (`*_boxes.png`) so the placement can
                 be judged by eye.
  2. MAGIC       two plain ideas -> magic prompt (the engine of the settings, or
                 --engine ollama|local) -> time, validity, the draft saved -> render the
                 first.
  3. DESCRIBE    the HAND logo render -> local Describe (Qwen3.5-9B + LM head) -> draft:
                 validity, does it read the texts back, boxes drawn on the image.
  4. --no-other-text   the coffee logo idea of lot 1 in PLAIN text, 4 seeds, with and
                 without a trailing "No other text." : counts the renders Mike must
                 look at (`plain_*.png` against `plainnot_*.png`).

Everything lands in bench/lot2/ (images, drafts .json, results.json, report.md).
THE GPU MUST BE FREE (same check as tools/gpu_lot1_check.py).

Usage:
    .venv\\Scripts\\python tools\\gpu_lot2_check.py
    .venv\\Scripts\\python tools\\gpu_lot2_check.py --engine local --no-other-text --steps 30
"""
import json
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
OUT = os.path.join(ROOT, "bench", "lot2")

IDEAS = [
    "a minimalist Swiss-style poster: a huge black \"5B\" on an off-white grid, a red "
    "circle overlapping the numerals, a small caption \"MOSS, real typography\" in the "
    "lower left corner",
    "a vintage coffee shop logo with the words \"BEAN THERE\" around a steaming cup",
]
PLAIN_LOGO = ('A vintage coffee shop logo with the words "BEAN THERE" curved around a '
              'steaming cup of coffee, circular badge, two-colour print, dark brown on '
              'cream, clean vector emblem on a plain cream background.')
HAND = {
    "Logo (square)": {"BRAND": "BEAN THERE", "THE BRAND IDEA": "a steaming coffee cup"},
    "Poster (A-series, portrait)": {"HEADLINE": "GRAND OPENING",
                                    "DATE · PLACE": "SATURDAY 9 AM · RUE DU PAIN",
                                    "THE SUBJECT": "a basket of golden croissants"},
}

RESULTS = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": {}}


def _say(msg):
    print(f"[lot2] {msg}", flush=True)


def _arg(name, default=None, cast=str):
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return cast(sys.argv[i + 1])
    return default


def _save():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False)
    _report()


def hand_layout(name):
    """A preset with its placeholders replaced by real texts: (draft_text, (w, h))."""
    import cz_caption
    import cz_design
    p = cz_design.PRESETS[name]
    reps = HAND[name]
    items = []
    for it in p["items"]:
        it = dict(it)
        if it.get("label") == "text" and it.get("text") in reps:
            it["text"] = reps[it["text"]]
        for k, v in reps.items():
            it["caption"] = it["caption"].replace(k, v)
        items.append(it)
    cap = p["caption"]
    for k, v in reps.items():
        cap = cap.replace(k, v)
    lay = cz_caption.build(cap, p["meta"], "single", "#F4E9D8 #3B2314", items)
    w, h = cz_design.size_of(name)
    return cz_caption.to_prompt(lay, w, h), (w, h)


def draw_boxes(img, draft_text, path):
    """The draft's boxes drawn on a copy of the render (text = orange, object = blue)."""
    from PIL import ImageDraw
    d = json.loads(draft_text)
    out = img.convert("RGB").copy()
    dr = ImageDraw.Draw(out)
    for it in d.get("items") or []:
        b = it.get("bbox")
        if not (isinstance(b, list) and len(b) == 4):
            continue
        col = (245, 158, 11) if it.get("label") == "text" else (59, 130, 246)
        dr.rectangle([b[0], b[1], b[2], b[3]], outline=col, width=4)
        dr.text((b[0] + 6, b[1] + 4), (it.get("text") or it.get("caption") or "")[:30], fill=col)
    out.save(path)


def render(P, tag, prompt, w, h, steps, seed):
    import torch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    img = P.generate(prompt, w, h, steps, seed)
    dt = time.time() - t0
    path = os.path.join(OUT, f"{tag}.png")
    img.save(path)
    r = {"size": f"{img.width}x{img.height}", "seconds": round(dt, 1),
         "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2),
         "image": os.path.relpath(path, ROOT)}
    _say(f"{tag}: {r['size']} in {r['seconds']} s")
    return img, r


def _write(tag, text):
    with open(os.path.join(OUT, f"{tag}.json"), "w", encoding="utf-8") as f:
        f.write(text)


def main():
    import gpu_lot1_check as G1
    G1.OUT = OUT
    os.makedirs(OUT, exist_ok=True)
    steps = _arg("--steps", 50, int)
    seed = _arg("--seed", 12345, int)
    engine = _arg("--engine")
    P = G1.prechecks()
    import cz_caption
    import cz_magic
    try:
        import outlines  # noqa: F401
        RESULTS["steps"]["0_env"] = {"outlines": True}
    except ImportError:
        RESULTS["steps"]["0_env"] = {"outlines": False}
        _say("outlines is NOT installed: the local magic prompt is unconstrained "
             "(pip install outlines==1.3.3 outlines-core==0.2.14).")
    RESULTS["steps"]["0_env"]["engine"] = engine or cz_magic.engine()
    off = _arg("--offload")
    if off:
        P.set_offload_mode(off)

    # 1. Hand-built layouts
    hand = {}
    for name in HAND:
        tag = "hand_" + ("logo" if "Logo" in name else "poster")
        try:
            text, (w, h) = hand_layout(name)
            _write(tag, text)
            warnings = cz_caption.validate(text)
            img, r = render(P, tag, text, w, h, steps, seed)
            draw_boxes(img, text, os.path.join(OUT, f"{tag}_boxes.png"))
            r["warnings"] = warnings
            r["texts"] = [it["text"] for it in json.loads(text)["items"] if it.get("text")]
            hand[tag] = r
        except Exception as e:
            hand[tag] = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}
            _say(f"{tag} FAILED: {e}")
        RESULTS["steps"]["1_hand"] = hand
        _save()

    # 2. Magic prompt
    magic = []
    for i, idea in enumerate(IDEAS):
        try:
            t0 = time.time()
            w, h = (720, 1024) if i == 0 else (1024, 1024)
            text, warnings, used = cz_magic.expand(idea, w, h, seed=seed + i, which=engine)
            dt = time.time() - t0
            _write(f"magic_{i + 1}", text)
            r = {"idea": idea, "engine": used, "seconds": round(dt, 1), "warnings": warnings,
                 "valid": not warnings, "chars": len(text),
                 "texts": [it.get("text") for it in json.loads(text)["items"] if it.get("text")]}
            _say(f"magic {i + 1} ({used}): {r['seconds']} s, {len(warnings)} note(s), "
                 f"texts {r['texts']}")
            img, rr = render(P, f"magic_{i + 1}", text, w, h, steps, seed)
            draw_boxes(img, text, os.path.join(OUT, f"magic_{i + 1}_boxes.png"))
            r["render"] = rr
            magic.append(r)
        except Exception as e:
            magic.append({"idea": idea, "error": f"{type(e).__name__}: {e}",
                          "trace": traceback.format_exc()})
            _say(f"magic {i + 1} FAILED: {type(e).__name__}: {e}")
        RESULTS["steps"]["2_magic"] = magic
        _save()

    # 3. Describe the hand logo
    try:
        from PIL import Image
        src = os.path.join(OUT, "hand_logo.png")
        img = Image.open(src)
        t0 = time.time()
        text, warnings = cz_magic.describe(img)
        dt = time.time() - t0
        _write("describe_logo", text)
        draw_boxes(img, text, os.path.join(OUT, "describe_logo_boxes.png"))
        found = [t for t in ("BEAN THERE",) if t.lower() in text.lower()]
        RESULTS["steps"]["3_describe"] = {"seconds": round(dt, 1), "warnings": warnings,
                                          "texts_read_back": found,
                                          "items": len(json.loads(text)["items"])}
        _say(f"describe: {dt:.1f} s, {len(warnings)} note(s), texts read back: {found}")
    except Exception as e:
        RESULTS["steps"]["3_describe"] = {"error": f"{type(e).__name__}: {e}",
                                          "trace": traceback.format_exc()}
        _say(f"describe FAILED: {type(e).__name__}: {e}")
    _save()

    # 4. "No other text" on the plain coffee logo (the lot 1 finding), optional
    if "--no-other-text" in sys.argv:
        rows = []
        for i in range(4):
            for tag, suffix in (("plain", ""), ("plainnot", " No other text.")):
                try:
                    _, r = render(P, f"{tag}_s{seed + i}", PLAIN_LOGO + suffix, 1024, 1024,
                                  steps, seed + i)
                    rows.append({"form": tag, "seed": seed + i, **r})
                except Exception as e:
                    rows.append({"form": tag, "seed": seed + i, "error": str(e)})
                RESULTS["steps"]["4_no_other_text"] = rows
                _save()
        _say("no-other-text: 8 renders to compare by eye (plain_* vs plainnot_*)")

    RESULTS["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _save()
    _say(f"done: {os.path.relpath(os.path.join(OUT, 'report.md'), ROOT)}")


def _report():
    s = RESULTS["steps"]
    L = ["# Lot 2 on the GPU: layout drafts, magic prompt, Describe", "",
         f"Started {RESULTS['started']}" + (f", finished {RESULTS['finished']}"
                                            if "finished" in RESULTS else ", NOT finished"),
         "", f"- environment: {s.get('0_env')}", ""]
    hand = s.get("1_hand") or {}
    if hand:
        L += ["## Hand-built layouts (presets with real texts)", "",
              "| render | size | s | warnings | texts | image |", "|---|---|---|---|---|---|"]
        for k, v in hand.items():
            if "error" in v:
                L.append(f"| {k} | ERROR {v['error']} | | | | |")
            else:
                L.append(f"| {k} | {v['size']} | {v['seconds']} | {len(v['warnings'])} | "
                         f"{v['texts']} | `{v['image']}` (+ `_boxes.png`) |")
        L.append("")
    magic = s.get("2_magic") or []
    if magic:
        L += ["## Magic prompt", "", "| idea | engine | s | valid | texts | render s |",
              "|---|---|---|---|---|---|"]
        for m in magic:
            if "error" in m:
                L.append(f"| {m['idea'][:40]}... | ERROR {m['error']} | | | | |")
            else:
                L.append(f"| {m['idea'][:40]}... | {m['engine']} | {m['seconds']} | "
                         f"{m['valid']} | {m['texts']} | "
                         f"{(m.get('render') or {}).get('seconds', '')} |")
        L.append("")
    d = s.get("3_describe")
    if d:
        L += ["## Describe (local)", "", "```", json.dumps(d, indent=1, ensure_ascii=False),
              "```", ""]
    rows = s.get("4_no_other_text")
    if rows:
        L += ["## Plain text with and without \"No other text.\" (judge by eye)", "",
              "| form | seed | s | image |", "|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['form']} | {r['seed']} | {r.get('seconds', r.get('error'))} | "
                     f"`{r.get('image', '')}` |")
        L.append("")
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
