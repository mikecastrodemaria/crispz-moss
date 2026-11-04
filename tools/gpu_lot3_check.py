"""Lot 3 on the GPU: emulated img2img, inpaint, outpaint, reframe, refine (MOSS-Image).

The acceptance of lot 3 is on the CPU (tests/test_moss_emulated.py); this script shows
what the emulation gives on the real weights, through the app's own functions:

  0. INIT       bench/lot2/hand_poster.png (+ hand_poster.json, its layout draft) when
                they exist, else a fresh render of the Poster preset.
  1. IMG2IMG    the init at denoise 0.3 and 0.6 (P._refine_whole), same layout prompt.
  2. INPAINT    a box in the middle repainted with another subject (P.inpaint_run).
  3. OUTPAINT   the init extended to the right (P.outpaint_directions) with its layout
                draft, PLACED on the source (lot 3b); --plain uses the preset's summary
                caption instead (the first pass did: invented text in the new areas).
  4. REFRAME    the init reframed to 16:9 (P.reframe, contain), same prompt.
  5. REFINE A/B denoise 0.3 at 12 then 36 steps, same seed (the lot 7 question of
                crispz-ideogram, to re-measure on MOSS: 50 nominal steps).
  6. DRAFT      the Poster layout at 12, 20, 25 and 50 steps, same seed: which "Draft"
                preset is worth keeping (judge the texts by eye). --skip-draft skips it.

Each output is saved in bench/lot3/ with its time; report.md lists them. Look at them:
the numbers say nothing about whether the text survived a 0.6 img2img.
THE GPU MUST BE FREE (same check as tools/gpu_lot1_check.py).

Usage:  .venv\\Scripts\\python tools\\gpu_lot3_check.py [--steps 50] [--skip-draft] [--plain]
"""
import json
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
OUT = os.path.join(ROOT, "bench", "lot3")
RESULTS = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": {}}


def _say(msg):
    print(f"[lot3] {msg}", flush=True)


def _save():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False)
    lines = ["# Lot 3 on the GPU: emulated img2img / inpaint (MOSS-Image)", "",
             f"Started {RESULTS['started']}" + (f", finished {RESULTS['finished']}"
                                                if "finished" in RESULTS else ", NOT finished"),
             "", "| step | seconds | peak GB | size | image / error |", "|---|---|---|---|---|"]
    for k, v in RESULTS["steps"].items():
        lines.append(f"| {k} | {v.get('seconds', '')} | {v.get('peak_reserved_gb', '')} | "
                     f"{v.get('size', '')} | {v.get('image') or v.get('error', '')} |")
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _step(tag, fn):
    import torch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        img = fn()
        path = os.path.join(OUT, f"{tag}.png")
        img.save(path)
        RESULTS["steps"][tag] = {
            "seconds": round(time.time() - t0, 1), "size": f"{img.width}x{img.height}",
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2),
            "image": os.path.relpath(path, ROOT)}
        _say(f"{tag}: {img.width}x{img.height} in {time.time() - t0:.1f} s")
        return img
    except Exception as e:
        RESULTS["steps"][tag] = {"error": f"{type(e).__name__}: {e}",
                                 "trace": traceback.format_exc()}
        _say(f"{tag} FAILED: {type(e).__name__}: {e}")
        return None
    finally:
        _save()


def main():
    import gpu_lot1_check as G1
    from PIL import Image, ImageDraw
    G1.OUT = OUT
    os.makedirs(OUT, exist_ok=True)
    steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 50
    P = G1.prechecks()
    import cz_design

    preset = "Poster (A-series, portrait)"
    caption = cz_design.PRESETS[preset]["caption"]            # plain text, no boxes
    hand = os.path.join(ROOT, "bench", "lot2", "hand_poster.png")
    hand_json = os.path.join(ROOT, "bench", "lot2", "hand_poster.json")
    if os.path.isfile(hand) and os.path.isfile(hand_json):
        init = Image.open(hand).convert("RGB")
        prompt = open(hand_json, encoding="utf-8").read()
        _say(f"init: {hand} ({init.width}x{init.height}), prompt: its layout draft")
    else:
        prompt = cz_design.prompt_of(preset)
        w, h = cz_design.size_of(preset)
        init = _step("0_init", lambda: P.generate(prompt, w, h, steps, 12345))
        if init is None:
            sys.exit(1)

    _step("1_img2img_0.3", lambda: P._refine_whole(P.get_pipe("img2img"), init, 0.3, steps,
                                                   prompt, 7))
    _step("1_img2img_0.6", lambda: P._refine_whole(P.get_pipe("img2img"), init, 0.6, steps,
                                                   prompt, 7))
    w, h = init.size
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rectangle([w // 4, h // 3, 3 * w // 4, 2 * h // 3], fill=255)
    mask.save(os.path.join(OUT, "2_inpaint_mask.png"))
    _step("2_inpaint", lambda: P.inpaint_run(
        init, mask, "a large red retro bicycle, flat vector illustration", steps, 1.0, 7))
    fill_prompt = caption if "--plain" in sys.argv else prompt
    _step("3_outpaint_right", lambda: P.outpaint_directions(init, None, ["right"], fill_prompt,
                                                            steps, 7))
    _step("4_reframe_16x9", lambda: P.reframe(init, 16, 9, "contain", fill_prompt, steps, 7))
    # 5. refine A/B: the same denoise at 12 and 36 steps (lot 7 of crispz-ideogram found
    # them identical to the eye on Ideogram 4; MOSS runs 50 nominal steps, to re-check)
    for n in (12, 36):
        _step(f"5_refine_0.3_{n}steps", lambda n=n: P._refine_whole(
            P.get_pipe("img2img"), init, 0.3, n, prompt, 11))
    # 6. a Draft preset worth keeping? the layout at fewer steps, same seed
    if "--skip-draft" not in sys.argv:
        lp = cz_design.prompt_of(preset)
        lw, lh = cz_design.size_of(preset)
        for n in (12, 20, 25, 50):
            _step(f"6_draft_{n}steps", lambda n=n: P.generate(lp, lw, lh, n, 4242))
    RESULTS["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _save()
    _say(f"done: {os.path.relpath(os.path.join(OUT, 'report.md'), ROOT)}")


if __name__ == "__main__":
    main()
