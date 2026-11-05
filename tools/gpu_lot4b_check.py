"""Lot 4b on the GPU: editing from references with the Moss-Image-Edit LoRA.

The acceptance of lot 4b is on the CPU (tests/test_moss_edit.py: the reference prefix
equals the vendored forward without references, the cache, the adapter, the app's path);
this script runs the real thing on the 5090, with the lot 2 poster as the image to edit:

  0. EDIT 1 REF      "Replace the bread in the basket with golden croissants, keep
                     everything else identical, the texts included." (50 steps, CFG 4:
                     the author's inference defaults), output = the poster's size.
  1. EDIT CFG 3      the same at CFG 3 (the app's default slider).
  2. EDIT 2 REFS     the poster + the lot 2 logo as a second reference: "Add the coffee
                     logo of the second image in the top right corner of the poster,
                     keep the rest identical."
  3. EDIT 512        step 0 again with references read at 512 px on their long side
                     (--ref-edge, ComfyUI's node default): time and fidelity to compare.
  4. BACKGROUND      the logo alone: "Change the background to a sunny beach, keeping
                     the logo." (the author's own example).
  5. NO LORA         a control: the same call with the LoRA weight at 0 is REFUSED by
                     name (nothing rendered): the base weights never saw a prefix.

Each render goes to bench/lot4b/ with its time, VRAM peak and prefix size; report.md
lists them. The LoRA (Comfy-Org/MOSS-Image loras/moss_image_edit_lora_rank_64_bf16,
about 300 MB; its author states NO licence for the weights) is fetched once into
<loras_dir>/_hf-edit/moss-image-edit.safetensors by the app's own preset code.
THE GPU MUST BE FREE (same check as tools/gpu_lot1_check.py).

Usage:  .venv\\Scripts\\python tools\\gpu_lot4b_check.py [--steps 50] [--ref-edge 512]
                                                       [--skip-2refs] [--skip-512]
"""
import json
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
OUT = os.path.join(ROOT, "bench", "lot4b")
POSTER = os.path.join(ROOT, "bench", "lot2", "hand_poster.png")
LOGO = os.path.join(ROOT, "bench", "lot2", "hand_logo.png")
RESULTS = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": {}}
PROMPT_1 = ("Replace the bread in the basket with golden croissants, keep everything else "
            "identical, the texts included.")
PROMPT_2 = ("Add the coffee logo of the second image in the top right corner of the poster, "
            "keep the rest identical.")
PROMPT_BG = "Change the background to a sunny beach, keeping the logo."


def _say(msg):
    print(f"[lot4b] {msg}", flush=True)


def _save():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False)
    lines = ["# Lot 4b on the GPU: editing from references (Moss-Image-Edit LoRA)", "",
             f"Started {RESULTS['started']}" + (f", finished {RESULTS['finished']}"
                                                if "finished" in RESULTS else ", NOT finished"),
             "", f"LoRA: {RESULTS.get('lora', '?')}", "",
             "| step | seconds | peak GB | refs (px) | prefix tokens | image / error |",
             "|---|---|---|---|---|---|"]
    for k, v in RESULTS["steps"].items():
        lines.append(f"| {k} | {v.get('seconds', '')} | {v.get('peak_reserved_gb', '')} | "
                     f"{v.get('refs', '')} | {v.get('tokens', '')} | "
                     f"{v.get('image') or v.get('error', '')} |")
    lines += ["", "To judge on the images: is the asked change made and ONLY it (texts, "
                  "layout, colours of the rest untouched)? Does the second reference's logo "
                  "come through? 1024 against 512 px references: fidelity against time."]
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _step(tag, fn, refs=""):
    import torch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    _LOG.clear()
    try:
        img = fn()
        path = os.path.join(OUT, f"{tag}.png")
        img.save(path)
        tokens = ""
        for line in _LOG:
            if "prefix tokens" in line:
                tokens = line.split("(")[1].split(" prefix")[0]
        RESULTS["steps"][tag] = {
            "seconds": round(time.time() - t0, 1), "size": f"{img.width}x{img.height}",
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2),
            "refs": refs, "tokens": tokens, "image": os.path.relpath(path, ROOT)}
        _say(f"{tag}: {img.width}x{img.height} in {time.time() - t0:.1f} s")
        return img
    except Exception as e:
        RESULTS["steps"][tag] = {"error": f"{type(e).__name__}: {e}", "refs": refs,
                                 "trace": traceback.format_exc()}
        _say(f"{tag} FAILED: {type(e).__name__}: {e}")
        return None
    finally:
        _save()


_LOG = []


def main():
    import gpu_lot1_check as G1
    G1.OUT = OUT
    os.makedirs(OUT, exist_ok=True)
    args = sys.argv
    steps = int(args[args.index("--steps") + 1]) if "--steps" in args else 50
    edge = int(args[args.index("--ref-edge") + 1]) if "--ref-edge" in args else 512
    for p in (POSTER, LOGO):
        if not os.path.isfile(p):
            _say(f"missing {p}: run tools/gpu_lot2_check.py first (it renders the poster "
                 f"and the logo this script edits)")
            return
    from PIL import Image
    P = G1.prechecks()
    import cz_core
    import cz_edit_loras
    real_log = cz_core._log

    def spy(msg):
        _LOG.append(str(msg))
        real_log(msg)
    cz_core._log = spy
    P._log = spy
    poster = Image.open(POSTER).convert("RGB")
    logo = Image.open(LOGO).convert("RGB")
    seed = 4242
    old_guidance = P.GUIDANCE
    P.EDIT_LORAS, P.EDIT_LORAS_ENABLED = [], True
    _say(f"edit LoRA: {cz_edit_loras.EDIT_PRESET} "
         f"({'on disk' if cz_edit_loras.edit_ready() else 'to download, about 300 MB'}); "
         f"{cz_edit_loras.EDIT_LICENCE}")
    P.get_pipe("txt2img")
    size = f"{poster.width}x{poster.height}"
    P.GUIDANCE = 4.0
    _step("0_edit_1ref_cfg4", lambda: P.generate_omni([poster], PROMPT_1, "", 0, 0, steps, seed),
          refs=f"poster {size} @1024")
    RESULTS["lora"] = ", ".join(f"{os.path.basename(p)}@{w:g}" for p, w in P._APPLIED_EDIT_LORAS)
    P.GUIDANCE = 3.0
    _step("1_edit_1ref_cfg3", lambda: P.generate_omni([poster], PROMPT_1, "", 0, 0, steps, seed),
          refs=f"poster {size} @1024")
    P.GUIDANCE = 4.0
    if "--skip-2refs" not in args:
        _step("2_edit_2refs_logo", lambda: P.generate_omni([poster, logo], PROMPT_2, "", 0, 0, steps, seed),
              refs=f"poster {size} + logo {logo.width}x{logo.height} @1024")
    if "--skip-512" not in args:
        os.environ["MOSS_EDIT_REF_MAX_EDGE"] = str(edge)
        _step(f"3_edit_1ref_refs{edge}", lambda: P.generate_omni([poster], PROMPT_1, "", 0, 0, steps, seed),
              refs=f"poster {size} @{edge}")
        os.environ.pop("MOSS_EDIT_REF_MAX_EDGE", None)
    _step("4_logo_background", lambda: P.generate_omni([logo], PROMPT_BG, "", 0, 0, steps, seed),
          refs=f"logo {logo.width}x{logo.height} @1024")
    # the control: weight 0 = no edit LoRA in effect -> a named refusal, nothing rendered
    pipe = P.get_pipe("txt2img")
    path = cz_edit_loras.resolve(cz_edit_loras.EDIT_PRESET)
    P.EDIT_LORAS = [(path, 0.0)]
    try:
        pipe.set_loras([(path, 0.0)])
        pipe(prompt=PROMPT_1, image=poster, num_inference_steps=1)
        RESULTS["steps"]["5_no_lora_control"] = {"error": "NOT refused (unexpected)"}
    except NotImplementedError as e:
        RESULTS["steps"]["5_no_lora_control"] = {"image": f"refused as expected: {str(e)[:90]}..."}
        _say("no-LoRA control: refused by name, as expected")
    finally:
        P.EDIT_LORAS = []
        P.GUIDANCE = old_guidance
        cz_core._log = real_log
        P._log = real_log
    RESULTS["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _save()
    _say(f"done: {os.path.relpath(os.path.join(OUT, 'report.md'), ROOT)}")


if __name__ == "__main__":
    main()
