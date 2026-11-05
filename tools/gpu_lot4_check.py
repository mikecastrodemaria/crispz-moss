"""Lot 4 on the GPU: MOSS-Image LoRAs through crispz-moss's own loader (cz_moss_lora).

The acceptance of lot 4 is on the CPU (tests/test_moss_lora.py: every published layout,
the exact arithmetic, nothing applied in silence); this script shows two community LoRAs
on the real weights, downloaded on your say-so (both Apache 2.0):

  0. BASELINE   the Poster preset's layout, 1152x1632, 50 steps, CFG 3, seed 4242.
  1. DEBAND     MATLOWAI/MOSS-Image-SFT-DeBand, moss_image_sft_banding_fix_L37-41_bf16
                (ComfyUI "diff" layout, fused qkv): the same render at 1.0. Compare the
                smooth areas (sky, backdrop): the SFT's faint vertical stripes should go.
  2. DMD LoRA   lodestones/MOSS-Image-SFT-8-steps-dmd lora.safetensors (kohya layout,
                rank 128, fp32, 1.2 GB): the same layout at 8 steps / CFG 1 (the author's
                recipe also swaps the time embedding and samples with a data-space loop:
                NOT reproduced here, so this is "the LoRA alone on the release scheduler",
                an experiment, judge the texts) and at 12 steps / CFG 3 for comparison.
                --skip-dmd skips the download and the renders.
  3. DRAFT      the base model at 12 steps / CFG 3, the Draft preset, same seed.

Each render goes to bench/lot4/ with its time and VRAM peak; report.md lists them. The
LoRA files land in the app's loras/ folder (loras_dir), where the UI lists them too.
THE GPU MUST BE FREE (same check as tools/gpu_lot1_check.py).

Usage:  .venv\\Scripts\\python tools\\gpu_lot4_check.py [--steps 50] [--skip-dmd]
"""
import json
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
OUT = os.path.join(ROOT, "bench", "lot4")
RESULTS = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": {}}
FILES = {
    "deband": ("MATLOWAI/MOSS-Image-SFT-DeBand", "moss_image_sft_banding_fix_L37-41_bf16.safetensors",
               "Apache 2.0 (model card)"),
    "dmd": ("lodestones/MOSS-Image-SFT-8-steps-dmd", "lora.safetensors", "Apache 2.0 (model card)"),
}


def _say(msg):
    print(f"[lot4] {msg}", flush=True)


def _save():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False)
    lines = ["# Lot 4 on the GPU: MOSS-Image LoRAs", "",
             f"Started {RESULTS['started']}" + (f", finished {RESULTS['finished']}"
                                                if "finished" in RESULTS else ", NOT finished"),
             "", "| step | seconds | peak GB | LoRA report | image / error |", "|---|---|---|---|---|"]
    for k, v in RESULTS["steps"].items():
        lines.append(f"| {k} | {v.get('seconds', '')} | {v.get('peak_reserved_gb', '')} | "
                     f"{v.get('lora', '')} | {v.get('image') or v.get('error', '')} |")
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _fetch(key):
    """The LoRA file into loras_dir (hf_hub_download, once), its licence said."""
    import cz_pipeline as P
    from huggingface_hub import hf_hub_download
    repo, name, lic = FILES[key]
    dst_dir = P.LORAS_DIR
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, name)
    if os.path.isfile(dst):
        _say(f"{name}: already in {dst_dir}")
        return dst
    _say(f"downloading {repo}/{name} ({lic}) into {dst_dir} ...")
    got = hf_hub_download(repo_id=repo, filename=name, local_dir=dst_dir)
    return got if os.path.isfile(got) else dst


def _step(tag, fn, lora=""):
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
            "lora": lora, "image": os.path.relpath(path, ROOT)}
        _say(f"{tag}: {img.width}x{img.height} in {time.time() - t0:.1f} s")
        return img
    except Exception as e:
        RESULTS["steps"][tag] = {"error": f"{type(e).__name__}: {e}", "lora": lora,
                                 "trace": traceback.format_exc()}
        _say(f"{tag} FAILED: {type(e).__name__}: {e}")
        return None
    finally:
        _save()


def _set(P, loras):
    """Applies exactly these (path, weight) on the loaded pipe; the loader's summary."""
    P.LORAS = list(loras)
    pipe = P.get_pipe("txt2img")
    ok = P._apply_loras(pipe, force=True)
    rep = ", ".join(f"{os.path.basename(p)}@{w:g}" for p, w in pipe.applied_loras) or "none"
    return f"{'ok' if ok else 'FAILED'}: {rep}"


def main():
    import gpu_lot1_check as G1
    G1.OUT = OUT
    os.makedirs(OUT, exist_ok=True)
    steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 50
    P = G1.prechecks()
    import cz_design
    preset = "Poster (A-series, portrait)"
    prompt = cz_design.prompt_of(preset)
    w, h = cz_design.size_of(preset)
    seed = 4242
    old_guidance = P.GUIDANCE

    _step("0_baseline", lambda: P.generate(prompt, w, h, steps, seed), lora=_set(P, []))
    try:
        deband = _fetch("deband")
        rep = _set(P, [(deband, 1.0)])
        _step("1_deband_L37-41", lambda: P.generate(prompt, w, h, steps, seed), lora=rep)
    except Exception as e:
        RESULTS["steps"]["1_deband_L37-41"] = {"error": f"{type(e).__name__}: {e}"}
        _say(f"deband FAILED: {e}")
    if "--skip-dmd" not in sys.argv:
        try:
            dmd = _fetch("dmd")
            rep = _set(P, [(dmd, 1.0)])
            P.GUIDANCE = 1.0
            _step("2_dmd_lora_8steps_cfg1", lambda: P.generate(prompt, w, h, 8, seed), lora=rep)
            P.GUIDANCE = 3.0
            _step("2_dmd_lora_12steps_cfg3", lambda: P.generate(prompt, w, h, 12, seed), lora=rep)
        except Exception as e:
            RESULTS["steps"]["2_dmd_lora"] = {"error": f"{type(e).__name__}: {e}"}
            _say(f"dmd FAILED: {e}")
        finally:
            P.GUIDANCE = old_guidance
    _set(P, [])
    P.GUIDANCE = 3.0
    _step("3_draft_12steps", lambda: P.generate(prompt, w, h, 12, seed), lora="none")
    P.GUIDANCE = old_guidance
    RESULTS["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _save()
    _say(f"done: {os.path.relpath(os.path.join(OUT, 'report.md'), ROOT)}")


if __name__ == "__main__":
    main()
