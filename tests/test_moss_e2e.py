"""MOSS-Image on the real GPU, end to end, through the app's own functions (lot 6):

    load       -> _ensure_base()    the MossPipe (bf16 DiT, Qwen3.5-9B encoder, FLUX.2 VAE)
    gen 1024   -> generate()        the Logo preset's layout, 50 steps (a text in a box)
    gen 2 Mpx  -> generate()        1456x1456, plain text (CZ_E2E_2MPX=0 skips it)
    img2img    -> _refine_whole()   emulated SDEdit, strength 0.35
    inpaint    -> inpaint_run()     emulated latent blending; outside the mask the
                                    original comes back pixel for pixel
    stop       -> an interrupt at step 3 still returns an image

A GPU run, not a unit test: it loads the real public weights (27 GB on the first run)
and SKIPS cleanly without CUDA, without access to the repo, or when CZ_E2E=0. GPU free
first (close crispz-ideogram, crispz-flux3, ComfyUI, Ollama). Images in
tests/e2e_*.png (gitignored), timings and VRAM peaks in tests/e2e_timings.txt.

    MOSS_E2E_MODEL   another base repo or local folder (default OpenMOSS-Team/MOSS-Image-SFT)
    CZ_E2E=0         skip even with a GPU
    CZ_E2E_2MPX=0    skip the 2 Mpx render

Run:  .venv\\Scripts\\python -m pytest tests\\test_moss_e2e.py -s
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

REPO = os.environ.get("MOSS_E2E_MODEL") or "OpenMOSS-Team/MOSS-Image-SFT"
# Pinned before cz_pipeline is imported: the run must not follow the last model or
# single file chosen in the UI (preferences.json).
os.environ["MOSS_MODEL"] = REPO
os.environ.pop("MOSS_TRANSFORMER", None)

OUT = os.path.dirname(os.path.abspath(__file__))
STEPS = 50
PLAIN = ("A bold minimalist bakery poster on warm cream paper, a wicker basket of "
         "croissants in the centre, the headline \"GRAND OPENING\" at the top and the line "
         "\"SATURDAY 9 AM\" at the bottom, flat graphic design. No other text.")


def _why_skip():
    if os.environ.get("CZ_E2E", "1") == "0":
        return "CZ_E2E=0"
    try:
        import torch
    except Exception as e:
        return f"torch unavailable: {e}"
    if not torch.cuda.is_available():
        return "needs CUDA (27 GB of bf16 weights)"
    if os.path.isdir(REPO):
        return None
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(REPO, "model_index.json")
    except Exception as e:
        return f"no access to {REPO}: {type(e).__name__}"
    return None


def _peak():
    import torch
    return torch.cuda.max_memory_reserved() / 1024 ** 3


def _step(name, fn, log):
    import torch
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    out = fn()
    line = f"  {name:<26} {time.time() - t:7.1f} s   peak={_peak():5.2f} GB"
    print(line, flush=True)
    log.append(line)
    return out


@pytest.mark.skipif(_why_skip() is not None, reason=_why_skip() or "")
def test_moss_end_to_end():
    from PIL import ImageDraw, Image
    import cz_design
    import cz_moss
    import cz_pipeline as p
    log = []
    print(f"\nbase repo: {p.BASE_REPO}  offload: {p.OFFLOAD_MODE}  encoder: {p.PRECISION}")

    pipe = _step("load", p._ensure_base, log)
    assert isinstance(pipe, cz_moss.MossPipe), type(pipe).__name__

    logo = cz_design.prompt_of("Logo (square)")                 # a layout, drawn for 1024
    img = _step("gen 1024 (layout)", lambda: p.generate(logo, 1024, 1024, STEPS, 7), log)
    img.save(os.path.join(OUT, "e2e_1_gen1024.png"))
    assert img.size == (1024, 1024)
    assert img.getextrema() != ((0, 0), (0, 0), (0, 0)), "a black image"

    if os.environ.get("CZ_E2E_2MPX", "1") != "0":
        big = _step("gen 1456 (2 Mpx)", lambda: p.generate(PLAIN, 1456, 1456, STEPS, 7), log)
        big.save(os.path.join(OUT, "e2e_2_gen2mpx.png"))
        assert big.size == (1456, 1456)

    r = _step("img2img 0.35", lambda: p._refine_whole(p.get_pipe("img2img"), img, 0.35,
                                                     STEPS, logo, 7), log)
    r.save(os.path.join(OUT, "e2e_3_img2img.png"))
    assert r.size == img.size

    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rectangle((320, 420, 704, 700), fill=255)
    ip = _step("inpaint", lambda: p.inpaint_run(
        img, mask, "a large red retro bicycle, flat vector illustration", STEPS, 1.0, 7),
        log)
    ip.save(os.path.join(OUT, "e2e_4_inpaint.png"))
    assert ip.size == img.size
    assert ip.getpixel((40, 980)) == img.getpixel((40, 980)), "outside the mask is kept"

    calls = []
    real_cb = p._step_end_kwargs

    def with_stop(*a, **k):
        def cb(pipe_, i, t, kw):
            calls.append(i)
            if i == 2:
                pipe_._interrupt = True
            return kw
        return {"callback_on_step_end": cb}
    p._step_end_kwargs = with_stop
    try:
        s = _step("stop at step 3", lambda: p.generate(PLAIN, 1024, 1024, STEPS, 7), log)
    finally:
        p._step_end_kwargs = real_cb
    assert calls[:3] == [0, 1, 2] and len(calls) == 3, calls
    assert s.size == (1024, 1024)

    report = os.path.join(OUT, "e2e_timings.txt")
    with open(report, "w", encoding="utf-8") as f:
        f.write(f"{p.BASE_REPO} offload={p._effective_offload()} encoder={p.PRECISION}\n"
                + "\n".join(log) + "\n")
    print(f"\ntimings written to {report}")


if __name__ == "__main__":
    why = _why_skip()
    if why:
        print(f"SKIP test_moss_e2e: {why}")
        sys.exit(0)
    test_moss_end_to_end()
    print("ALL OK")
