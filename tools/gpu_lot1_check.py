"""Lot 1 validation on the GPU: MOSS-Image through the app's own code path, then the
text A/B against the crispz-ideogram posters.

What it measures and checks, in this order (each step is written to the report as soon
as it is done, so a crash still leaves the earlier figures):

  0. PRECHECKS   no other process on the GPU (another crispz open = abort), CUDA, the
                 versions, the SDPA kernels at head_dim 128, access to the (public) repo.
  1. LOADING     cz_pipeline.get_pipe("txt2img") from an empty state: time, VRAM at rest
                 (torch allocated / reserved, and the driver's figure).
  2. RENDERS     1024x1024 twice (cold, then warm = the steady state), 1456x1456 (2 Mpx,
                 the top of the recommended range), 2048x1152. Time, s/step, VRAM peak,
                 denoise peak against VAE-decode peak. 50 steps, CFG 3.0.
  3. PREVIEW     the live-preview frames of the warm 1024 render (FLUX.2 matrix on the
                 (B, 128, h, w) latents) against the final image: mean error on 0-255.
  4. STOP        a 1024 render stopped from another thread through request_stop(), as
                 the UI's Stop button does: delay before the call returns, and an image
                 must still come back.
  5. ENCODER     --nf4: the text encoder reloaded in nf4 (bitsandbytes): load time,
                 VRAM at rest, the warm 1024 render again, pixel difference against bf16.
  6. A/B TEXT    the three crispz-ideogram lot-2 designs (bakery poster "GRAND OPENING",
                 coffee logo "BEAN THERE", Swiss poster) rendered by MOSS as PLAIN TEXT and
                 as a LAYOUT JSON, N seeds each (--ab-seeds, 4 by default), at the sizes
                 Ideogram used. Images in bench/lot1/ab/, a table in bench/lot1/ab.md with
                 the time per image and the columns Mike fills in by eye (exact text,
                 ghost text in the background, banding). When ../crispz-ideogram/bench/lot2
                 exists (or --ideogram-bench DIR), the Ideogram renders are linked next
                 to them.

Everything lands in bench/lot1/ (images, frames, results.json, report.md, ab.md).

THE GPU MUST BE FREE: close crispz-ideogram, crispz-flux3 and any other crispz first. An
app that has not rendered yet shows nothing in nvidia-smi and still loads its model at
the first click: closing it is the only safe way (FORK-flux3-CLAUDE.md §3).

Usage (Windows, from the repo root):
    .venv\\Scripts\\python tools\\gpu_lot1_check.py
    .venv\\Scripts\\python tools\\gpu_lot1_check.py --offload model --skip-2mpx --ab-seeds 2

  --offload MODE       none | model | sequential | auto (default: the app's setting)
  --steps N            50 by default (the official setting)
  --seed N             12345 by default
  --skip-2mpx          skips the 1456x1456 and 2048x1152 renders
  --skip-ab            skips the text A/B
  --ab-seeds N         seeds per design and per prompt form in the A/B (4 by default)
  --ab-only            loads, then the A/B alone
  --nf4                also measures the text encoder in nf4 (a reload)
  --ideogram-bench DIR the crispz-ideogram bench/lot2 folder (for the A/B links)
  --force              runs even if another process holds the GPU (you have been warned)
"""
import json
import os
import subprocess
import sys
import threading
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "bench", "lot1")
AB_OUT = os.path.join(OUT, "ab")

PROMPT = ('A bold retro poster for a bakery opening. Large headline text "GRAND OPENING" '
          'at the top in thick cream serif letters, below it smaller text "SATURDAY 9 AM" '
          'in a red ribbon banner, a basket of golden croissants in the center, warm '
          'orange and teal palette, clean flat vector illustration style.')
NEGATIVE = ""

# --- The A/B designs: crispz-ideogram's lot 2 posters, in MOSS's two prompt forms --------
# Layout JSON (USERGUIDE.md): bbox = [y_min, x_min, y_max, x_max] in PIXELS, origin at the
# canvas centre, y down. The sizes are the ones Ideogram rendered in lot 2 (hand.png
# 720x1024, the logo 1024x1024, the Swiss poster 720x1024).


def _box(w, h, x0, y0, x1, y1):
    """Top-left pixel coordinates -> MOSS's centred [y_min, x_min, y_max, x_max]."""
    return [int(y0 - h / 2), int(x0 - w / 2), int(y1 - h / 2), int(x1 - w / 2)]


def designs():
    d = []
    w, h = 720, 1024
    d.append({
        "name": "poster_bakery", "width": w, "height": h,
        "texts": ["GRAND OPENING", "SATURDAY 9 AM", "RUE DU PAIN"],
        "plain": PROMPT.replace('"SATURDAY 9 AM"', '"SATURDAY 9 AM · RUE DU PAIN"'),
        "layout": {
            "summary": {"meta": ["poster"], "canvas": "single",
                        "caption": "A bold retro poster for a bakery opening: thick cream "
                                   "serif headline on a teal background, a red ribbon banner "
                                   "with the date, a basket of golden croissants, warm orange "
                                   "and teal palette, clean flat vector illustration style."},
            "items": [
                {"label": "text", "caption": "Huge cream serif headline, two words on one "
                                             "line, centered at the top.",
                 "text": "GRAND OPENING", "bbox": _box(w, h, 40, 60, 680, 220)},
                {"label": "text", "caption": "Smaller white sans-serif text inside a red "
                                             "ribbon banner under the headline.",
                 "text": "SATURDAY 9 AM · RUE DU PAIN", "bbox": _box(w, h, 90, 250, 630, 330)},
                {"label": "object", "caption": "A wicker basket full of golden croissants, "
                                               "flat vector style, centered.",
                 "bbox": _box(w, h, 110, 400, 610, 900)},
            ],
            "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}})
    w = h = 1024
    d.append({
        "name": "logo_coffee", "width": w, "height": h,
        "texts": ["BEAN THERE"],
        "plain": 'A vintage coffee shop logo with the words "BEAN THERE" curved around a '
                 'steaming cup of coffee, circular badge, two-colour print, dark brown on '
                 'cream, clean vector emblem on a plain cream background.',
        "layout": {
            "summary": {"meta": ["logo"], "canvas": "single",
                        "caption": "A vintage circular coffee shop badge logo, dark brown "
                                   "ink on a plain cream background, two-colour print look, "
                                   "clean vector emblem."},
            "items": [
                {"label": "text", "caption": "Bold condensed serif words curved along the "
                                             "top of the badge.",
                 "text": "BEAN THERE", "bbox": _box(w, h, 200, 150, 824, 330)},
                {"label": "object", "caption": "A steaming coffee cup in the centre of the "
                                               "badge, three curls of steam, flat vector.",
                 "bbox": _box(w, h, 330, 360, 694, 760)},
            ],
            "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}})
    w, h = 720, 1024
    d.append({
        "name": "poster_swiss", "width": w, "height": h,
        "texts": ["5B", "MOSS, real typography"],
        "plain": 'A minimalist Swiss-style poster: a huge black "5B" on an off-white grid, '
                 'a red circle overlapping the numerals, a small caption "MOSS, real '
                 'typography" in the lower left corner, Helvetica-like typography, '
                 'International Typographic Style.',
        "layout": {
            "summary": {"meta": ["poster"], "canvas": "single",
                        "caption": "A minimalist Swiss-style poster on an off-white paper "
                                   "with a faint grid, a red circle overlapping huge black "
                                   "numerals, International Typographic Style."},
            "items": [
                {"label": "text", "caption": "Huge black grotesque numerals, bold, filling "
                                             "the upper two thirds.",
                 "text": "5B", "bbox": _box(w, h, 60, 120, 660, 620)},
                {"label": "object", "caption": "A flat red circle overlapping the numerals.",
                 "bbox": _box(w, h, 380, 380, 700, 700)},
                {"label": "text", "caption": "A small black sans-serif caption in the lower "
                                             "left corner, one line.",
                 "text": "MOSS, real typography", "bbox": _box(w, h, 60, 900, 420, 950)},
            ],
            "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}})
    return d


RESULTS = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "prompt": PROMPT, "steps": {}}


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
    _write_report()


def _say(msg):
    print(f"[lot1] {msg}", flush=True)


def _gb(n):
    return round(n / 1024 ** 3, 2)


def _smi_used_gb():
    """The driver's view of the VRAM in use (all processes), in GB, or None."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        used, total = (float(x) for x in out.strip().splitlines()[0].split(","))
        return round(used / 1024, 2), round(total / 1024, 2)
    except Exception:
        return None, None


def _gpu_processes():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout
        return [ln.strip() for ln in out.splitlines() if ln.strip()]
    except Exception as e:
        return [f"(nvidia-smi unavailable: {e})"]


BUSY_GB = 4.0
_MODEL_RUNTIMES = ("python", "comfy", "ollama", "crispz", "pinokio", "llama", "koboldcpp",
                   "lmstudio", "lm studio", "invokeai", "forge", "fooocus")


def _suspects(procs):
    out = []
    for p in procs:
        low = p.lower()
        mem = p.rsplit(",", 1)[-1].strip().lower()
        heavy = False
        if mem.endswith("mib"):
            try:
                heavy = float(mem[:-3]) > 1024
            except ValueError:
                pass
        if heavy or any(k in low for k in _MODEL_RUNTIMES):
            out.append(p)
    return out


# --- 0. Prechecks -----------------------------------------------------------------------
def prechecks():
    import torch
    r = {}
    procs = _gpu_processes()
    r["gpu_processes"] = procs
    suspects = _suspects(procs)
    used, total = _smi_used_gb()
    r["driver_used_at_start_gb"] = used
    busy = used is not None and used > BUSY_GB
    if (suspects or busy) and "--force" not in sys.argv:
        if suspects:
            _say("A MODEL PROCESS MAY HOLD THE GPU:")
            for p in suspects:
                _say(f"   {p}")
        if busy:
            _say(f"{used} GB of VRAM already in use (more than {BUSY_GB} GB): something "
                 "has a model loaded.")
        _say("Close crispz-ideogram / crispz-flux3 (and any other crispz / ComfyUI / "
             "Ollama), then run again. --force skips this check.")
        sys.exit(2)
    _say(f"GPU free enough: {used} GB in use at start, {len(procs)} process(es) listed, "
         "none of them a model runtime")
    if not torch.cuda.is_available():
        _say("CUDA is not available to this torch build: nothing to measure.")
        sys.exit(2)
    r["gpu"] = torch.cuda.get_device_name(0)
    r["torch"] = torch.__version__
    r["cuda"] = torch.version.cuda
    import diffusers
    import transformers
    r["diffusers"] = diffusers.__version__
    r["transformers"] = transformers.__version__
    try:
        import bitsandbytes as bnb
        r["bitsandbytes"] = bnb.__version__
    except Exception as e:
        r["bitsandbytes"] = f"not importable: {e}"
        if "--nf4" in sys.argv:
            _say(f"bitsandbytes does not import ({e}): --nf4 cannot run.")
            sys.exit(2)
    for mod in ("flash_linear_attention", "fla", "causal_conv1d", "triton"):
        try:
            __import__(mod)
            r[f"kernel_{mod}"] = "yes"
        except Exception:
            r[f"kernel_{mod}"] = "no (torch fallback)"
    r["sdpa_kernels"] = _sdpa_kernels()
    _say(f"SDPA kernels usable at head_dim 128 (MOSS-Image): {r['sdpa_kernels']}")
    import cz_pipeline as P
    try:
        with open(P._hw_profile_path(), encoding="utf-8") as f:
            prof = json.load(f)
        r["hw_profile"] = prof
        if "downgrade" in json.dumps(prof).lower():
            _say(f"NOTE: {P._hw_profile_path()} holds a cached offload downgrade: 'auto' "
                 "will start in 'model'. Delete the file (or pass --offload) to decide "
                 "afresh.")
    except FileNotFoundError:
        r["hw_profile"] = None
    except Exception as e:
        r["hw_profile"] = f"unreadable: {e}"
    repo = P.BASE_REPO
    r["repo"] = repo
    r["precision"] = P.PRECISION
    r["single_file_transformer"] = P.ZIMAGE_TRANSFORMER or None
    r["text_encoder"] = P.TEXT_ENCODER or None
    if not os.path.isdir(repo):
        try:
            from huggingface_hub import auth_check
            auth_check(repo, token=P._hf_token())
            r["repo_access"] = "OK"
        except Exception as e:
            r["repo_access"] = f"REFUSED: {type(e).__name__}: {e}"
            _say(f"No access to {repo}: {type(e).__name__}: {e} (the repo is public: a "
                 f"network or proxy problem?)")
            RESULTS["steps"]["0_prechecks"] = r
            _save()
            sys.exit(2)
    RESULTS["steps"]["0_prechecks"] = r
    _save()
    _say(f"prechecks OK: {r['gpu']}, torch {r['torch']} (CUDA {r['cuda']}), diffusers "
         f"{r['diffusers']}, transformers {r['transformers']}, repo {repo}")
    return P


def _sdpa_kernels():
    """Which SDPA kernels accept MOSS-Image's attention shape (bf16, 24 heads of 128), with
    and without a boolean mask. The pipeline passes no mask (one sample at a time at its
    true length), so the no-mask column is the one that runs."""
    import torch
    import torch.nn.functional as F
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception as e:
        return f"unknown ({e})"
    try:
        q = torch.randn(1, 24, 512, 128, device="cuda", dtype=torch.bfloat16)
        mask = torch.ones(1, 1, 512, 512, device="cuda", dtype=torch.bool)
    except Exception as e:
        return f"unknown ({type(e).__name__})"
    out = {}
    for name, be in (("flash", SDPBackend.FLASH_ATTENTION),
                     ("efficient", SDPBackend.EFFICIENT_ATTENTION),
                     ("cudnn", getattr(SDPBackend, "CUDNN_ATTENTION", None))):
        if be is None:
            continue
        for tag, m in (("", None), ("+mask", mask)):
            try:
                with sdpa_kernel(be):
                    F.scaled_dot_product_attention(q, q, q, attn_mask=m)
                out[name + tag] = "yes"
            except Exception:
                out[name + tag] = "no"
    del q, mask
    torch.cuda.empty_cache()
    return out


# --- Frames of the live preview: the app's callback, with a recorder on top --------------
class FrameRecorder:
    """Wraps cz_pipeline._store_preview: the app's own preview still runs; on top, at a
    few steps, the frame is kept."""

    def __init__(self, P, keep_steps):
        self.P = P
        self.keep = set(keep_steps)
        self.frames = {}
        self.orig = P._store_preview

    def __enter__(self):
        P = self.P
        orig = self.orig

        def rec(latents, step, total, width=0, height=0, pipe=None):
            orig(latents, step, total, width, height, pipe=pipe)
            if step in self.keep or step == total:
                try:
                    self.frames[step] = P.latent_preview_image(
                        latents, width, height, proj=P._preview_proj_for(pipe))
                except Exception as e:
                    self.frames[step] = f"error: {e}"
        P._store_preview = rec
        P.preview_begin()
        return self

    def __exit__(self, *a):
        self.P._store_preview = self.orig
        self.P.preview_end()


def _mae(preview, final):
    import numpy as np
    from PIL import Image
    w, h = preview.size
    a = np.asarray(preview.convert("RGB"), dtype=np.float32)
    b = np.asarray(final.convert("RGB").resize((w, h), Image.BOX), dtype=np.float32)
    return round(float(np.abs(a - b).mean()), 1)


def _pixel_diff(a, b):
    import numpy as np
    x = np.asarray(a.convert("RGB"), dtype=np.int16)
    y = np.asarray(b.convert("RGB"), dtype=np.int16)
    if x.shape != y.shape:
        return None, None
    d = np.abs(x - y)
    return round(float(d.mean()), 3), int(d.max())


# --- Renders -----------------------------------------------------------------------------
def render(P, tag, width, height, steps, seed, prompt=PROMPT, negative=NEGATIVE,
           out_dir=None):
    import torch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    split = {}
    vae = getattr(getattr(P._BASE_PIPE, "_pipe", P._BASE_PIPE), "vae", None)
    real_decode = getattr(vae, "decode", None)
    if real_decode is not None:
        def decode(*a, **k):
            torch.cuda.synchronize()
            split["denoise_peak_reserved_gb"] = _gb(torch.cuda.max_memory_reserved())
            split["before_decode_alloc_gb"] = _gb(torch.cuda.memory_allocated())
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            out = real_decode(*a, **k)
            torch.cuda.synchronize()
            split["decode_peak_reserved_gb"] = _gb(torch.cuda.max_memory_reserved())
            split["decode_peak_alloc_gb"] = _gb(torch.cuda.max_memory_allocated())
            split["decode_seconds"] = round(time.time() - t, 1)
            return out
        vae.decode = decode
    t0 = time.time()
    try:
        img = P.generate(prompt, width, height, steps, seed, negative_prompt=negative)
    finally:
        if real_decode is not None:
            del vae.decode
    torch.cuda.synchronize()
    dt = time.time() - t0
    path = os.path.join(out_dir or OUT, f"{tag}.png")
    img.save(path)
    used, total = _smi_used_gb()
    r = {"size": f"{img.width}x{img.height}", "steps": steps, "seconds": round(dt, 1),
         "s_per_step": round(dt / steps, 3),
         "peak_alloc_gb": _gb(torch.cuda.max_memory_allocated()),
         "peak_reserved_gb": max(_gb(torch.cuda.max_memory_reserved()),
                                 split.get("denoise_peak_reserved_gb", 0)),
         "driver_used_after_gb": used, "driver_total_gb": total,
         "offload": P._effective_offload(), "precision": P.PRECISION,
         "image": os.path.relpath(path, ROOT), **split}
    _say(f"{tag}: {r['size']} in {r['seconds']} s ({r['s_per_step']} s/step), "
         f"peak {r['peak_reserved_gb']} GB reserved")
    if split:
        _say(f"   denoise peak {split.get('denoise_peak_reserved_gb')} GB, VAE decode peak "
             f"{split.get('decode_peak_reserved_gb')} GB in {split.get('decode_seconds')} s")
    return img, r


def stop_test(P, steps, seed, stop_after_s):
    """Stops a 1024 render from another thread, like the UI's Stop button."""
    box = {}

    def run():
        try:
            box["img"] = P.generate(PROMPT, 1024, 1024, steps, seed)
        except Exception as e:
            box["err"] = f"{type(e).__name__}: {e}"
    th = threading.Thread(target=run, daemon=True)
    P.preview_begin()
    t0 = time.time()
    th.start()
    time.sleep(stop_after_s)
    t_stop = time.time()
    P.request_stop()
    th.join(timeout=300)
    t_end = time.time()
    P._STOP = False
    snap = P.preview_snapshot()
    P.preview_end()
    r = {"stop_after_s": stop_after_s, "returned_after_stop_s": round(t_end - t_stop, 1),
         "total_s": round(t_end - t0, 1), "thread_finished": not th.is_alive(),
         "image_returned": "img" in box, "error": box.get("err"),
         "last_preview_step": snap.get("step")}
    if "img" in box:
        p = os.path.join(OUT, "stop_partial.png")
        box["img"].save(p)
        r["image"] = os.path.relpath(p, ROOT)
    ok = (r["image_returned"] and r["returned_after_stop_s"] < 15
          and (r["last_preview_step"] or 0) < steps)
    r["verdict"] = "OK" if ok else "CHECK"
    _say(f"stop: returned {r['returned_after_stop_s']} s after the request, image "
         f"{'yes' if r['image_returned'] else 'NO'} -> {r['verdict']}")
    return r


def load(P, label):
    import torch
    P.free_vram()
    torch.cuda.synchronize()
    base_used, _ = _smi_used_gb()
    t0 = time.time()
    P.get_pipe("txt2img")
    torch.cuda.synchronize()
    dt = time.time() - t0
    used, total = _smi_used_gb()
    r = {"seconds": round(dt, 1), "offload": P._effective_offload(),
         "offload_asked": P.OFFLOAD_MODE, "precision": P.PRECISION,
         "alloc_gb": _gb(torch.cuda.memory_allocated()),
         "reserved_gb": _gb(torch.cuda.memory_reserved()),
         "driver_used_before_gb": base_used, "driver_used_gb": used,
         "driver_total_gb": total,
         "note": "the first run includes the download of ~27 GB"}
    _say(f"{label}: loaded in {r['seconds']} s, offload {r['offload']}, encoder "
         f"{r['precision']}, at rest {r['alloc_gb']} GB allocated / driver {used} GB of {total}")
    return r


def _encode_time(P, prompt):
    """Seconds to encode one prompt with the text encoder (the encoder-only offload's
    round trip included under 'model')."""
    import torch
    pipe = P.get_pipe("txt2img")
    torch.cuda.synchronize()
    t0 = time.time()
    pipe.encode_prompt(prompt)
    torch.cuda.synchronize()
    return round(time.time() - t0, 2)


# --- 6. The text A/B ----------------------------------------------------------------------
def ab_test(P, steps, seed0, n_seeds, ideogram_bench):
    os.makedirs(AB_OUT, exist_ok=True)
    rows = []
    for d in designs():
        with open(os.path.join(AB_OUT, f"{d['name']}_layout.json"), "w", encoding="utf-8") as f:
            json.dump(d["layout"], f, ensure_ascii=False, indent=1)
        for form in ("plain", "layout"):
            prompt = d["plain"] if form == "plain" else json.dumps(d["layout"], ensure_ascii=False)
            for i in range(n_seeds):
                seed = seed0 + i
                tag = f"{d['name']}_{form}_s{seed}"
                try:
                    _, r = render(P, tag, d["width"], d["height"], steps, seed, prompt=prompt,
                                  out_dir=AB_OUT)
                    rows.append({"design": d["name"], "form": form, "seed": seed,
                                 "texts": d["texts"], "seconds": r["seconds"],
                                 "s_per_step": r["s_per_step"], "peak_reserved_gb": r["peak_reserved_gb"],
                                 "image": r["image"]})
                except Exception as e:
                    rows.append({"design": d["name"], "form": form, "seed": seed,
                                 "error": f"{type(e).__name__}: {e}"})
                    _say(f"{tag} FAILED: {type(e).__name__}: {e}")
                RESULTS["steps"]["6_ab"] = {"rows": rows, "ideogram_bench": ideogram_bench}
                _save()
    _write_ab(rows, ideogram_bench)
    _say(f"A/B: {len(rows)} renders in {AB_OUT}, table in {os.path.join(OUT, 'ab.md')}")


_IDEOGRAM_REFS = {"poster_bakery": "hand.png", "logo_coffee": "magic_1.png",
                  "poster_swiss": "magic_0.png"}


def _write_ab(rows, ideogram_bench):
    L = ["# Lot 1 A/B: MOSS-Image against the crispz-ideogram lot-2 designs", "",
         "Same ideas, two prompt forms (plain text, layout JSON), the seeds listed. The "
         "columns `exact text`, `ghost text` and `banding` are yours to fill in by eye "
         "(yes / no / partly); the time is measured. Ideogram's renders are the crispz-"
         "ideogram bench (lot 2) when found.", ""]
    for d in designs():
        L += [f"## {d['name']} ({d['width']}x{d['height']}): texts {', '.join(repr(t) for t in d['texts'])}", ""]
        ref = None
        if ideogram_bench:
            cand = os.path.join(ideogram_bench, _IDEOGRAM_REFS.get(d["name"], ""))
            if os.path.isfile(cand):
                ref = cand
        if ref:
            L.append(f"Ideogram 4 reference: `{ref}`")
            L.append("")
        L += ["| form | seed | s | s/step | image | exact text | ghost text | banding |",
              "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if r["design"] != d["name"]:
                continue
            if "error" in r:
                L.append(f"| {r['form']} | {r['seed']} | ERROR {r['error']} | | | | | |")
            else:
                L.append(f"| {r['form']} | {r['seed']} | {r['seconds']} | {r['s_per_step']} | "
                         f"`{r['image']}` | | | |")
        L.append("")
    ok = [r for r in rows if "error" not in r]
    if ok:
        plain = [r["seconds"] for r in ok if r["form"] == "plain"]
        lay = [r["seconds"] for r in ok if r["form"] == "layout"]
        if plain and lay:
            L.append(f"Mean time: plain {sum(plain) / len(plain):.1f} s, layout "
                     f"{sum(lay) / len(lay):.1f} s (a layout prompt is longer to encode).")
    L += ["", "Verdict (Mike): at least as good as Ideogram 4 on the text, or not: ________", ""]
    with open(os.path.join(OUT, "ab.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main():
    steps = _arg("--steps", 50, int)
    seed = _arg("--seed", 12345, int)
    n_ab = _arg("--ab-seeds", 4, int)
    ideogram_bench = _arg("--ideogram-bench") or os.path.join(os.path.dirname(ROOT),
                                                              "crispz-ideogram", "bench", "lot2")
    if not os.path.isdir(ideogram_bench):
        ideogram_bench = None
    os.makedirs(OUT, exist_ok=True)
    P = prechecks()
    import torch

    off = _arg("--offload")
    if off:
        P.set_offload_mode(off)
    P.LIVE_PREVIEW_ENABLED = True
    ab_only = "--ab-only" in sys.argv

    # 1. Loading
    try:
        RESULTS["steps"]["1_load"] = load(P, "load")
        _save()
    except Exception as e:
        RESULTS["steps"]["1_load"] = {"error": f"{type(e).__name__}: {e}",
                                      "trace": traceback.format_exc()}
        _save()
        _say(f"LOADING FAILED: {type(e).__name__}: {e}")
        sys.exit(1)

    # 2. Renders: 1024 cold and warm (preview frames on the warm one), 2 Mpx, 2048x1152
    keep = {1, steps // 4, steps // 2, (3 * steps) // 4, steps}
    plan = [] if ab_only else [("gen1024_cold", 1024, 1024, False), ("gen1024_warm", 1024, 1024, True)]
    if not ab_only and "--skip-2mpx" not in sys.argv:
        plan += [("gen1456", 1456, 1456, False), ("gen2048x1152", 2048, 1152, False)]
    for tag, w, h, rec in plan:
        try:
            if rec:
                with FrameRecorder(P, keep) as fr:
                    img, r = render(P, tag, w, h, steps, seed)
                pv = {}
                for st, frame in sorted(fr.frames.items()):
                    if not hasattr(frame, "save"):
                        pv[str(st)] = {"error": str(frame)}
                        continue
                    frame.save(os.path.join(OUT, f"preview_step{st:02d}.png"))
                    pv[str(st)] = {"mae": _mae(frame, img)}
                last = pv.get(str(steps), {})
                pv["verdict"] = ("OK" if last.get("mae") is not None and last["mae"] < 40
                                 else "CHECK")
                RESULTS["steps"]["3_preview"] = pv
                _say(f"preview, last step: error {last.get('mae')} (FLUX.2 matrix on MOSS "
                     f"latents, 0-255) -> {pv['verdict']}")
            else:
                img, r = render(P, tag, w, h, steps, seed)
            RESULTS["steps"][f"2_{tag}"] = r
        except Exception as e:
            RESULTS["steps"][f"2_{tag}"] = {"error": f"{type(e).__name__}: {e}",
                                            "trace": traceback.format_exc()}
            _say(f"{tag} FAILED: {type(e).__name__}: {e}")
        _save()
    try:
        RESULTS["steps"]["2_encode_prompt_s"] = _encode_time(P, PROMPT)
        _say(f"text encoder: {RESULTS['steps']['2_encode_prompt_s']} s for the poster prompt")
    except Exception as e:
        RESULTS["steps"]["2_encode_prompt_s"] = f"error: {e}"
    _save()

    # 4. Stop, about a third of the way into a warm 1024 render
    warm = RESULTS["steps"].get("2_gen1024_warm", {}).get("seconds") or 30
    if not ab_only:
        try:
            RESULTS["steps"]["4_stop"] = stop_test(P, steps, seed + 1, max(2.0, warm / 3))
        except Exception as e:
            RESULTS["steps"]["4_stop"] = {"error": f"{type(e).__name__}: {e}"}
        _save()

    # 5. The text encoder in nf4: a reload, the warm 1024 again, pixel difference
    if "--nf4" in sys.argv and not ab_only:
        try:
            from PIL import Image
            before = P.PRECISION
            P.set_precision("nf4")
            r = {"load": load(P, "load nf4")}
            r["encode_prompt_s"] = _encode_time(P, PROMPT)
            img, rr = render(P, "gen1024_nf4_encoder", 1024, 1024, steps, seed)
            r["render"] = rr
            ref = os.path.join(OUT, "gen1024_warm.png")
            if os.path.isfile(ref):
                r["vs_bf16_mean_abs_diff"], r["vs_bf16_max_abs_diff"] = _pixel_diff(
                    Image.open(ref), img)
                _say(f"nf4 encoder vs bf16: mean pixel diff {r['vs_bf16_mean_abs_diff']}, "
                     f"max {r['vs_bf16_max_abs_diff']}")
            RESULTS["steps"]["5_nf4_encoder"] = r
            P.set_precision(before)
            RESULTS["steps"]["5_nf4_encoder"]["reload_bf16"] = load(P, "load bf16 again")
        except Exception as e:
            RESULTS["steps"]["5_nf4_encoder"] = {"error": f"{type(e).__name__}: {e}",
                                                 "trace": traceback.format_exc()}
            _say(f"nf4 encoder FAILED: {type(e).__name__}: {e}")
        _save()

    # 6. The text A/B
    if "--skip-ab" not in sys.argv:
        try:
            ab_test(P, steps, seed, n_ab, ideogram_bench)
        except Exception as e:
            RESULTS["steps"]["6_ab"] = {"error": f"{type(e).__name__}: {e}",
                                        "trace": traceback.format_exc()}
            _say(f"A/B FAILED: {type(e).__name__}: {e}")
    RESULTS["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _save()
    _say(f"done: {os.path.relpath(os.path.join(OUT, 'report.md'), ROOT)}")


def _write_report():
    s = RESULTS["steps"]
    L = ["# Lot 1 on the GPU: MOSS-Image (bf16)", "",
         f"Started {RESULTS['started']}" + (f", finished {RESULTS['finished']}"
                                            if "finished" in RESULTS else ", NOT finished"),
         ""]
    pre = s.get("0_prechecks", {})
    if pre:
        L += [f"- GPU: {pre.get('gpu')}, torch {pre.get('torch')} (CUDA {pre.get('cuda')})",
              f"- diffusers {pre.get('diffusers')}, transformers {pre.get('transformers')}, "
              f"bitsandbytes {pre.get('bitsandbytes')}",
              f"- repo {pre.get('repo')}, encoder precision {pre.get('precision')}, access "
              f"{pre.get('repo_access', 'local folder')}",
              f"- kernels: " + ", ".join(f"{k[7:]} {v}" for k, v in pre.items()
                                         if k.startswith("kernel_")), ""]
    ld = s.get("1_load", {})
    if ld:
        L += ["## Loading", "", "| | value |", "|---|---|"]
        for k in ("seconds", "offload_asked", "offload", "precision", "alloc_gb", "reserved_gb",
                  "driver_used_before_gb", "driver_used_gb", "error"):
            if k in ld:
                L.append(f"| {k} | {ld[k]} |")
        L.append("")
    rows = [(k, v) for k, v in s.items() if k.startswith("2_gen")]
    if rows:
        L += ["## Renders", "", "| render | size | s | s/step | peak alloc GB | "
              "peak reserved GB | driver after GB | offload |", "|---|---|---|---|---|---|---|---|"]
        for k, v in rows:
            if "error" in v:
                L.append(f"| {k} | ERROR: {v['error']} | | | | | | |")
            else:
                L.append(f"| {k} | {v['size']} | {v['seconds']} | {v['s_per_step']} | "
                         f"{v['peak_alloc_gb']} | {v['peak_reserved_gb']} | "
                         f"{v['driver_used_after_gb']} | {v['offload']} |")
        for k, v in rows:
            if "decode_peak_reserved_gb" in v:
                L.append(f"- {k}: denoise peak {v.get('denoise_peak_reserved_gb')} GB, VAE "
                         f"decode peak {v['decode_peak_reserved_gb']} GB (alloc "
                         f"{v.get('decode_peak_alloc_gb')}) in {v.get('decode_seconds')} s")
        if "2_encode_prompt_s" in s:
            L.append(f"- text encoder: {s['2_encode_prompt_s']} s for the poster prompt "
                     f"(offload round trip included under 'model')")
        if pre.get("sdpa_kernels"):
            L.append(f"- SDPA kernels at head_dim 128: {pre['sdpa_kernels']}")
        L.append("")
    pv = s.get("3_preview")
    if pv:
        L += ["## Live preview (mean error against the final image, 0-255)", "",
              "| step | error |", "|---|---|"]
        for k, v in pv.items():
            if k != "verdict" and isinstance(v, dict) and "mae" in v:
                L.append(f"| {k} | {v['mae']} |")
        L += ["", f"Verdict: {pv.get('verdict')}", ""]
    st = s.get("4_stop")
    if st:
        L += ["## Stop", "", "```", json.dumps(st, indent=1), "```", ""]
    nf = s.get("5_nf4_encoder")
    if nf:
        L += ["## Text encoder in nf4", "", "```", json.dumps(nf, indent=1), "```", ""]
    ab = s.get("6_ab")
    if ab:
        n = len(ab.get("rows", []))
        L += ["## Text A/B", "", f"{n} renders, see `bench/lot1/ab.md`"
              + (f" (error: {ab['error']})" if "error" in ab else ""), ""]
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
