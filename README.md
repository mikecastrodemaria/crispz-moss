# crispz-moss

The **MOSS-Image** studio of the crispz family: text in images and design (posters,
logos, packaging, slides, covers, dense typography), Fooocus-style, **100% local**.
MOSS-Image (OpenMOSS, Fudan, October 2026) is a 5B open-weights model under **Apache
2.0**: commercial use allowed. Measured against Ideogram 4 on the same posters (24
renders), it rendered every requested text exactly, faster, and with no licence strings
attached. The other crispz studios stay better at photos and editing.

> **Work in progress (0.1.0-dev, 2026-11-04).** Lots 1 to 3 of 6 are in: the engine,
> the single-file loaders, the layout editor, the magic prompt, Describe, and the
> emulated img2img / inpaint / outpaint. LoRAs (lot 4), the local tools (lot 5) and the
> release (lot 6) follow. The specification and the work log are in `CLAUDE.md`.

## Licence: read this first

- **The model.** MOSS-Image's weights and pipeline code are **Apache 2.0** (OpenMOSS).
  Its text encoder is Qwen3.5-9B (Apache 2.0) and its VAE is FLUX.2 klein's (Apache 2.0,
  to confirm on the black-forest-labs repo at release). Commercial use is allowed:
  client deliverables made locally are fine. Community LoRAs each carry their own
  licence, read before any client use.
- **This app's code** is under CC BY-NC 4.0 (see [LICENSE.txt](LICENSE.txt)).
- **Nothing leaves the machine.** There is no cloud function at all. Ollama, when used,
  is a local server (its URL is logged at every call).

## What it does

| | |
|---|---|
| **Generate** | MOSS-Image text to image, 50 steps, CFG 3.0 (official settings), a real negative prompt, 256 to 2048 a side, validated from 1 to 2 Mpx |
| **Layout JSON** | MOSS-Image reads structured layouts: an editor builds them (summary, meta, palette, every text with its exact content and its box), boxes drawn with the mouse on a canvas at the chosen ratio, raw JSON mode. A layout is the guarantee "nothing but these words" |
| **Design presets** | poster (A series), logo, business card, book cover, slide 16:9, packaging, YouTube thumbnail, story 9:16, banner 6:1, each at its native size |
| **Magic prompt** | a plain idea into a layout, through Ollama (a local LLM, seconds) or locally with the model's own Qwen3.5-9B encoder and its LM head (about a minute) |
| **Describe** | an image into a layout: through an Ollama vision model, or locally with the Qwen3.5-9B encoder (its vision tower reads the texts verbatim and places the boxes) |
| **img2img, inpaint, outpaint, reframe** | **emulated**: MOSS-Image has no native pipeline for them. SDEdit on the tail of its own schedule, latent blending under the mask |
| **Upscale** | ESRGAN + diffusion refine, tiled |
| **Single files** | ComfyUI / CivitAI DiTs (bf16, fp8, int8 ConvRot) and Qwen3.5 encoders (bf16, int8 ConvRot) load; the 4-bit builds are refused with the reason |
| **Character catalog** | the MOSS-Image Character Catalog (34,249 samples: characters by copyright, styles, each with its prompt, tags and facets) as Asset Browser tabs, downloaded once on a click, the sample renders optionally kept locally (thumbnails 270 MB, full renders 1.8 GB); a card sends its prompt or tags to the prompt box. Uncurated, third-party IP: not for client work |
| **Send to prompt** | from the Asset Browser, an output's prompt, a LoRA's trigger words or one of its CivitAI example prompts goes straight into the app's prompt box |
| **Vectorize (SVG)** | locally with vtracer: made for logos, icons and flat colours |
| **Remove BG** | locally with rembg |
| **Edit with references** | the **Moss-Image-Edit LoRA** (community, lot 4b): the image to edit plus up to 3 references enter the DiT as a prefix, the instruction in plain words; fetched once (about 300 MB), its author states no licence for the weights. Native multi-reference editing: handed to **crispz-flux3** on the same machine (see below) |
| **LoRAs** | MOSS-Image LoRAs in every published layout (kohya, peft, moss-edit-lora, ComfyUI diffs such as DeBand), applied hot by the app's own loader; what cannot apply is named, nothing is skipped in silence |
| **The rest of the family** | styles, wildcards, X/Y/Z grids, job queue, Asset Browser, CivitAI search, face swap, detailer, CLI and the family protocol v1 (`czp`) |

## Requirements

- An **NVIDIA GPU** with CUDA. The DiT is 10 GB in bf16, the text encoder 17 GB in bf16
  (or about 6 GB in nf4 through bitsandbytes), the VAE 0.25 GB.
- **32 GB of VRAM** measured below with the `auto` offload: the DiT and the VAE stay on
  the card, the encoder comes in for each new prompt. With less VRAM, the nf4 encoder
  (`moss_precision`) keeps everything resident at 21 GB flat; `sequential` goes lower.
- About 48 GB of system RAM is comfortable (the bf16 encoder waits there between
  prompts); the nf4 encoder removes that need.
- Python 3.11 or later (3.13 advised), Windows or Linux.

## Install

Windows: `install.bat`, then `run.bat`. Linux: `install.sh`, then `run.sh`.

Do **not** run `pip install -r requirements.txt` in the installed `.venv`: gradio 5.50
declares `pillow<12` and pip would downgrade Pillow 13 (the install scripts handle it).
To add one dependency, install it alone at its locked version.

The first render downloads about 27 GB (`OpenMOSS-Team/MOSS-Image-SFT`, public, no
token), 25 to 30 s to load from the cache afterwards. The first local magic prompt adds
a 2 GB LM head; the first Remove BG the 176 MB u2net model.

**An old `config.txt`** (written by another crispz) keeps its values: check
`default_model_profile` (50 steps, guidance 3.0) and the performance preset.

## Measured on an RTX 5090

Windows, torch 2.11 cu128, bf16, offload `auto` (encoder only), 50 steps, CFG 3.0.

| | |
|---|---|
| Load from the cache | 25 to 30 s |
| VRAM at rest | 9.7 GB allocated (DiT + VAE) |
| 1024x1024 | 43 s (0.85 s/step), 15 GB peak once the prompt is cached; 27 GB while a new prompt was encoded before `moss_encode_swap` (the DiT now steps aside: about 19 GB expected, to measure), 21 GB flat with the nf4 encoder |
| 1456x1456 (2 Mpx) | 102 s (2.0 s/step), 15.6 GB peak |
| 2048x1152 | 116 s (2.3 s/step), 15.7 GB peak |
| Stop | the partial image comes back 0.8 s after the request |
| Magic prompt | Ollama 12 to 20 s; local (Qwen3.5-9B + LM head) 77 s |
| Describe (local) | 76 s, the logo's text read back, boxes on the symbol and the text |
| img2img 1152x1632, denoise 0.3 / 0.6 | 71 s (first render, cold) / 52 s |
| Inpaint, outpaint, reframe (work area capped at ~1 Mpx) | 43 s, 44 s, 40 s |
| Draft, the poster at 12 / 20 / 25 / 50 steps | 25 / 34 / 43 / 85 s, texts exact at every count |

A/B against Ideogram 4 (crispz-ideogram's lot 2 posters, 4 seeds each, plain text and
layout): 24 renders out of 24 with every requested text exact, small fixed texts
included, no banding; the bakery poster in 25 s against 56 s.

## Getting good text

- **Use layouts.** Plain text works, but in plain text the model sometimes adds words it
  infers from the brief (a misspelt "COFFEE SHOP" on a coffee logo). With a layout,
  where every text is listed with its box, it never did. Start from a design preset or
  the magic prompt, then fix the texts in the editor.
- The model keeps the text before the box: a box drawn too small is overflowed, the
  words stay whole.
- Outpaint and reframe from a plain-text prompt fill the new area with invented text: a
  layout placed on the original image (lot 3b) is the fix.
- Inpaint: describe the whole picture, not only the patch.

## Editing with references

Two ways, both local.

**Edit (references) tab (lot 4b)**: MOSS-Image's base weights never saw a reference,
but the community **Moss-Image-Edit LoRA** (chinoll/Moss-Image-Edit, repacked by
Comfy-Org, rank 64) teaches it the "reference prefix" contract: the image to edit (Ref 1)
and up to 3 more references are VAE-encoded and put in front of the sequence at t = 0,
seeing only each other; the target and the instruction see everything; the result takes
Ref 1's size. The app rewrote that contract on its vendored DiT (nothing of the GPLv3
node code is reused) and caches the prefix's keys and values once per edit, so the
references cost little per step. The LoRA is fetched once (about 300 MB) on the first
Edit and applied for edits only. **Its author states no licence for the weights** (the
training code is GPLv3): check before a client deliverable. References enter at their
own size (long side capped at 2048, `moss_edit_ref_max_edge`): the LoRA was trained on
pairs of the same size, and on the 5090 a poster read at 1024 came back with garbled
texts. The LoRA was trained with ONE reference (the image to edit): extra references are
experimental, their content leaks into the result. Texts are redrawn, not copied: check
them. The author's inference script uses 50 steps and CFG 4 (measured better than CFG 3
on the texts). First measurements: `tools/gpu_lot4b_check.py`, see CLAUDE.md.

**Edit in crispz-flux3 tab**: native multi-reference editing. It hands the image (and
up to 3 references) to crispz-flux3 on the same machine, through its own protocol v1
command line:
- set the crispz-flux3 install folder in the tab (`crispz_flux3_dir`);
- if crispz-flux3's app is running, the edit goes through it (model warm); if not, it
  runs in crispz-flux3's own process after crispz-moss frees its VRAM;
- mind the ports: both apps start on 7860, and the second one started takes 7861.
  crispz-flux3 looks for its own app at its `cli_protocol.instance_url` (7860 by
  default), so **start crispz-flux3 first**, or point that setting at the port its app
  really uses. If it finds crispz-moss there instead, the hand-over says so and runs
  crispz-flux3 cold.

Nothing leaves the machine: the files go under the output folder (`handoff/`).

## Command line

`python app.py --help`, the cheat sheet in [README_CLI.md](README_CLI.md), and the
family protocol: `czp caps` / `czp gen --spec spec.json` (routes to the running app,
else runs locally). `caps` says what is emulated (`supports.emulated`).

## Tests

`.venv\Scripts\python tools\run_tests.py` (CPU, a tiny random MOSS-Image, one process
per test file: what the CI runs). The lot scripts in `tools/` (`gpu_lot1_check.py`,
`gpu_lot2_check.py`, `gpu_lot3_check.py`) measure each part on the card, GPU free.

## Credits

OpenMOSS for MOSS-Image and its pipeline (Apache 2.0, vendored in
`third_party/moss_image/`); Qwen (Qwen3.5-9B, Apache 2.0); Black Forest Labs (the FLUX.2
VAE); diffusers and transformers; vtracer (MIT); rembg (MIT); the crispz family
(crispz-klein, crispz-flux3, crispz-ideogram) this studio is forked from. How and why:
[FORK.md](FORK.md); the full work log: [CLAUDE.md](CLAUDE.md).
