# crispz-moss

The **MOSS-Image** studio of the crispz family: text in images and design (posters,
logos, packaging, slides, covers, dense typography), Fooocus-style, **100% local**.
MOSS-Image (OpenMOSS, Fudan, October 2026) is a 5B open-weights model under **Apache
2.0**: commercial use allowed. Measured against Ideogram 4 on the same posters (24
renders), it rendered every requested text exactly, faster, and with no licence strings
attached. The other crispz studios stay better at photos and editing.

> **1.0.0, 2026-11-05.** The six lots of the specification are in and measured on an
> RTX 5090: the engine and the single-file loaders, the layout editor, the magic prompt
> and Describe, the emulated img2img / inpaint / outpaint, LoRAs and editing with
> references, the local tools and the hand-overs, the GPU end-to-end test and this
> documentation. The specification and the work log (every measurement, every decision)
> are in `CLAUDE.md`.

## Licence: read this first

The chain, as stated at the source (Hugging Face model cards and LICENSE files, GitHub,
read on 2026-11-05):

| Component | Licence | What it means |
|---|---|---|
| MOSS-Image weights (`OpenMOSS-Team/MOSS-Image-SFT`, `-Base`) and bundled pipeline code | **Apache 2.0** (card and LICENSE file) | commercial use allowed; keep the attribution and say what you changed |
| Pipeline code vendored in `third_party/moss_image/` | Apache 2.0 (OpenMOSS GitHub) | unchanged copy, with its LICENSE and origin commit |
| Text encoder Qwen3.5-9B (`Qwen/Qwen3.5-9B`, Alibaba Cloud) | Apache 2.0 (card and LICENSE) | same; the SFT card names no encoder, the identification comes from its config (`Qwen3_5Model`) and the Comfy-Org file names |
| VAE (the FLUX.2 "small decoder", `black-forest-labs/FLUX.2-klein-4B`) | Apache 2.0 at the repo level (card: "fully open under Apache 2.0"); no separate licence file for the `vae/` folder | the 9B klein is non-commercial, the 4B and its VAE are not |
| Comfy-Org repack and its LoRAs, DeBand, the DMD LoRA | Apache 2.0 badges | read each card before client use |
| Moss-Image-Edit LoRA (chinoll; the Comfy-Org `moss_image_edit_lora_rank_64` file) | **no licence stated for the weights** (the training code is GPLv3) | the app says so wherever it fetches it; not for a client deliverable until the author says |
| Character Catalog (hide1) | MIT for the author's data; the characters and renders are third-party IP | reference only, not for client work |
| This app's code | CC BY-NC 4.0 ([LICENSE.txt](LICENSE.txt)) | |

**Nothing leaves the machine.** There is no cloud function at all. Ollama, when used, is
a local server (its URL is logged at every call); the hand-overs call crispz-flux3 and
crispz-ideogram on this machine.

## What it does

| | |
|---|---|
| **Generate** | MOSS-Image text to image, 50 steps, CFG 3.0 (official settings), a real negative prompt, 256 to 2048 a side, validated from 1 to 2 Mpx. Performance presets: MOSS (50 steps), Draft (12 steps, the base model) and **Draft (8 steps, DMD LoRA)**: the lodestones 8-step DMD adapter (Apache 2.0, 1.2 GB, fetched once) in slot 1 at CFG 1, one DiT pass per step, texts exact on the 5090; it leaves slot 1 again with another preset |
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
| 1024x1024 | 43 s (0.85 s/step), 15 GB peak once the prompt is cached; 23.5 GB while a new prompt is encoded (the encoder and its temporaries, the DiT stepping aside: `moss_encode_swap`), 21 GB flat with the nf4 encoder |
| 1456x1456 (2 Mpx) | 102 s (2.0 s/step), 15.6 GB peak |
| 2048x1152 | 116 s (2.3 s/step), 15.7 GB peak |
| Stop | the partial image comes back 0.8 s after the request |
| Magic prompt | Ollama 12 to 20 s; local (Qwen3.5-9B + LM head) 77 s |
| Describe (local) | 76 s, the logo's text read back, boxes on the symbol and the text |
| img2img 1152x1632, denoise 0.3 / 0.6 | 75 s (first render, new prompt) / 52 s, texts exact at both |
| Inpaint, outpaint, reframe (work area capped at ~1 Mpx) | 45 s, 47 s, 46 s |
| Draft, the poster at 12 / 20 / 25 / 50 steps | 25 / 34 / 43 / 85 s, texts exact at every count |
| LoRAs (DeBand, the DMD LoRA at 8 steps CFG 1: the Draft (8 steps, DMD LoRA) preset) | 100 s, 18 s (about 9 s of render), texts exact |
| Edit with references, 1152x1632 poster, CFG 4 / 3 | 170 s / 125 s, 24 GB peak, texts exact; a 1024 logo on a new background 62 s |

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
- Outpaint and reframe: from a plain-text prompt the new area fills with invented text;
  with a layout placed on the original image (lot 3b) the gibberish goes, but the model
  still continued the headline across the new bands at canvas scale. Each new band is
  now declared as an empty-background item of the layout (and named in a plain prompt):
  to be measured.
- Inpaint: describe the whole picture, not only the patch.

## Performance presets: MOSS, Draft, Draft with the DMD LoRA

The **Performance** radio of Settings sets the steps and the CFG in one click (and, for
the third one, a LoRA). Measured on the RTX 5090 on the poster preset (1152x1632):

| preset | steps / CFG | what it is for | time (prompt already encoded) |
|---|---|---|---|
| **MOSS (50 steps)** | 50 / 3.0 | the official settings, the final render | about 85 s (110 s with the first encoding) |
| **Draft (12 steps)** | 12 / 3.0 | a quick look on the base model: the texts held at 12 steps on every poster measured, the composition and the details may move against the 50-step render of the same seed | about 18 to 22 s |
| **Draft (8 steps, DMD LoRA)** | 8 / 1.0 + LoRA | the lodestones 8-step DMD adapter in LoRA slot 1 at 1.0, CFG 1 (one DiT pass per step, no negative branch): a clean, complete image with exact texts on the measured seed | about 9 s of render |
| **Draft (4 steps, Turbo LoRA)** | 4 / 1.0 + LoRA | akhaliq's Euler-native Turbo adapter (warm-started from the DMD, trained on the release scheduler: its anchors are exactly our 4-step timesteps), the author's "final" tier, the best text of the few-step adapters by his own strips | not measured here yet; the author: 5.6 s per 1024 on an A100 |
| **Draft (3 steps, Turbo LoRA)** | 3 / 1.0 + LoRA | the same family's draft tier: 25 % faster, single sharp subjects, primary text correct (the 2-step is a documented failure) | not measured here yet |

The Turbo adapters were trained at 1024x1024 only. Above that the app raises the step
count at render time, following the author's table (6 steps around 1536 on the long
side, 8 at 2048; the 3-step gets the same rule, which he did not measure for it), and
says so in the log; a step count you set higher is never lowered. Above 1024 a long,
descriptive prompt holds much better than a short one (a layout is already that). They
are kept unmerged in fp32 as the author asks: the distillation delta is smaller than
bf16's rounding, merged in bf16 the fine text goes first. On the Hub both files are
called `lora.safetensors` (the 4-step) and `lora_3step.safetensors`: the 4-step is only
recognised under its stable names (`moss-turbo-4-steps.safetensors`,
`moss_image_sft_4step_v2.0.safetensors`), since a `lora.safetensors` with that header
is also the DMD file.

How the DMD preset gets its LoRA: it looks in the LoRA folders first (under its stable
name `moss-dmd-8-steps.safetensors`, or under the Hub's generic `lora.safetensors` when
the file's header is the DMD one), otherwise it fetches it once from
`lodestones/MOSS-Image-SFT-8-steps-dmd` (1.2 GB, Apache 2.0) into `loras/_hf-edit/`,
the licence written in the log. Pick another preset and the LoRA leaves slot 1 again;
your own LoRAs in the other slots are untouched (the Turbo presets fetch theirs the
same way, from `akhaliq/MOSS-Image-SFT-4-steps`). Should the download fail, the preset
falls back to Draft (12 steps) and says so. The XYZ "Performance" axis and the CLI
apply the same rule. Note: the DMD author's full recipe also replaces the time
embedding and samples with its own loop; only the LoRA is used here, and it held on
images. A `config.txt` that copied `performance_presets` keeps its own list: add the
line `"Draft (8 steps, DMD LoRA)": [8, 1.0, "moss-dmd-8-steps"]` or drop the key.

The steps and the CFG stay editable after a preset (the sliders below the radio); a
preset is just a starting point. Guidance 3 is the official value, 3 to 6 advised, raise
it for a short plain-text prompt.

## img2img, inpaint, outpaint, reframe and upscale: emulated

MOSS-Image ships one pipeline, text to image. Everything below is **emulated** by the
app on top of it (SDEdit on the tail of the model's own schedule, the image encoded by
the same VAE; latent blending under a mask), and the family protocol announces it as
such (`supports.emulated`). Measured on the 5090 on the lot 2 poster (1152x1632), offload
"model": img2img 0.3 in 55 s and 0.6 in 52 s (texts exact, poster intact), inpaint 43 to
47 s, outpaint one side 47 s, reframe 16:9 46 s, refine at 12 steps 8 s.

**Where**: the **Input Image** panel of the Generate tab (tick "Input Image"), one tab
per tool. The prompt box is the prompt of the edit: describe the whole picture, not only
the patch; with a layout, keep it (it follows the image, see below).

- **Upscale or img2img**: load an image. **ESRGAN upscale** (factor, model) then
  **Refine (img2img)** at a denoise: the denoise is the fraction of the schedule that
  runs (0.3 = the last 15 of 50 steps), 0 skips the refine (ESRGAN only). The **Use
  case presets** (Photo, Subtle, Detailed, Portrait, 4K tiled, Low VRAM) set factor,
  denoise, steps and tiles; the 12 to 16 step counts were measured against 36 (same
  images to the eye, 53 % slower). **Vary (subtle)** and **Vary (strong)** are a plain
  img2img at 0.25 and 0.6 on the loaded image. Large outputs run in **tiles** (ESRGAN
  tiles and diffusion tiles with an overlap): the 4K preset renders 2880x4096 under
  18 GB with no visible seam. **Force aspect ratio** (Settings) crops or extends the
  loaded image to the chosen ratio before the pass.
- **Inpaint / Outpaint**, one editor for three modes:
  - **Brush (inpaint)**: paint the area to redo. The mask is max-pooled on the 16 px
    grid of the latents; outside it the original is kept pixel for pixel (checked by the
    GPU end-to-end test), the seams are blended. A full-white mask is a plain img2img.
  - **Expand sides (outpaint)**: tick Left / Right / Top / Bottom (about 30 % each) or
    **Center (all sides)**. The new bands start from pure noise under the mask.
  - **Reframe**: a **Target ratio** and a **Fit** (contain = the whole image kept, bands
    added; cover = cropped). Same machinery as outpaint.
  Inpaint, outpaint and reframe work on an area capped near 1 Mpx and paste the result
  back at the original size, so they run faster than a full 1.9 Mpx render.
- **Detail faces / Detail hands** (Advanced): after a render or an upscale, detect the
  faces (and hands), inpaint each at its own denoise, paste back with a feathered edge.
  Measured: a face retouched (teeth, eyes) with the rest of the poster untouched.
- **Upscale after generate**: chains each txt2img image through the upscale pipeline
  with no manual step; works after an edit too.

**Layouts and the emulated tools.** A layout is written for one size. In img2img,
inpaint and refine it is rescaled to the image. In outpaint and reframe it is **placed**
on the original image inside the new canvas (never stretched), the new bands are
declared in it as empty-background items and "No other text." is forced; a plain-text
prompt gets the same sentence naming the sides. This came from measurements: with a
plain prompt the new areas filled with invented words, with the stretched layout the
headline continued across the bands. The log says what it did.

**What to expect.** The first render after a new prompt costs 30 to 40 s more than the
others (the text encoder comes to the card and leaves again, the DiT is parked in RAM
meanwhile) and peaks at 19 to 23.5 GB depending on the prompt's length; the next ones
stay near 15 GB. Stop renders the partial image within a second or two.

## Editing with references

Two ways, both local.

**Edit (references) tab (lot 4b)**: MOSS-Image's base weights never saw a reference,
but the community **Moss-Image-Edit LoRA** (chinoll/Moss-Image-Edit, repacked by
Comfy-Org, rank 64) teaches it the "reference prefix" contract: the image to edit (Ref 1)
and up to 3 more references are VAE-encoded and put in front of the sequence at t = 0,
seeing only each other; the target and the instruction see everything; the result takes
Ref 1's size (tick **Output size from Settings** in the tab to use the Settings width /
height and ratio instead: the LoRA was trained on same-size pairs, texts may hold less
well). The app rewrote that contract on its vendored DiT (nothing of the GPLv3
node code is reused) and caches the prefix's keys and values once per edit, so the
references cost little per step. The LoRA is fetched once (about 300 MB) on the first
Edit and applied for edits only. **Its author states no licence for the weights** (the
training code is GPLv3): check before a client deliverable. References enter at their
own size (long side capped at 2048, `moss_edit_ref_max_edge`): the LoRA was trained on
pairs of the same size, and on the 5090 a poster read at 1024 came back with garbled
texts. The LoRA was trained with ONE reference (the image to edit): extra references are
experimental, their content leaks into the result. Texts are redrawn, not copied: check
them. The author's inference script uses 50 steps and CFG 4 (measured better than CFG 3
on the texts). Measured on the RTX 5090 (50 steps): the 1152x1632 poster edited with its texts exact in 125 to 170 s (peak 24 GB, the prefix's K/V cache 3.5 GB), a 1024 logo put on a beach in 62 s; the same poster read at 1024 came back with broken letters.

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

## Comparing with Ideogram 4: crispz-ideogram

The **Compare in crispz-ideogram** tab renders the main prompt with crispz-ideogram on
the same machine, at the same size and seed, through its protocol v1 command line: a
MOSS layout is translated to the Ideogram 4 caption schema (same boxes on the same
0..1000 grid, texts verbatim, graphic-design defaults for the style fields Ideogram
wants), plain text goes as it is. The prompt as sent is shown in the tab and kept next
to the result (`handoff/<date>/ideogram_prompt.json`). Set `crispz_ideogram_dir`; the
running-app or cold-process rule and the port note above apply the same way.
**Ideogram 4's weights are non-commercial: the comparison is for your eyes, never a
client deliverable.**

## Character and style catalog

The **MOSS-Image Character Catalog** (hide1 on Hugging Face: 33,935 characters from
3,953 series and 314 styles, each rendered with MOSS-Image and tagged with its prompt,
its tags and its facets: hair, eyes, gender, outfit) is browsable from the app. Nothing
is fetched until you ask for it, and nothing leaves the machine but the downloads you
click.

1. **Advanced panel (tick "Advanced"), tab "Styles", under the style list: accordion
   "Character catalog (MOSS-Image, 34,249 samples)"** (closed by default). Click
   **Download / update the catalog (7 MB, huggingface.co)**. The index lands in
   `cache/moss/`, the log writes a `CATALOG:` line, the status under the button says
   what is on disk. A second click only updates it.
2. **Asset Browser**: two new tabs, **Characters** and **Styles**. The two buttons of
   the accordion, **Open Asset Browser: Characters** and **Open Asset Browser:
   Styles**, open it straight on that tab (the general "Open Asset Browser" button
   of the Save tab opens it on Outputs). Folders on the left are the series (one per
   copyright), the search box matches names, tags and prompts, the cards come 600 at a
   time. A card opens the sheet: name, series, facets, prompt, tags, the full
   "generation prompt" (the whole scene the catalog rendered) and a link to the sample
   on Hugging Face.
3. **Images**: by default the cards show no picture. Tick **Show thumbnails from
   huggingface.co** in the accordion to let the browser load them from the Hub (that is
   the only case where it talks to huggingface.co while you browse). Or keep them
   locally: **Download the thumbnails (34,249 files, ~270 MB)** or **Download the full
   renders (~1.8 GB)** fetch the dataset into `cache/moss/character_catalog/` (resumable:
   a second click only takes what is missing; about an hour per batch at the anonymous
   Hub rate, faster with an `hf_token` in the config). Local files always win over the
   Hub.
4. **Send to crispz-moss**: on a sheet, the buttons **prompt**, **tags**, **generation
   prompt** and **append tags** put the text into the app's prompt box (switch back to
   the crispz-moss tab: the box is filled and Gradio has seen it). The same bridge sends
   an output's prompt, a LoRA's trigger words or one of its CivitAI example prompts.
5. **Autocompletion**: the catalog's character, series and style names are written to
   `tags/moss_catalog.csv` and offered in the prompt box from the next launch, like any
   CSV of `tags/`.

**Rights, shown in the accordion, the browser and every sheet**: the catalog is
uncurated, and its characters, artworks and renders are third-party intellectual
property (the catalog's own NOTICE). The author's data is MIT. Use it for reference,
styles and prompt ideas, not for a client deliverable.

## Command line

`python app.py --help`, the cheat sheet in [README_CLI.md](README_CLI.md), and the
family protocol: `czp caps` / `czp gen --spec spec.json` (routes to the running app,
else runs locally). `caps` says what is emulated (`supports.emulated`).

## Tests

`.venv\Scripts\python tools\run_tests.py` (CPU, a tiny random MOSS-Image, one process
per test file: what the CI runs). `tests\test_moss_e2e.py` is the GPU end-to-end run
(load, 1024, 2 Mpx, img2img, inpaint, Stop; skips without CUDA, timings in
`tests/e2e_timings.txt`; pytest is not in `requirements.txt`, install it alone:
`.venv\Scripts\python -m pip install pytest`). Passed on the RTX 5090 on 05/11/2026 in
4 min 56: load 35 s, 1024 in 83 s (first render of the process, prompt and negative
freshly encoded), 1456x1456 in 109 s, img2img 0.35 in 15 s, inpaint in 47 s, Stop in
2.8 s, peaks 15 to 20.5 GB. The lot scripts in `tools/` (`gpu_lot1_check.py` to
`gpu_lot4b_check.py`) measure each part on the card, GPU free.

## Pinokio

A 1-click launcher (install, start, update, reset; the same venv rules as `install.bat`)
is published as its own repo: https://github.com/mikecastrodemaria/crispz-moss.pinokio.

## Credits

OpenMOSS for MOSS-Image and its pipeline (Apache 2.0, vendored in
`third_party/moss_image/`); Qwen (Qwen3.5-9B, Apache 2.0); Black Forest Labs (the FLUX.2
VAE); hide1 for the MOSS-Image Character Catalog (MIT for the data; the characters are
third-party IP); diffusers and transformers; vtracer (MIT); rembg (MIT); the crispz family
(crispz-klein, crispz-flux3, crispz-ideogram) this studio is forked from. How and why:
[FORK.md](FORK.md); the full work log: [CLAUDE.md](CLAUDE.md).
