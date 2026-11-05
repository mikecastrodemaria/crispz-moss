# crispz-moss.pinokio

1-click [Pinokio](https://pinokio.computer) launcher for
**[crispz-moss](https://github.com/mikecastrodemaria/crispz-moss)**, the **MOSS-Image**
studio of the crispz family: text in images and design (posters, logos, packaging, slides,
dense typography), Fooocus-style, 100% local.

## Licence first

MOSS-Image's weights and pipeline code are **Apache 2.0** (OpenMOSS): commercial use is
allowed, client deliverables made locally are fine, keep the attribution. Its text
encoder (Qwen3.5-9B) and its VAE (FLUX.2 klein 4B) are Apache 2.0 too. Community LoRAs
each carry their own licence; the app shows it next to the one it fetches (the
Moss-Image-Edit LoRA's author states none for its weights). The launcher and the app's
code are CC BY-NC 4.0.

## What it does

- **Install** clones `mikecastrodemaria/crispz-moss` into `app/`, creates a
  **Python 3.11** venv, installs `requirements.txt` (diffusers 0.42.1 for the FLUX.2 VAE,
  transformers 5 for the Qwen3.5 encoder, outlines for the local magic prompt, vtracer
  for local vectorization, bitsandbytes for the optional nf4 encoder), Pillow 13 apart,
  the extras (rembg for Remove BG), the Face Swap deps and PyTorch (torch 2.11 + CUDA
  cu128 on NVIDIA). Ends with a check that the Qwen3.5 encoder, the FLUX.2 VAE, outlines
  and vtracer import. The MOSS-Image pipeline itself is vendored in the app.
- **Start** runs `python app.py` and opens the Gradio Web UI.
- **Update** force-updates the launcher and the app to their `origin/main`, refreshes
  deps, reasserts the torch build, re-runs the check.
- **Reset** removes `app/` (and its venv) to reinstall from scratch.

## Before the first render

Nothing to accept, no token: the weights are public. The first render downloads about
27 GB (DiT 10.2 GB, Qwen3.5-9B text encoder 16.8 GB, VAE 0.25 GB). The first magic
prompt in local mode adds a 2 GB head (Ollama, the default, needs none), the first Remove
BG the 176 MB u2net model, the first edit with references the 300 MB Moss-Image-Edit LoRA.

## Requirements

- An **NVIDIA GPU**: measured on an RTX 5090 (32 GB). With the encoder-only offload
  (`auto`), the DiT and the VAE stay on the card (9.7 GB) and the 16.8 GB encoder comes
  in only for a new prompt (23.5 GB peak then, 15 GB once the prompt is cached). On a
  24 GB card, set the encoder to nf4 in the app (`moss_precision`: 21 GB flat).
- **About 48 GB of RAM** is comfortable: the bf16 encoder waits in RAM between prompts.

Measured on an RTX 5090, 50 steps, CFG 3:

| | |
|---|---|
| Load from the cache | 25 to 30 s |
| 1024x1024 | 43 s, 15 GB peak |
| 1456x1456 (2 Mpx) | 102 s, 15.6 GB peak |
| Edit with references, 1152x1632 poster | 125 to 170 s, 24 GB peak |

## What is local, what is not

Everything runs on your machine; the app has no cloud function. Ollama, when used for the
magic prompt or Describe, is a local server. The hand-overs to crispz-flux3 (editing) and
crispz-ideogram (comparison) call those apps on the same machine.
