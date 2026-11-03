# pipeline_moss_image.py from OpenMOSS-Team/MOSS-Image (Apache License 2.0)

Copied UNMODIFIED from https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT
(`pipeline_moss_image.py` at the repo root, revision f972bd4b4b9e3852133cc5da954d87bc32cd8fb9,
2026-10-21), identical byte for byte to the one in `OpenMOSS-Team/MOSS-Image-Base`
(revision 85975793d8b3d432f1b6e27f352abaa4c2a4c0d3). SHA-256 of the file:

    1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31

The licence (Apache 2.0, the repo's own `LICENSE`) is next to this file. It covers the
MODEL WEIGHTS of MOSS-Image as well (same repo, same LICENSE file).

Why it is vendored: the official way to run MOSS-Image in diffusers is
`DiffusionPipeline.from_pretrained(..., custom_pipeline="pipeline_moss_image",
trust_remote_code=True)`, which downloads and executes this very file from the Hub at
run time. crispz-moss never runs remote code: the file is kept here, read once, and a
new upstream version shows up as a readable diff. `cz_moss` subclasses the classes
defined here (MOSSImageTransformer, MOSSImageScheduler, MOSSImagePipeline) and adds
what crispz needs (a step-end callback, Stop, a custom sigma schedule for the emulated
img2img, the encoder-only offload) WITHOUT touching this file.

To check the copy against upstream:

    python -c "import hashlib;print(hashlib.sha256(open('third_party/moss_image/pipeline_moss_image.py','rb').read()).hexdigest())"
