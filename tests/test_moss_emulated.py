"""Lot 3: img2img and inpaint EMULATED on MOSS-Image (tiny random pipeline, CPU).

What the acceptance asks (CLAUDE.md lot 3): reduced steps, the area outside the mask
kept, a Stop that still returns the partial image. Plus: the schedule really is the tail
of MOSS's own (shifted, snapped), the latents are the exact inverse of the decode, the
scheduler is left as it was, the live preview counts the steps that run.

Run:  .venv/Scripts/python -m pytest tests/test_moss_emulated.py -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import cz_moss  # noqa: E402
from moss_tiny import tiny_pipe  # noqa: E402

_PIPE = None


def _pipe():
    global _PIPE
    if _PIPE is None:
        _PIPE = tiny_pipe()
    return _PIPE


def _gen(s=0):
    return torch.Generator().manual_seed(s)


def _init(w=256, h=256):
    img = Image.new("RGB", (w, h), (30, 60, 200))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w // 2, h], fill=(230, 40, 40))
    d.ellipse([w // 4, h // 4, 3 * w // 4, 3 * h // 4], fill=(250, 220, 30))
    return img


def _count_calls(pipe):
    calls = []
    tr = pipe._pipe.transformer
    h = tr.register_forward_pre_hook(lambda m, a: calls.append(1))
    return calls, h


def test_img2img_runs_the_tail_of_the_native_schedule():
    pipe = _pipe()
    seen = {}
    real_call = pipe._pipe.__class__.__call__

    def spy(self, *a, **kw):
        seen.update(kw)
        out = real_call(self, *a, **kw)
        seen["timesteps"] = self.scheduler.timesteps.clone()
        return out
    calls, h = _count_calls(pipe)
    pipe._pipe.__class__.__call__ = spy
    try:
        out = pipe(prompt="a poster", image=_init(320, 256), strength=0.5,
                   num_inference_steps=8, generator=_gen(1), guidance_scale=1.0).images[0]
    finally:
        pipe._pipe.__class__.__call__ = real_call
        h.remove()
    assert out.size == (320, 256)                      # the image's own size
    full = cz_moss.native_sigmas(8)
    n = len(full)
    assert len(calls) == n - int(n - n * 0.5)          # the last half of the steps
    ts = seen["timesteps"].float() / 1000.0
    assert torch.allclose(ts, torch.tensor(full[int(n - n * 0.5):]), atol=1e-4), (ts, full)
    assert seen["latents"].shape == (1, 128, 256 // 16, 320 // 16)
    assert seen["sigmas"] == full[int(n - n * 0.5):]
    # the next plain call runs the native schedule again (nothing left behind)
    seen.clear()
    pipe._pipe.__class__.__call__ = spy
    try:
        pipe(prompt="x", width=256, height=256, num_inference_steps=8, generator=_gen(),
             guidance_scale=1.0)
    finally:
        pipe._pipe.__class__.__call__ = real_call
    assert torch.allclose(seen["timesteps"].float() / 1000.0, torch.tensor(full), atol=1e-4)


def test_strength_extremes():
    pipe = _pipe()
    calls, h = _count_calls(pipe)
    try:
        pipe(prompt="x", image=_init(), strength=0.0, num_inference_steps=8, generator=_gen(),
             guidance_scale=1.0)
        assert len(calls) == 1                         # a 0 denoise still touches it
        calls.clear()
        pipe(prompt="x", image=_init(), strength=1.0, num_inference_steps=8, generator=_gen(),
             guidance_scale=1.0)
        assert len(calls) == len(cz_moss.native_sigmas(8))
    finally:
        h.remove()


def test_low_strength_stays_closer_to_the_init():
    pipe = _pipe()
    init = _init()
    a = np.asarray(init, dtype=np.float32)
    low = pipe(prompt="x", image=init, strength=0.2, num_inference_steps=10,
               generator=_gen(3), guidance_scale=1.0).images[0]
    high = pipe(prompt="x", image=init, strength=1.0, num_inference_steps=10,
                generator=_gen(3), guidance_scale=1.0).images[0]
    d_low = np.abs(np.asarray(low, dtype=np.float32) - a).mean()
    d_high = np.abs(np.asarray(high, dtype=np.float32) - a).mean()
    assert d_low < d_high, (d_low, d_high)


def test_encode_latents_is_the_exact_inverse_of_the_decode():
    """pack_latents undoes what decode_latents does before the VAE (bn de-normalization,
    the (c//4, 2, 2) unpatchify in FLUX order), on random tokens, bit for bit."""
    pipe = _pipe()
    vae = pipe._pipe.vae
    z = torch.randn(2, 128, 5, 7)
    bn, eps = vae.bn, vae.config.batch_norm_eps
    r = (z * (bn.running_var + eps).sqrt().view(1, -1, 1, 1) + bn.running_mean.view(1, -1, 1, 1))
    b, c, hh, ww = r.shape
    r = r.reshape(b, c // 4, 2, 2, hh, ww).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * hh, 2 * ww)
    back = cz_moss.MossPipe.pack_latents(r, vae)
    assert torch.allclose(back, z, atol=1e-5)
    x0 = pipe._encode_latents(_init(320, 256), 320, 256, "cpu")
    assert x0.shape == (1, 128, 16, 20)


def test_inpaint_keeps_the_area_outside_the_mask():
    """Checked on the LATENTS the decoder receives: outside the mask, after the last step
    (sigma 0), they are the init image's own latents, exactly; inside, they moved.
    (Pixels cannot be compared: the FLUX.2 VAE decoder has a mid-block self-attention,
    every decoded pixel depends on the whole latent.)"""
    pipe = _pipe()
    init = _init(256, 256)
    mask = Image.new("L", (256, 256), 0)
    ImageDraw.Draw(mask).rectangle([160, 0, 255, 255], fill=255)        # right side
    last = {}

    def grab(p, i, t, kw):
        last["lat"] = kw["latents"].clone()
        return kw
    pipe(prompt="x", image=init, mask_image=mask, strength=1.0, num_inference_steps=6,
         generator=_gen(5), guidance_scale=1.0, callback_on_step_end=grab)
    x0 = pipe._encode_latents(init, 256, 256, "cpu")[0]
    m = cz_moss.MossPipe._mask_latents(mask, 256, 256, "cpu")[0, 0] > 0
    lat = last["lat"][0]
    assert m.sum() > 0 and (~m).sum() > 0
    assert torch.allclose(lat[:, ~m], x0[:, ~m], atol=1e-5)
    assert (lat[:, m] - x0[:, m]).abs().mean() > 1e-2


def test_full_white_mask_is_plain_img2img():
    pipe = _pipe()
    full = Image.new("L", (256, 256), 255)
    a = pipe(prompt="x", image=_init(), mask_image=full, strength=0.5, num_inference_steps=4,
             generator=_gen(7), guidance_scale=1.0, output_type="np").images
    b = pipe(prompt="x", image=_init(), strength=0.5, num_inference_steps=4,
             generator=_gen(7), guidance_scale=1.0, output_type="np").images
    assert np.abs(np.asarray(a) - np.asarray(b)).max() < 1e-5


def test_stop_returns_the_partial_image():
    pipe = _pipe()
    calls, h = _count_calls(pipe)

    def stop(p, i, t, kw):
        if i == 1:
            p._interrupt = True
        return kw
    mask = Image.new("L", (256, 256), 0)
    ImageDraw.Draw(mask).rectangle([0, 0, 128, 128], fill=255)
    try:
        out = pipe(prompt="x", image=_init(), mask_image=mask, strength=1.0,
                   num_inference_steps=8, generator=_gen(), guidance_scale=1.0,
                   callback_on_step_end=stop)
    finally:
        h.remove()
    assert len(calls) == 2 and out.images[0].size == (256, 256)


def test_mask_latents_max_pool():
    m = Image.new("L", (64, 32), 0)
    m.putpixel((20, 5), 255)                         # one pixel in token (row 0, col 1)
    t = cz_moss.MossPipe._mask_latents(m, 64, 32, "cpu")
    assert t.shape == (1, 1, 2, 4) and t[0, 0, 0, 1] == 1.0 and t.sum() == 1.0


def test_an_image_without_strength_is_still_refused_by_name():
    with pytest.raises(NotImplementedError, match="crispz-flux3"):
        _pipe()(prompt="x", image=_init(), num_inference_steps=2)


def test_progress_and_preview_count_the_steps_that_run():
    """The bar iterates the truncated timesteps (right by itself) and the live preview
    reads _cz_run_steps: both say the steps the emulated img2img really runs; the
    counter is cleared after the call, the scheduler is not wrapped."""
    pipe = _pipe()
    inner = pipe._pipe
    lengths, previews = [], []
    real_bar = type(inner).progress_bar

    def spy_bar(self, iterable=None, total=None):
        lengths.append(len(iterable) if iterable is not None else total)
        return real_bar(self, iterable=iterable, total=total)

    def cb(p, i, t, kw):
        previews.append(getattr(p, "_cz_run_steps", None))
        return kw
    type(inner).progress_bar = spy_bar
    try:
        pipe(prompt="x", image=_init(), strength=0.5, width=256, height=256,
             num_inference_steps=8, generator=_gen(), guidance_scale=1.0,
             callback_on_step_end=cb)
    finally:
        type(inner).progress_bar = real_bar
    n = len(cz_moss.native_sigmas(8))
    run = n - int(n - n * 0.5)
    assert lengths == [run], lengths
    assert previews and set(previews) == {run} and len(previews) == run
    assert inner._cz_run_steps is None
    assert "set_timesteps" not in vars(inner.scheduler)


def test_negative_prompt_and_batch_work_in_img2img():
    """CFG (a real negative) and a batch of 2 go through the emulated path: two images
    of the init's size, the latents expanded to the batch."""
    pipe = _pipe()
    out = pipe(prompt="x", negative_prompt="blurry", image=_init(), strength=0.5,
               num_inference_steps=4, generator=_gen(2), num_images_per_prompt=2,
               guidance_scale=3.0).images
    assert len(out) == 2 and out[0].size == (256, 256)
