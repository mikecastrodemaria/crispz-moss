"""cz_moss.MossPipe on a tiny random MOSS-Image (CPU, no download but the tokenizer).

Lot 1: text to image through crispz's calling convention, the subclassed pipeline equal
to the vendored one, callback and Stop, the negative prompt reaching the encoder (and
cached), size fitting, the named refusals (img2img / inpaint until lot 3, editing, LoRAs
until lot 4), the encoder-only offload, the family detection the LoRA and checkpoint
guards rely on, and the ComfyUI / CivitAI single files (bf16 and int8 ConvRot, fused
qkv) read back to the very same weights.
Run: python -m pytest tests/test_moss_engine.py -q
"""
import os
import sys
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cz_moss  # noqa: E402
from moss_tiny import (tiny_pipe, tiny_transformer, tiny_text_encoder,  # noqa: E402
                       save_tiny_comfy_file, save_tiny_comfy_text_encoder)

_PIPE = None


def _pipe():
    global _PIPE
    if _PIPE is None:
        _PIPE = tiny_pipe()
    return _PIPE


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def test_txt2img_size_and_callback():
    seen = []
    shapes = []

    def cb(p, i, t, kw):
        seen.append(i)
        shapes.append(tuple(kw["latents"].shape))
        return kw
    out = _pipe()(prompt="a poster that says HELLO", width=256, height=512,
                  num_inference_steps=3, generator=_gen(), callback_on_step_end=cb)
    assert out.images[0].size == (256, 512)
    assert seen == [0, 1, 2]
    assert shapes[0] == (1, 128, 32, 16)            # (B, 128, h/16, w/16)


def test_deterministic_seed():
    a = np.asarray(_pipe()(prompt="x", width=256, height=256, num_inference_steps=2,
                           generator=_gen(5)).images[0])
    b = np.asarray(_pipe()(prompt="x", width=256, height=256, num_inference_steps=2,
                           generator=_gen(5)).images[0])
    assert (a == b).all()


def test_subclass_renders_the_vendored_image():
    """MossPipeline's loop is upstream's: same seed, same settings, the same pixels as
    the untouched MOSSImagePipeline (built on the same tiny components)."""
    raw = tiny_pipe(raw=True)
    ours = tiny_pipe()
    kw = dict(width=256, height=256, num_inference_steps=4, guidance_scale=3.0,
              negative_prompt="blurry", uncond_skip_layers=[1], output_type="np")
    a = raw(prompt="a red sign", generator=_gen(11), **kw).images
    b = ours(prompt="a red sign", generator=_gen(11), **kw).images
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_fit_size():
    assert cz_moss.fit_size(1000, 1000) == (992, 992)
    assert cz_moss.fit_size(100, 100) == (256, 256)
    assert cz_moss.fit_size(4096, 4096) == (2048, 2048)
    w, h = cz_moss.fit_size(2048, 2048)
    assert w * h <= cz_moss.MAX_PIXELS
    w, h = cz_moss.fit_size(2048, 1536)                   # 3 Mpx: kept (under 4)
    assert (w, h) == (2048, 1536)
    w, h = cz_moss.fit_size(2048, 352)                    # a 6:1 banner stays possible
    assert (w, h) == (2048, 352)


def test_interrupt_returns_a_partial_image():
    def cb(p, i, t, kw):
        if i == 0:
            p._interrupt = True
        return kw
    out = _pipe()(prompt="x", width=256, height=256, num_inference_steps=4,
                  generator=_gen(), callback_on_step_end=cb)
    assert out.images[0].size == (256, 256)
    out = _pipe()(prompt="x", width=256, height=256, num_inference_steps=2, generator=_gen())
    assert out.images[0].size == (256, 256)          # the next call starts clean


def test_img2img_inpaint_and_edit_are_refused_by_name():
    with pytest.raises(NotImplementedError, match="lot 3"):
        _pipe()(prompt="x", width=256, height=256, num_inference_steps=1,
                image=Image.new("RGB", (8, 8)), strength=0.5)
    with pytest.raises(NotImplementedError, match="lot 3"):
        _pipe()(prompt="x", width=256, height=256, num_inference_steps=1,
                image=Image.new("RGB", (8, 8)), mask_image=Image.new("L", (8, 8)))
    with pytest.raises(NotImplementedError, match="crispz-flux3"):
        _pipe()(prompt="x", width=256, height=256, num_inference_steps=1,
                image=Image.new("RGB", (8, 8)))
    with pytest.raises(NotImplementedError, match="lot 4"):
        _pipe().load_lora_weights("x.safetensors")


def test_negative_prompt_reaches_the_encoder_and_is_cached():
    """A real CFG: the negative changes the image, is encoded ONCE for the same text
    (the cache), and guidance 1.0 skips it altogether."""
    pipe = tiny_pipe()
    calls = []
    real = pipe._pipe.encode_prompt

    def spy(prompt, *a, **k):
        calls.append(prompt)
        return real(prompt, *a, **k)
    pipe._pipe.encode_prompt = spy
    kw = dict(width=256, height=256, num_inference_steps=2, output_type="np")
    a = pipe(prompt="a cat", negative_prompt="", generator=_gen(1), **kw).images
    b = pipe(prompt="a cat", negative_prompt="blurry, watermark", generator=_gen(1), **kw).images
    c = pipe(prompt="a cat", negative_prompt="blurry, watermark", generator=_gen(1), **kw).images
    assert not np.array_equal(np.asarray(a), np.asarray(b))
    assert np.array_equal(np.asarray(b), np.asarray(c))
    assert calls == ["", "a cat", "blurry, watermark", "a cat", "a cat"], calls
    calls.clear()
    pipe(prompt="a cat", negative_prompt="blurry", guidance_scale=1.0, generator=_gen(1), **kw)
    assert calls == ["a cat"], calls


def test_encode_prompt_feeds_the_pipeline_like_the_text():
    """cz_pipeline's embedding cache keeps the tuple's first element and passes it back
    as prompt_embeds: the image must not change."""
    pipe = _pipe()
    kw = dict(width=256, height=256, num_inference_steps=2, negative_prompt="",
              output_type="np")
    ref = pipe(prompt="GRAND OPENING", generator=_gen(2), **kw).images
    emb = pipe.encode_prompt(prompt="GRAND OPENING", device=torch.device("cpu"))
    assert isinstance(emb, tuple) and emb[0].dim() == 3 and emb[1].dtype == torch.bool
    out = pipe(prompt=None, prompt_embeds=emb[0], generator=_gen(2), **kw).images
    assert np.array_equal(np.asarray(ref), np.asarray(out))


def test_skip_layers_are_clamped_to_the_model():
    """The official uncond skip (block 22) does not exist on a 2-block test model: dropped,
    said once, never a ValueError from the pipeline."""
    pipe = _pipe()
    assert pipe._skip_layers_for(None) == ()
    assert pipe._skip_layers_for([1, 7]) == (1,)
    assert cz_moss.UNCOND_SKIP_LAYERS == (22,)


def test_custom_sigmas_run_the_given_schedule():
    """`sigmas` replaces the native schedule with no shift on top (lot 3 truncates the
    native one this way): the steps seen are the sigmas given."""
    pipe = _pipe()
    seen = []
    sig = cz_moss.native_sigmas(6)
    assert len(sig) == 6 and sig[0] > sig[-1] > 0
    tail = sig[3:]
    pipe(prompt="x", width=256, height=256, num_inference_steps=6, generator=_gen(),
         sigmas=tail, callback_on_step_end=lambda p, i, t, kw: seen.append(float(t)) or kw)
    assert len(seen) == 3
    want = [s * 1000 for s in tail]
    assert all(abs(a - b) < 1e-3 for a, b in zip(seen, want)), (seen, want)


def test_family_detection():
    full = ["video_embedder.weight", "layers.0.proj.q.weight", "shared_block_modulation.weight"]
    assert cz_moss.family_from_keys(full) == "moss"
    comfy = ["layers.0.proj.qkv.weight", "final_linear_video.weight"]
    assert cz_moss.family_from_keys(comfy) == "moss"
    ideo = ["llm_cond_proj.weight", "layers.0.attention.to_q.weight"]
    assert cz_moss.family_from_keys(ideo) == "ideogram"
    assert cz_moss.family_from_keys(["early_stream_modulations.image.lin.weight"]) == "flux3"
    assert cz_moss.family_from_keys(["double_blocks.0.img_attn.qkv.weight"]) == "flux2"
    # LoRAs: the MOSS edit LoRA (no '.weight' suffix), the control LoRA, an Ideogram one
    lora_moss = {"layers.0.proj.q.lora_A": {"shape": [64, 3072]}}
    assert cz_moss.family_from_header(lora_moss) == "moss"
    lora_ctrl = {"text_embedder.blocks.0.qkv.lora_A.weight": {"shape": [64, 3072]}}
    assert cz_moss.family_from_header(lora_ctrl) == "moss"
    lora_ideo = {"transformer.layers.0.attention.to_q.lora_A.weight": {"shape": [16, 4608]}}
    assert cz_moss.family_from_header(lora_ideo) == "ideogram"


def test_repo_detection():
    assert cz_moss.is_engine_repo(cz_moss.DEFAULT_REPO)
    assert cz_moss.is_engine_repo("OpenMOSS-Team/MOSS-Image-Base")
    assert cz_moss.is_engine_repo("someone/moss-image-mirror")
    assert not cz_moss.is_engine_repo("ideogram-ai/ideogram-4-nf4-diffusers")
    assert not cz_moss.is_engine_repo("black-forest-labs/flux-3-image-klein")
    assert cz_moss.is_base_repo("OpenMOSS-Team/MOSS-Image-Base")
    assert not cz_moss.is_base_repo(cz_moss.DEFAULT_REPO)


def test_hadamard_is_its_own_inverse():
    H = cz_moss._hadamard(16)
    assert torch.allclose(H @ H, torch.eye(16), atol=1e-6)
    w = torch.randn(6, 64)
    assert torch.allclose(cz_moss.rotate_convrot(cz_moss.rotate_convrot(w, 16), 16), w, atol=1e-5)
    with pytest.raises(ValueError):
        cz_moss._hadamard(8)


@pytest.mark.parametrize("int8", [False, True])
def test_comfy_single_file_transformer_reads_back(int8):
    """A ComfyUI / CivitAI single file (fused qkv, comfy_attention blobs, int8 ConvRot
    with per-row scales) loads into the vendored class: bf16 exact, int8 within the
    quantization error, and the same shapes everywhere."""
    tr = tiny_transformer(seed=3)
    with tempfile.TemporaryDirectory() as d:
        path = save_tiny_comfy_file(os.path.join(d, "moss_tiny.safetensors"), tr, int8=int8)
        assert cz_moss.single_file_refusal(path) is None
        fmts = cz_moss.quant_formats(path)
        assert bool(fmts) == int8
        loaded = cz_moss.load_transformer("unused", torch.bfloat16, single_file=path)
    want = {k: v.to(torch.bfloat16) for k, v in tr.state_dict().items()}
    got = loaded.state_dict()
    assert set(got) == set(want), (set(got) ^ set(want))
    for k in want:
        if k == "rope.bands":
            assert torch.equal(got[k], tr.rope.bands)
            continue
        a, b = got[k].float(), want[k].float()
        if int8 and k.startswith("layers.") and ".proj." in k:
            assert (a - b).abs().max() <= 0.02 * b.abs().max() + 1e-3, k
        else:
            assert torch.equal(a, b), k
    assert loaded.rope.bands.dtype == torch.float32 and loaded.rope.bands.device.type == "cpu"


def test_a_plain_fp8_single_file_is_cast_back():
    """CivitAI's 'fp8' builds: an e4m3fn cast with no weight_scale (the ComfyUI way), or
    a scaled one. Both read back within fp8 precision; the fused qkv is split too."""
    from safetensors.torch import save_file
    tr = tiny_transformer(seed=6)
    sd = {k: v.detach().clone() for k, v in tr.state_dict().items()}
    out = {}
    for k, v in sd.items():
        if ".proj.q.weight" in k:
            stem = k[:-len("q.weight")]
            w = torch.cat([sd[stem + "q.weight"], sd[stem + "k.weight"], sd[stem + "v.weight"]], 0)
            out[stem + "qkv.weight"] = w.to(torch.float8_e4m3fn)
        elif ".proj.k.weight" in k or ".proj.v.weight" in k:
            continue
        elif k.startswith("layers.") and ".proj." in k:
            out[k] = v.to(torch.float8_e4m3fn)
        else:
            out[k] = v.to(torch.bfloat16) if k != "rope.bands" else v
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "moss_fp8.safetensors")
        save_file({k: v.contiguous() for k, v in out.items()}, path)
        assert cz_moss.single_file_refusal(path) is None
        loaded = cz_moss.load_transformer("unused", torch.bfloat16, single_file=path)
    for k, v in tr.state_dict().items():
        a, b = loaded.state_dict()[k].float(), v.float()
        tol = 0.07 if k.startswith("layers.") and ".proj." in k else 0.01
        assert (a - b).abs().max() <= tol * b.abs().max() + 1e-3, k


def test_comfy_single_file_refusals():
    """A w4a8 GPTQ build and a LoRA are refused by name, before any weight is read."""
    import json
    from safetensors.torch import save_file
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "w4.safetensors")
        blob = json.dumps({"format": "asym_w4a8_int8", "group_size": 16}).encode()
        save_file({"layers.0.proj.qkv.weight": torch.zeros(96, 16, dtype=torch.int8),
                   "layers.0.proj.qkv.weight_codebook": torch.zeros(16),
                   "layers.0.proj.qkv.comfy_quant": torch.frombuffer(bytearray(blob),
                                                                     dtype=torch.uint8).clone(),
                   "video_embedder.weight": torch.zeros(32, 128, dtype=torch.bfloat16)}, p)
        why = cz_moss.single_file_refusal(p)
        assert why and "w4a8" in why and "bf16 or int8" in why
        lora = os.path.join(d, "lora.safetensors")
        save_file({"layers.0.proj.q.lora_A.weight": torch.zeros(4, 32, dtype=torch.bfloat16)},
                  lora)
        assert "LoRA" in cz_moss.single_file_refusal(lora)
        vae = os.path.join(d, "vae.safetensors")
        save_file({"decoder.conv_in.weight": torch.zeros(4, 4, dtype=torch.bfloat16)}, vae)
        assert cz_moss.single_file_refusal(vae) is None       # not ours: left to the caller


@pytest.mark.parametrize("int8", [False, True])
def test_comfy_single_file_text_encoder_reads_back(int8):
    enc = tiny_text_encoder(seed=4)
    with tempfile.TemporaryDirectory() as d:
        path = save_tiny_comfy_text_encoder(os.path.join(d, "qwen35_tiny.safetensors"), enc,
                                            int8=int8)
        assert cz_moss.is_text_encoder_file(path)
        loaded = cz_moss.load_single_file_text_encoder(path, torch.bfloat16, config=enc.config)
    want = {k: v.to(torch.bfloat16).float() for k, v in enc.state_dict().items()}
    got = {k: v.float() for k, v in loaded.state_dict().items()}
    assert set(got) == set(want)
    for k in want:
        if int8 and k.endswith("_proj.weight") and want[k].dim() == 2 and want[k].shape[1] % 16 == 0:
            assert (got[k] - want[k]).abs().max() <= 0.02 * want[k].abs().max() + 1e-3, k
        else:
            assert torch.equal(got[k], want[k]), k
    # It runs (the rotary buffers were rebuilt), and encodes like the original.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("OpenMOSS-Team/MOSS-Image-SFT", subfolder="tokenizer")
    ids = tok("a sign", return_tensors="pt")
    with torch.no_grad():
        a = enc.to(torch.bfloat16)(**ids, use_cache=False).last_hidden_state.float()
        b = loaded(**ids, use_cache=False).last_hidden_state.float()
    assert a.shape == b.shape
    tol = 0.25 if int8 else 1e-6
    assert (a - b).abs().max() <= tol * (a.abs().max() + 1e-6), (a - b).abs().max()


def test_model_offload_moves_only_the_text_encoder():
    """'model' offload on MOSS = the encoder alone leaves the card: in before the prompts
    are encoded, back at the first DiT forward, never per step; no accelerate hook
    anywhere. Execution device = CPU here: what is checked is WHEN the encoder moves,
    that the DiT never does, and that the image does not change."""
    pipe = tiny_pipe()
    kw = dict(width=256, height=256, num_inference_steps=3, output_type="np",
              negative_prompt="blurry")
    ref = pipe(prompt="GRAND OPENING", generator=_gen(3), **kw).images
    raw = pipe._pipe
    pipe.enable_model_cpu_offload(device="cpu")
    assert pipe._all_hooks and not getattr(raw, "_all_hooks", None)
    for name in ("text_encoder", "transformer", "vae"):
        assert not hasattr(getattr(raw, name), "_hf_hook"), name

    moves = []
    te_to = raw.text_encoder.to
    tr_to = raw.transformer.to
    raw.text_encoder.to = lambda *a, **k: moves.append(("te", str(a[0]))) or te_to(*a, **k)
    raw.transformer.to = lambda *a, **k: moves.append(("tr", str(a[0]))) or tr_to(*a, **k)
    try:
        out = pipe(prompt="GRAND OPENING", generator=_gen(3), **kw).images
        out2 = pipe(prompt="GRAND OPENING", generator=_gen(3), **kw).images
    finally:
        del raw.text_encoder.to, raw.transformer.to
    # Image 1: in (negative miss), out at the first DiT forward. Image 2: the negative is
    # cached, the positive still needs the encoder: in, out. Never the transformer.
    assert moves == [("te", "cpu"), ("te", "cpu"), ("te", "cpu"), ("te", "cpu")], moves
    assert not pipe._te_on_gpu
    assert np.array_equal(np.asarray(out), np.asarray(ref))
    assert np.array_equal(np.asarray(out2), np.asarray(ref))

    # With the positive handed as embeddings (cz_pipeline's cache) and the negative
    # cached, the encoder is not touched at all.
    emb = pipe.encode_prompt("GRAND OPENING")
    moves.clear()
    raw.text_encoder.to = lambda *a, **k: moves.append(("te", str(a[0]))) or te_to(*a, **k)
    try:
        pipe(prompt=None, prompt_embeds=emb[0], generator=_gen(3), **kw)
    finally:
        del raw.text_encoder.to
    assert moves == [], moves

    # diffusers' own model offload comes back on the raw pipe: replaced.
    raw.enable_model_cpu_offload(device="cpu")
    assert raw._all_hooks
    pipe(prompt="x", width=256, height=256, num_inference_steps=1, generator=_gen(0))
    assert not getattr(raw, "_all_hooks", None)
    assert not hasattr(raw.transformer, "_hf_hook")
    assert pipe._offload_device is not None

    pipe.maybe_free_model_hooks()
    pipe.remove_all_hooks()
    assert not pipe._all_hooks and pipe._offload_device is None and not pipe._tr_pre


def test_release_drops_the_weights():
    pipe = tiny_pipe()
    pipe.release()
    assert pipe.released
    with pytest.raises(RuntimeError, match="released"):
        pipe(prompt="x", width=256, height=256, num_inference_steps=1)
