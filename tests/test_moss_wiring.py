"""cz_pipeline / cz_protocol / cz_core on the MOSS engine (lot 1, CPU).

A tiny MossPipe is put in place of the loaded model and the real entry point runs:
generate(). Also: get_pipe hands the one object back, the guidance slider and the
negative prompt reach the adapter, the sampler is left alone, the live preview reads the
(B, 128, h, w) latents, the protocol announces what lot 1 can and cannot do, the profile
gives 50 steps / 3.0, a FLUX.3 or Ideogram LoRA is refused with the app it belongs to, a
MOSS LoRA is refused by name until lot 4, and the checkpoint menu lists a ComfyUI /
CivitAI single file (bf16 and int8) and refuses the 4-bit one.
Run: python -m pytest tests/test_moss_wiring.py -q
"""
import json
import os
import struct
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cz_core  # noqa: E402
import cz_moss  # noqa: E402
import cz_pipeline as P  # noqa: E402
import cz_protocol  # noqa: E402
from moss_tiny import tiny_pipe, tiny_transformer, save_tiny_comfy_file  # noqa: E402

CALLS = []
_TINY = None


def _install():
    global _TINY
    if _TINY is None:
        _TINY = tiny_pipe()
    pipe = _TINY
    real = cz_moss.MossPipe.__call__

    def spy(self, *a, **kw):
        CALLS.append(dict(kw))
        return real(self, *a, **kw)
    cz_moss.MossPipe.__call__ = spy
    P._BASE_PIPE = pipe
    P._DERIVED = {"txt2img": pipe}
    P._LOADED_KEY = (P.BASE_REPO, P.ZIMAGE_TRANSFORMER, P.OFFLOAD_MODE, P.PRECISION)
    P._BASE_SCHED_CONFIG = dict(pipe.scheduler.config)
    P.LORAS = []
    P.EDIT_LORAS = []
    return real


def _run(fn):
    CALLS.clear()
    real = _install()
    try:
        return fn()
    finally:
        cz_moss.MossPipe.__call__ = real
        P._BASE_PIPE, P._DERIVED, P._LOADED_KEY = None, {}, None


def test_defaults_point_at_moss():
    assert P.BASE_REPO == cz_moss.DEFAULT_REPO
    assert P.PRECISION == "bf16"
    assert P.CFG_MODEL_KEY == "moss_model"
    assert P.GUIDANCE == 3.0


def test_generate_reaches_the_adapter_with_guidance_and_negative():
    old = P.GUIDANCE
    P.GUIDANCE = 4.5
    try:
        img = _run(lambda: P.generate("a sign that says OPEN", 256, 512, 3, 1,
                                      negative_prompt="blurry"))
    finally:
        P.GUIDANCE = old
    assert img.size == (256, 512)
    assert CALLS[-1]["guidance_scale"] == 4.5
    assert CALLS[-1]["negative_prompt"] == "blurry"


def test_get_pipe_is_one_object():
    def go():
        base = P.get_pipe("txt2img")
        assert P.get_pipe("img2img") is base and P.get_pipe("inpaint") is base
    _run(go)


def test_negative_prompt_is_passed():
    assert P._cfg("blurry") == {"negative_prompt": "blurry"}
    assert P._cfg("") == {"negative_prompt": ""}


def test_sampler_left_alone():
    def go():
        pipe = P.get_pipe("txt2img")
        sched = pipe.scheduler
        old = (P.SAMPLER, P.SCHEDULE)
        P.SAMPLER, P.SCHEDULE = "unipc", "karras"
        try:
            P._apply_sampler(pipe)
        finally:
            P.SAMPLER, P.SCHEDULE = old
        assert pipe.scheduler is sched
    _run(go)


def test_live_preview_reads_the_4d_latents():
    """The app's real step-end callback on the tiny pipe: a frame comes out of the
    (B, 128, h, w) latents through the FLUX.2 matrix as is (MOSS packs its tokens in the
    FLUX order and normalizes them with the matching bn statistics)."""
    was = P.LIVE_PREVIEW_ENABLED
    P.LIVE_PREVIEW_ENABLED = True
    P.preview_begin()
    try:
        kw = P._step_end_kwargs(3, 512, 256)
        _run(lambda: _TINY(prompt="GRAND OPENING", width=512, height=256,
                           num_inference_steps=3, generator=torch.Generator().manual_seed(0),
                           **kw))
        snap = P.preview_snapshot()
        assert snap["img"] is not None and snap["step"] == 3, snap
        assert snap["img"].width > snap["img"].height
    finally:
        P.preview_end()
        P.LIVE_PREVIEW_ENABLED = was
    # The grid is checked against the asked resolution: a mismatch gives no preview.
    assert P.latent_preview_image(torch.zeros(1, 128, 16, 32), 1024, 1024) is None
    assert P.latent_preview_image(torch.zeros(1, 128, 16, 32), 512, 256) is not None
    assert P._preview_proj_for(_TINY) is None


def test_protocol_caps():
    caps = cz_protocol.caps_dict()["supports"]
    assert caps["negative"] is True
    assert caps["img2img"] is True and caps["inpaint"] is True
    assert caps["emulated"] == ["img2img", "inpaint"]
    assert caps["edit"] is False and caps["refs"] is False
    assert caps["cloud"]["enabled"] is False and caps["cloud"]["ops"] == []
    assert cz_protocol.TOOL == "crispz-moss"


def test_profile():
    assert cz_core.profile_for_model("OpenMOSS-Team/MOSS-Image-SFT") == (50, 3.0)
    assert cz_core.profile_for_model("mossImage_sft_3301904.safetensors") == (50, 3.0)
    assert cz_core.profile_for_model("dirtyMoss_3317613.safetensors") == (50, 3.0)


def _fake_safetensors(names_shapes):
    hdr, off = {}, 0
    for n, shape in names_shapes:
        size = 2
        for d in shape:
            size *= d
        hdr[n] = {"dtype": "BF16", "shape": shape, "data_offsets": [off, off + size]}
        off += size
    raw = json.dumps(hdr).encode()
    f = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
    f.write(struct.pack("<Q", len(raw)) + raw + b"\0" * off)
    f.close()
    return f.name


def test_foreign_loras_are_refused_and_named():
    path = _fake_safetensors([
        ("transformer.mode_blocks.image.0.attn.to_q.lora_A.weight", [4, 8]),
        ("transformer.mode_blocks.image.0.attn.to_q.lora_B.weight", [8, 4])])
    try:
        why = P._lora_unsupported(path)
    finally:
        os.unlink(path)
    assert why and "crispz-flux3" in why
    path = _fake_safetensors([
        ("transformer.layers.0.attention.to_q.lora_A.weight", [4, 4608]),
        ("transformer.layers.0.attention.to_q.lora_B.weight", [4608, 4])])
    try:
        why = P._lora_unsupported(path)
    finally:
        os.unlink(path)
    assert why and "crispz-ideogram" in why


def test_a_moss_lora_is_refused_until_lot_4():
    path = _fake_safetensors([
        ("layers.0.proj.q.lora_A.weight", [64, 3072]),
        ("layers.0.proj.q.lora_B.weight", [3072, 64])])
    try:
        why = P._lora_unsupported(path)
    finally:
        os.unlink(path)
    assert why and "lot 4" in why


def test_checkpoint_menu_lists_comfy_files_and_refuses_4bit():
    tr = tiny_transformer(seed=5)
    with tempfile.TemporaryDirectory() as d:
        bf16 = save_tiny_comfy_file(os.path.join(d, "mossImage_sft_bf16.safetensors"), tr)
        int8 = save_tiny_comfy_file(os.path.join(d, "dirtyMoss_int8.safetensors"), tr, int8=True)
        from safetensors.torch import save_file
        blob = json.dumps({"format": "asym_w4a8_int8", "group_size": 16}).encode()
        save_file({"layers.0.proj.qkv.weight": torch.zeros(96, 16, dtype=torch.int8),
                   "layers.0.proj.qkv.comfy_quant": torch.frombuffer(bytearray(blob),
                                                                     dtype=torch.uint8).clone(),
                   "video_embedder.weight": torch.zeros(32, 128, dtype=torch.bfloat16)},
                  os.path.join(d, "mossImage_int4.safetensors"))
        ideo = _fake_safetensors([("llm_cond_proj.weight", [4, 4])])
        os.replace(ideo, os.path.join(d, "ideogram.safetensors"))
        old = (P.CHECKPOINTS_DIR, P.CHECKPOINTS_EXTRA_DIR)
        P.CHECKPOINTS_DIR, P.CHECKPOINTS_EXTRA_DIR = d, ""
        P._FAMILY_CACHE.clear()
        try:
            listed = P.list_checkpoints()
            assert sorted(listed) == ["dirtyMoss_int8.safetensors", "mossImage_sft_bf16.safetensors"], listed
            assert "w4a8" in P.checkpoint_refusal("mossImage_int4.safetensors")
            assert "crispz-ideogram" in P.checkpoint_refusal("ideogram.safetensors")
            assert P.checkpoint_refusal("mossImage_sft_bf16.safetensors") is None
            # The single file goes through the engine's reader: the same weights.
            P.ZIMAGE_TRANSFORMER = int8
            try:
                loaded = P._load_transformer()
            finally:
                P.ZIMAGE_TRANSFORMER = None
            assert len(loaded.layers) == len(tr.layers)
            assert torch.equal(loaded.state_dict()["video_embedder.weight"].float(),
                               tr.state_dict()["video_embedder.weight"].to(torch.bfloat16).float())
        finally:
            P.CHECKPOINTS_DIR, P.CHECKPOINTS_EXTRA_DIR = old
            P._FAMILY_CACHE.clear()


def test_vram_estimate_follows_the_encoder_precision():
    old = P.PRECISION
    try:
        P.PRECISION = "bf16"
        a = P._base_vram_need_gb()
        P.PRECISION = "nf4"
        b = P._base_vram_need_gb()
    finally:
        P.PRECISION = old
    assert 24 < a < 27 and 14 < b < 17, (a, b)
    assert P._moss_auto("none", "fits", 31.8) == ("model", P._moss_auto("none", "fits", 31.8)[1])
    assert P._moss_auto("none", "fits", 48.0)[0] == "none"
    assert P._moss_auto("sequential", "x", 31.8)[0] == "sequential"


# --- lot 3: the app's img2img / inpaint / outpaint paths on the emulated MossPipe --------
def _img(w=256, h=256):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (w, h), (40, 90, 200))
    ImageDraw.Draw(im).ellipse([w // 4, h // 4, 3 * w // 4, 3 * h // 4], fill=(240, 200, 20))
    return im


def test_inpaint_run_goes_through_the_emulated_inpaint():
    from PIL import Image, ImageDraw
    mask = Image.new("L", (256, 256), 0)
    ImageDraw.Draw(mask).rectangle([128, 0, 255, 255], fill=255)
    out = _run(lambda: P.inpaint_run(_img(), mask, "a red door", 4, 0.8, 3))
    assert out.size == (256, 256)
    kw = CALLS[-1]
    assert kw["mask_image"] is not None and kw["strength"] == 0.8


def test_refine_whole_is_an_img2img():
    def go():
        return P._refine_whole(P.get_pipe("img2img"), _img(320, 256), 0.35, 8, "x", 1)
    out = _run(go)
    assert out.size == (320, 256)
    kw = CALLS[-1]
    assert kw["strength"] == 0.35 and kw.get("mask_image") is None


def test_reframe_and_directional_outpaint_run():
    out = _run(lambda: P.reframe(_img(256, 256), 16, 9, "contain", "x", 3, 1))
    assert out.width > out.height
    assert CALLS[-1]["mask_image"] is not None
    out2 = _run(lambda: P.outpaint_directions(_img(256, 256), None, ["right"], "x", 3, 1))
    assert out2.width > 256


def test_a_layout_draft_follows_the_image_size_in_img2img():
    """A draft drawn for another canvas does not force its size on an img2img: it is
    rescaled to the image's (said), and the output is the image's size."""
    import cz_design
    draft = cz_design.prompt_of("Logo (square)")          # drawn for 1024x1024
    said = []
    real = P._log                       # the note is logged by cz_pipeline (_qwen_call)
    P._log = lambda m: said.append(m)
    try:
        out = _run(lambda: P._refine_whole(P.get_pipe("img2img"), _img(320, 256), 0.5, 4,
                                           draft, 1))
    finally:
        P._log = real
    assert out.size == (320, 256)
    assert any("rescaled to the image's 320x256" in m for m in said), said


def test_outpaint_and_reframe_place_the_draft_and_guard_the_new_areas():
    """The outpaint and reframe paths hand the source rectangle to the engine: the
    compiled prompt the adapter receives keeps the draft's boxes inside it and says
    "no other text"; a plain prompt gets the guard too."""
    import cz_caption
    import cz_design
    draft = cz_design.prompt_of("Logo (square)")
    # The compiled text is turned into embeddings by cz_pipeline's cache before the
    # adapter sees it: the compilation itself is observed.
    made = []
    real = cz_moss.prepare_prompt

    def spy(prompt, width=None, height=None, fit=False, place=None):
        out = real(prompt, width, height, fit=fit, place=place)
        made.append((place, out))
        return out
    cz_moss.prepare_prompt = spy
    try:
        _run(lambda: P.outpaint_directions(_img(256, 256), None, ["right"], draft, 3, 1))
        place, (text, w, h, note) = made[-1]
        assert place is not None and place[0] == 0 and place[2] < w     # the source, left part
        d = cz_caption.parse(text)
        assert cz_caption.is_compiled(d) and "no other text" in d["summary"]["caption"].lower()
        assert w > h and CALLS[-1]["width"] == w                        # the canvas grew to the right
        lay, _, _ = cz_caption.from_text(text, w, h)
        for it in lay["items"]:
            x0, y0, x1, y1 = cz_caption.bbox_to_pixels(it["bbox"], w, h)
            assert x1 <= place[2] + 3, (it, x1, place)
        _run(lambda: P.reframe(_img(256, 256), 16, 9, "contain", "a plain poster", 3, 1))
        place, (text, _, _, _) = made[-1]
        assert place is not None and text.endswith("No other text.")
    finally:
        cz_moss.prepare_prompt = real

