"""Lot 4b: editing from references on MOSS-Image (the Moss-Image-Edit contract), CPU.

cz_moss.EditPrefix re-runs the vendored DiT with a reference prefix (references at t = 0
that see only each other, target and text that see everything, the output head on the
target slice only). What these tests lock down on the tiny model:
  - no reference = the vendored forward, bit for bit (skip layers included);
  - with references: the shape, determinism, the per-layer K/V cache equal to the
    recomputed prefix, references never written, the output changed by them;
  - the reference RoPE coordinates (r = i + 1, f = 0, rows and columns centred);
  - the adapter: a NAMED refusal without an edit LoRA, the source image's size kept,
    the references capped at the configured long side, at most four, Stop;
  - the app: generate_omni fetches the edit preset once, applies it for the edit, drops
    it again for text to image; the protocol announces edit / refs from the LoRA's
    presence; the scheduler config of the repo loads without the diffusers warning.
Run: python -m pytest tests/test_moss_edit.py -q
"""
import json
import logging
import os
import sys

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cz_moss  # noqa: E402
import cz_moss_lora  # noqa: E402
from moss_tiny import tiny_pipe, tiny_transformer  # noqa: E402


def _gen(seed=1):
    return torch.Generator().manual_seed(seed)


def edit_lora_file(path, seed=3):
    """A tiny moss-edit-lora file (the published metadata layout) on layers.0.proj.q."""
    g = torch.Generator().manual_seed(seed)
    cfg = {"format": "moss-edit-lora", "format_version": 1,
           "metadata": {"edit_contract": cz_moss.EDIT_CONTRACT},
           "modules": {"layers.0.proj.q": {"alpha": 8.0, "rank": 4, "in_features": 32}}}
    save_file({"layers.0.proj.q.lora_A": torch.randn(4, 32, generator=g) * 0.1,
               "layers.0.proj.q.lora_B": torch.randn(32, 4, generator=g) * 0.1},
              str(path), metadata={"adapter_config": json.dumps(cfg)})
    return str(path)


def _inputs(batch=2):
    torch.manual_seed(0)
    lat = torch.randn(batch, 128, 4, 6)
    emb = torch.randn(batch, 7, 16)
    mask = torch.ones(batch, 7, dtype=torch.bool)
    mask[-1, 5:] = False
    return lat, emb, mask


def test_without_references_the_prefix_forward_is_the_vendored_forward():
    tr = tiny_transformer()
    lat, emb, mask = _inputs()
    t = torch.tensor(500.0)
    with torch.no_grad():
        for skip in (None, [1]):
            base = tr(lat, t, emb, mask, skip_layers=skip).sample
            ours = cz_moss.EditPrefix(tr, [])(lat, t, emb, mask, skip_layers=skip).sample
            assert torch.equal(base, ours)


def test_references_change_the_output_and_the_cache_equals_the_recompute():
    tr = tiny_transformer()
    lat, emb, mask = _inputs()
    t = torch.tensor(500.0)
    refs = [torch.randn(1, 128, 3, 5), torch.randn(1, 128, 2, 2)]
    kept = [r.clone() for r in refs]
    with torch.no_grad():
        base = tr(lat, t, emb, mask).sample
        ep = cz_moss.EditPrefix(tr, refs)
        out = ep(lat, t, emb, mask).sample
        assert out.shape == base.shape and torch.isfinite(out).all()
        assert (out - base).abs().max() > 1e-3
        assert ep.kv is not None and len(ep.kv) == len(tr.layers)
        assert ep.tokens == 15 + 4 and ep.cache_bytes() == 2 * 2 * 19 * 32 * 2
        again = cz_moss.EditPrefix(tr, refs)(lat, t, emb, mask).sample
        assert torch.equal(out, again)
        ep.cache = False                                 # the prefix recomputed each call
        assert torch.allclose(out, ep(lat, t, emb, mask).sample, atol=1e-5)
        # the prefix is the references' alone: another t changes the body, not the kv
        kv0 = [(k.clone(), v.clone()) for k, v in ep.build(lat.device)]
        ep2 = cz_moss.EditPrefix(tr, refs)
        ep2(lat, torch.tensor(20.0), emb, mask)
        assert all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(kv0, ep2.kv))
    assert all(torch.equal(a, b) for a, b in zip(refs, kept))   # never written


def test_reference_rope_coordinates():
    ids = cz_moss.ref_rope_ids(1, 2, 3, torch.device("cpu"))
    assert ids.shape == (6, 4)
    assert (ids[:, 0] == 2).all() and (ids[:, 1] == 0).all()
    assert ids[:, 2].tolist() == [-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]
    assert ids[:, 3].tolist() == [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0]
    tr = tiny_transformer()
    table = cz_moss._rope_from_ids(tr.rope, ids)
    assert table.shape == (1, 6, 16)
    # the same layout as the vendored table: the target's image rows (r = 0, f = text
    # length) through the same function give the vendored forward's table
    base = tr.rope(2, 3, 4, torch.device("cpu"))
    h, w = torch.meshgrid(torch.arange(2), torch.arange(3), indexing="ij")
    img = torch.stack((torch.zeros_like(h), torch.full_like(h, 4), h, w), -1).flatten(0, 1).float()
    img[:, 2] -= 0.5
    img[:, 3] -= 1.0
    assert torch.allclose(cz_moss._rope_from_ids(tr.rope, img), base[:, :6])


def test_ref_size_caps_the_long_side_on_the_grid_and_never_enlarges():
    assert cz_moss.ref_size(100, 20, 64) == (64, 16)
    assert cz_moss.ref_size(2000, 1000, 1024) == (1024, 512)
    assert cz_moss.ref_size(300, 200, 1024) == (304, 192)      # grid only (12.5 rounds to 12)
    assert cz_moss.ref_size(5, 5, 1024) == (16, 16)


def test_the_adapter_refuses_an_edit_without_the_edit_lora_by_name():
    with pytest.raises(NotImplementedError, match="Moss-Image-Edit"):
        tiny_pipe()(prompt="x", num_inference_steps=1, image=Image.new("RGB", (32, 32)))


def test_the_adapter_edits_at_the_source_size_with_capped_references(tmp_path, monkeypatch):
    pipe = tiny_pipe()
    path = edit_lora_file(tmp_path / "edit.safetensors")
    assert cz_moss_lora.is_edit_lora(path) and cz_moss_lora.edit_contract(path) == cz_moss.EDIT_CONTRACT
    pipe.set_loras([(path, 1.0)])
    assert pipe.edit_loras_applied() == [path]
    monkeypatch.setenv("MOSS_EDIT_REF_MAX_EDGE", "64")
    seen = []
    real = pipe._encode_latents

    def spy(image, width, height, device):
        seen.append((width, height))
        return real(image, width, height, device)
    monkeypatch.setattr(cz_moss.MossPipe, "_encode_latents", lambda self, *a: spy(*a))
    calls = []
    real_call = pipe._pipe.__class__.__call__

    def spy_call(self, *a, **kw):
        calls.append(kw)
        return real_call(self, *a, **kw)
    monkeypatch.setattr(pipe._pipe.__class__, "__call__", spy_call)
    src = Image.new("RGB", (272, 256), (200, 30, 30))
    refs = [Image.new("RGB", (100, 20), (30, 30, 200)), Image.new("RGB", (40, 40)),
            Image.new("RGB", (40, 40)), Image.new("RGB", (40, 40)), Image.new("RGB", (40, 40))]
    out = pipe(prompt="make it blue", image=[src] + refs, num_inference_steps=2, generator=_gen())
    assert out.images[0].size == (272, 256)                 # the source's own size
    assert seen[0] == (64, 64) and seen[1] == (64, 16)      # capped at 64, on the grid
    assert len(seen) == cz_moss.EDIT_MAX_REFS                # six images -> four
    assert len(calls[-1]["reference_latents"]) == cz_moss.EDIT_MAX_REFS
    assert calls[-1]["reference_latents"][0].shape == (1, 128, 4, 4)
    out = pipe(prompt="make it blue", image=src, width=320, height=256,
               num_inference_steps=1, generator=_gen())
    assert out.images[0].size == (320, 256)                 # an explicit size wins
    # weight 0 = no edit LoRA in effect: refused again, by name
    pipe.set_loras([(path, 0.0)])
    with pytest.raises(NotImplementedError, match="Moss-Image-Edit"):
        pipe(prompt="x", num_inference_steps=1, image=src)


def test_stop_renders_the_partial_edit(tmp_path):
    pipe = tiny_pipe()
    pipe.set_loras([(edit_lora_file(tmp_path / "edit.safetensors"), 1.0)])
    steps = []

    def cb(p, i, t, kw):
        steps.append(i)
        if i == 0:
            p._interrupt = True
        return kw
    out = pipe(prompt="x", image=Image.new("RGB", (256, 256)), num_inference_steps=4,
               generator=_gen(), callback_on_step_end=cb)
    assert steps == [0] and out.images[0].size == (256, 256)


def test_generate_omni_applies_the_edit_preset_and_drops_it_for_txt2img(tmp_path, monkeypatch):
    import cz_edit_loras
    import cz_pipeline as P
    import test_moss_wiring as W
    path = edit_lora_file(tmp_path / "moss-image-edit.safetensors")
    fetched = []

    def resolve(name, download=True, progress=None):
        fetched.append(name)
        return path
    monkeypatch.setattr(cz_edit_loras, "resolve", resolve)
    monkeypatch.setenv("MOSS_EDIT_REF_MAX_EDGE", "64")
    src = Image.new("RGB", (256, 256), (10, 200, 10))
    said = []
    monkeypatch.setattr(P, "_log", lambda m: said.append(m))

    def go():
        pipe = P.get_pipe("txt2img")
        img = P.generate_omni([src], "make it red", "", 256, 256, 1, 7)
        assert img.size == (256, 256)
        assert fetched == [cz_edit_loras.EDIT_PRESET]
        assert P._APPLIED_EDIT_LORAS == [(path, 1.0)] and pipe.applied_loras == [(path, 1.0)]
        assert any("Moss-Image-Edit LoRA is applied" in m and "licence" in m for m in said), said
        P.generate_omni([src], "make it red", "", 256, 256, 1, 7)
        assert fetched == [cz_edit_loras.EDIT_PRESET] * 2   # resolved again (on disk: no download)
        # text to image afterwards: the base set (empty) comes back
        P.generate("a cat", 256, 256, 1, 7)
        assert pipe.applied_loras == [] and P._APPLIED_LORAS == []
        # a refused edit LoRA is a hard error, nothing generated
        bad = str(tmp_path / "flux3.safetensors")
        save_file({"transformer.mode_blocks.0.attn.to_q.lora_A.weight": torch.zeros(4, 3072),
                   "transformer.mode_blocks.0.attn.to_q.lora_B.weight": torch.zeros(3072, 4)}, bad)
        P.EDIT_LORAS = [(bad, 1.0)]
        try:
            with pytest.raises(RuntimeError, match="crispz-flux3"):
                P.generate_omni([src], "make it red", "", 256, 256, 1, 7)
        finally:
            P.EDIT_LORAS = []
    W._run(go)


def test_protocol_announces_edit_from_the_lora_presence(monkeypatch):
    import cz_edit_loras
    import cz_protocol
    assert cz_protocol.OMNI_SUPPORTED is True and cz_protocol.MAX_REFS == cz_moss.EDIT_MAX_REFS
    monkeypatch.setattr(cz_edit_loras, "edit_ready", lambda index=None: False)
    caps = cz_protocol.caps_dict()["supports"]
    assert caps["edit"] is False and caps["refs"] is False
    spec, warnings = cz_protocol.validate_spec({"protocol": 1, "prompt": "x", "refs": [__file__]})
    assert spec["refs"] == [] and any("Moss-Image-Edit" in w for w in warnings), warnings
    monkeypatch.setattr(cz_edit_loras, "edit_ready", lambda index=None: True)
    caps = cz_protocol.caps_dict()["supports"]
    assert caps["edit"] is True and caps["refs"] is True and caps["max_refs"] == 4
    assert any(e["name"] == cz_edit_loras.EDIT_PRESET and "licence" in e
               for e in cz_protocol.caps_dict()["edit_loras"])


def test_the_repo_scheduler_config_loads_without_the_diffusers_warning(tmp_path, caplog):
    d = tmp_path / "repo" / "scheduler"
    d.mkdir(parents=True)
    (d / "scheduler_config.json").write_text(json.dumps({
        "_class_name": "MOSSImageScheduler", "_diffusers_version": "0.38.0",
        "base_image_seq_len": 256, "base_shift": 0.5, "invert_sigmas": False,
        "max_image_seq_len": 4096, "max_shift": 1.15, "num_train_timesteps": 1000,
        "shift": 5.0, "shift_terminal": None, "stochastic_sampling": False,
        "time_shift_type": "exponential", "training_shift": 1.0, "use_beta_sigmas": False,
        "use_dynamic_shifting": False, "use_exponential_sigmas": False,
        "use_karras_sigmas": False}))
    with caplog.at_level(logging.WARNING):
        sch = cz_moss.load_scheduler(str(tmp_path / "repo"))
    assert sch.config.shift == 5.0 and sch.config.training_shift == 1.0
    assert "not expected and will be ignored" not in caplog.text
    sch.set_timesteps(4)
    assert len(sch.sigmas) == 5
