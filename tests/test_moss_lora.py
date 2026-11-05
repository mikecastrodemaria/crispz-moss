"""Lot 4: the MOSS-Image LoRA loader (cz_moss_lora) on the tiny DiT, CPU.

The four layouts seen on the published files (kohya, peft with ComfyUI's fused qkv,
moss-edit-lora's bare lora_A / lora_B with per-module alpha, ComfyUI diffs with norm
deltas), the exact arithmetic of a hooked Linear, keys that match nothing counted and
said, a control LoRA refused by name, unloading that restores everything, re-scaling
in place, and the app's path (cz_pipeline.LORAS -> MossPipe.set_loras).

Run: python -m pytest tests/test_moss_lora.py -q
"""
import json
import os
import sys

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cz_moss  # noqa: E402
import cz_moss_lora as L  # noqa: E402
from moss_tiny import tiny_pipe, tiny_transformer  # noqa: E402

DIM, HEAD, FFN = 32, 16, 64          # the tiny DiT (moss_tiny): proj.ffn_up_gate is 2 x FFN


def _rand(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g) * 0.1


def _save(tmp_path, name, sd, meta=None):
    p = str(tmp_path / name)
    save_file(sd, p, metadata=meta)
    return p


def test_classify_strips_prefixes_and_reads_every_layout():
    assert L.classify("diffusion_model.layers.0.proj.q.lora_down.weight") == ("down", "layers.0.proj.q")
    assert L.classify("diffusion_model.layers.0.proj.q.lora_up.weight") == ("up", "layers.0.proj.q")
    assert L.classify("diffusion_model.layers.0.proj.q.alpha") == ("alpha", "layers.0.proj.q")
    assert L.classify("layers.3.proj.ffn_down.lora_A.weight") == ("down", "layers.3.proj.ffn_down")
    assert L.classify("layers.3.proj.ffn_down.lora_B") == ("up", "layers.3.proj.ffn_down")
    assert L.classify("transformer.layers.3.proj.o.lora.up.weight") == ("up", "layers.3.proj.o")
    assert L.classify("diffusion_model.layers.37.proj.qkv.diff") == ("diff", "layers.37.proj.qkv")
    assert L.classify("diffusion_model.layers.37.norm1.diff") == ("diff", "layers.37.norm1")
    assert L.classify("diffusion_model.layers.37.proj.o.diff_b") == ("diff_b", "layers.37.proj.o")
    assert L.classify("video_embedder.weight") == ("full", "video_embedder")
    assert L.classify("something.odd") == ("unknown", "something.odd")


def test_kohya_file_scale_is_alpha_over_rank(tmp_path):
    down, up = _rand(4, DIM, seed=1), _rand(DIM, 4, seed=2)
    p = _save(tmp_path, "kohya.safetensors", {
        "diffusion_model.layers.0.proj.q.lora_down.weight": down,
        "diffusion_model.layers.0.proj.q.lora_up.weight": up,
        "diffusion_model.layers.0.proj.q.alpha": torch.tensor(2.0)})
    tr = tiny_transformer()
    x = _rand(1, 5, DIM, seed=3)
    base = tr.layers[0].proj["q"](x).clone()
    reps = L.apply(tr, [(p, 0.5)])
    assert reps[0]["modules"] == 1 and reps[0]["unknown"] == [] and reps[0]["missing"] == []
    got = tr.layers[0].proj["q"](x)
    want = base + (x @ down.t() @ up.t()) * (2.0 / 4) * 0.5
    assert torch.allclose(got, want, atol=1e-5)
    assert L.applied_modules(tr) == 1
    assert L.unload(tr) == 1 and L.applied_modules(tr) == 0
    assert torch.allclose(tr.layers[0].proj["q"](x), base)


def test_peft_fused_qkv_is_split_and_unknown_keys_are_said(tmp_path, monkeypatch):
    down, up = _rand(4, DIM, seed=4), _rand(3 * DIM, 4, seed=5)
    tdown, tup = _rand(4, DIM, seed=6), _rand(3 * DIM, 4, seed=7)
    p = _save(tmp_path, "comfy.safetensors", {
        "layers.0.proj.qkv.lora_A.weight": down, "layers.0.proj.qkv.lora_B.weight": up,
        "text_embedder.blocks.0.qkv.lora_A.weight": tdown,
        "text_embedder.blocks.0.qkv.lora_B.weight": tup,
        "layers.0.proj.nope.lora_A.weight": _rand(4, DIM), "layers.0.proj.nope.lora_B.weight": _rand(DIM, 4),
        "stray.tensor": torch.zeros(2)}, meta={"alpha": "4.0"})
    lf = L.read_file(p)
    assert lf.split == 1 and set(lf.items) == {"layers.0.proj.q", "layers.0.proj.k", "layers.0.proj.v",
                                                "text_embedder.blocks.0.qkv", "layers.0.proj.nope"}
    assert torch.equal(lf.items["layers.0.proj.k"]["up"], up[DIM:2 * DIM])
    assert lf.items["layers.0.proj.k"]["alpha"] == 4.0 and lf.unknown == ["stray.tensor"]
    said = []
    monkeypatch.setattr(L, "_log", lambda m: said.append(m))
    tr = tiny_transformer()
    x = _rand(1, 3, DIM, seed=8)
    base_v = tr.layers[0].proj["v"](x).clone()
    base_t = tr.text_embedder.blocks[0].qkv(x).clone()
    reps = L.apply(tr, [(p, 1.0)])
    assert reps[0]["modules"] == 4 and reps[0]["split"] == 1
    assert reps[0]["unknown"] == ["stray.tensor"] and reps[0]["missing"] == ["layers.0.proj.nope"]
    assert any("NOT applied: 1 unknown key(s), 1 module(s) not in this DiT" in m for m in said), said
    assert torch.allclose(tr.layers[0].proj["v"](x), base_v + (x @ down.t() @ up[2 * DIM:].t()), atol=1e-5)
    assert torch.allclose(tr.text_embedder.blocks[0].qkv(x), base_t + (x @ tdown.t() @ tup.t()), atol=1e-5)


def test_moss_edit_lora_bare_keys_take_their_per_module_alpha(tmp_path):
    down, up = _rand(4, FFN, seed=9), _rand(DIM, 4, seed=10)
    cfg = {"format": "moss-edit-lora", "modules": {
        "layers.1.proj.ffn_down": {"alpha": 8.0, "rank": 4, "in_features": FFN}}}
    p = _save(tmp_path, "edit.safetensors", {
        "layers.1.proj.ffn_down.lora_A": down, "layers.1.proj.ffn_down.lora_B": up},
        meta={"adapter_config": json.dumps(cfg)})
    lf = L.read_file(p)
    assert lf.items["layers.1.proj.ffn_down"]["alpha"] == 8.0
    tr = tiny_transformer()
    x = _rand(1, 2, FFN, seed=11)
    base = tr.layers[1].proj["ffn_down"](x).clone()
    L.apply(tr, [(p, 1.0)])
    assert torch.allclose(tr.layers[1].proj["ffn_down"](x), base + (x @ down.t() @ up.t()) * 2.0, atol=1e-5)


def test_comfy_diffs_hit_linears_norms_and_biases_and_unload_restores(tmp_path):
    qkv = _rand(3 * DIM, DIM, seed=12)
    n1, qn, ob = _rand(DIM, seed=13), _rand(HEAD, seed=14), _rand(DIM, seed=15)
    p = _save(tmp_path, "deband.safetensors", {
        "diffusion_model.layers.0.proj.qkv.diff": qkv,
        "diffusion_model.layers.0.norm1.diff": n1,
        "diffusion_model.layers.0.q_norm.diff": qn,
        "diffusion_model.layers.0.proj.o.diff_b": ob})
    tr = tiny_transformer()
    x = _rand(1, 2, DIM, seed=16)
    base_q = tr.layers[0].proj["q"](x).clone()
    base_o = tr.layers[0].proj["o"](x).clone()
    norm1_before = tr.layers[0].norm1.weight.detach().clone()
    reps = L.apply(tr, [(p, 0.5)])
    assert reps[0]["split"] == 1 and reps[0]["diffs"] == 3 and reps[0]["norms"] == 2
    assert torch.allclose(tr.layers[0].proj["q"](x), base_q + (x @ qkv[:DIM].t()) * 0.5, atol=1e-5)
    assert torch.allclose(tr.layers[0].proj["o"](x), base_o + ob * 0.5, atol=1e-5)
    assert torch.allclose(tr.layers[0].norm1.weight, norm1_before + n1 * 0.5)
    assert L.rescale(tr, [1.0]) is False                        # norm diffs: re-apply instead
    L.unload(tr)
    assert torch.allclose(tr.layers[0].norm1.weight, norm1_before)
    assert torch.allclose(tr.layers[0].proj["q"](x), base_q)
    assert not any(b.startswith("_cz_lora_") for m in tr.modules() for b in m._buffers)


def test_control_lora_and_empty_file_are_refused_by_name(tmp_path):
    p = _save(tmp_path, "control.safetensors", {
        "layers.0.proj.q.lora_A.weight": _rand(4, DIM), "layers.0.proj.q.lora_B.weight": _rand(DIM, 4),
        "video_embedder.weight": _rand(DIM, 384), "video_embedder.bias": _rand(DIM)})
    why = L.refusal(p)
    assert "CONTROL" in why and "video_embedder" in why
    with pytest.raises(ValueError, match="CONTROL"):
        L.apply(tiny_transformer(), [(p, 1.0)])
    p2 = _save(tmp_path, "empty.safetensors", {"layers.0.proj.q.alpha": torch.tensor(1.0)})
    assert "no LoRA tensors" in L.refusal(p2)
    p3 = _save(tmp_path, "ok.safetensors", {
        "layers.0.proj.q.lora_A.weight": _rand(4, DIM), "layers.0.proj.q.lora_B.weight": _rand(DIM, 4)})
    assert L.refusal(p3) is None


def test_rescale_in_place_and_weight_zero_is_identity(tmp_path):
    down, up = _rand(4, DIM, seed=17), _rand(DIM, 4, seed=18)
    p = _save(tmp_path, "k.safetensors", {
        "layers.0.proj.q.lora_A.weight": down, "layers.0.proj.q.lora_B.weight": up})
    tr = tiny_transformer()
    x = _rand(1, 2, DIM, seed=19)
    base = tr.layers[0].proj["q"](x).clone()
    L.apply(tr, [(p, 1.0)])
    one = tr.layers[0].proj["q"](x).clone()
    assert L.rescale(tr, [0.0]) is True
    assert torch.allclose(tr.layers[0].proj["q"](x), base, atol=1e-6)
    L.rescale(tr, [2.0])
    assert torch.allclose(tr.layers[0].proj["q"](x), base + 2 * (one - base), atol=1e-5)


def test_the_pipe_applies_them_and_the_render_changes(tmp_path):
    """MossPipe.set_loras: a random LoRA changes the image, the same file at another
    weight is rescaled without reloading, an empty set gives the original image back."""
    import numpy as np
    pipe = tiny_pipe()
    kw = dict(prompt="OPEN", width=256, height=256, num_inference_steps=3, output_type="np",
              guidance_scale=1.0)
    g = lambda: torch.Generator().manual_seed(5)  # noqa: E731
    ref = np.asarray(pipe(generator=g(), **kw).images)
    sd = {}
    for i in range(2):
        for m in ("q", "k", "v", "o"):
            sd[f"diffusion_model.layers.{i}.proj.{m}.lora_down.weight"] = _rand(4, DIM, seed=20 + i)
            sd[f"diffusion_model.layers.{i}.proj.{m}.lora_up.weight"] = _rand(DIM, 4, seed=30 + i) * 5
    p = _save(tmp_path, "style.safetensors", sd)
    reps = pipe.set_loras([(p, 1.0)])
    assert reps[0]["modules"] == 8 and pipe.applied_loras == [(p, 1.0)]
    out = np.asarray(pipe(generator=g(), **kw).images)
    assert np.abs(out.astype(float) - ref.astype(float)).mean() > 1e-3
    assert pipe.set_loras([(p, 0.3)]) == [] and pipe.applied_loras == [(p, 0.3)]   # rescaled
    pipe.set_loras([])
    assert pipe.applied_loras == []
    assert np.array_equal(np.asarray(pipe(generator=g(), **kw).images), ref)
    with pytest.raises(NotImplementedError, match="set_loras"):
        pipe.load_lora_weights(p)


def test_cz_pipeline_routes_loras_to_the_adapter(tmp_path, monkeypatch):
    """cz_pipeline.LORAS -> _apply_loras -> MossPipe.set_loras; a refused file is skipped
    with its reason, a missing one ignored, and _clear_loras empties the set."""
    import cz_pipeline as P
    pipe = tiny_pipe()
    good = _save(tmp_path, "good.safetensors", {
        "layers.0.proj.q.lora_A.weight": _rand(4, DIM), "layers.0.proj.q.lora_B.weight": _rand(DIM, 4)})
    bad = _save(tmp_path, "control.safetensors", {
        "layers.0.proj.q.lora_A.weight": _rand(4, DIM), "layers.0.proj.q.lora_B.weight": _rand(DIM, 4),
        "video_embedder.weight": _rand(DIM, 384)})
    said = []
    monkeypatch.setattr(P, "_log", lambda m: said.append(m))
    monkeypatch.setattr(P, "LORAS", [(good, 0.7), (bad, 1.0), (str(tmp_path / "none.safetensors"), 1.0)])
    monkeypatch.setattr(P, "_APPLIED_LORAS", [])
    assert P._apply_loras(pipe) is True
    assert pipe.applied_loras == [(good, 0.7)] and P._APPLIED_LORAS == [(good, 0.7)]
    assert any("SKIPPED" in m and "CONTROL" in m for m in said), said
    assert any("not found" in m for m in said)
    assert P._apply_loras(pipe) is True                           # same set: nothing redone
    P._clear_loras(pipe)
    assert pipe.applied_loras == []
