"""Edit-LoRA presets on crispz-moss (lot 4b).

The catalogue ships ONE preset, "Moss-Image-Edit" (the chinoll/Moss-Image-Edit weights
as repacked by Comfy-Org; their author states no licence, the preset says so). What
these tests lock down:
  - the shipped preset: its example instruction, weight 1.0, four inputs, the licence
    note, a Hugging Face source that lands under the stable name, found again under the
    repack's own file name;
  - edit_ready() (what the protocol announces as supports.edit) follows the file;
  - a preset declared in config.txt ('edit_loras') is picked up, by name and adapter name;
  - a FLUX.2 / FLUX.3 edit LoRA is refused by the family guard with one sentence, the
    MOSS edit LoRA passes it and is recognised as an edit LoRA;
  - picking a preset in the UI applies it at ITS advised weight and moves the slider.
"""
import json
import os
import sys
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import cz_edit_loras  # noqa: E402

SPEC = {"repo": "someone/moss-relight-lora", "weights": "moss_relight.safetensors",
        "adapter_name": "relight-moss", "prompt": "Relight the scene with a warm sunset.",
        "inputs": 1, "base": "moss", "weight": 0.7}


class _Declared:
    """A preset declared through config 'edit_loras', the way a user would."""

    def __enter__(self):
        self.old = cz_edit_loras.SPECS
        cz_edit_loras.SPECS = cz_edit_loras._apply_config_overrides({})
        cz_edit_loras.CONFIG["edit_loras"] = {"Relight Moss": dict(SPEC)}
        cz_edit_loras.SPECS = cz_edit_loras._apply_config_overrides({})
        return self

    def __exit__(self, *exc):
        cz_edit_loras.CONFIG.pop("edit_loras", None)
        cz_edit_loras.SPECS = self.old
        return False


def test_the_shipped_catalogue_is_the_moss_image_edit_lora():
    assert list(cz_edit_loras.EDIT_LORA_SPECS) == [cz_edit_loras.EDIT_PRESET]
    s = cz_edit_loras.spec("Moss-Image-Edit")
    assert s["repo"] == "Comfy-Org/MOSS-Image" and s["weights"].endswith("moss_image_edit_lora_rank_64_bf16.safetensors")
    assert s["prompt"] == cz_edit_loras.EDIT_PROMPT and "preserving" in s["prompt"]
    assert float(s["weight"]) == 1.0 and int(s["inputs"]) == 4
    assert "no licence" in s["licence"] and "GPLv3" in s["licence"]
    assert os.path.basename(s["weights"]) in s["local_names"]   # the repack's file name
    assert cz_edit_loras.local_path("Moss-Image-Edit").endswith("moss-image-edit.safetensors")
    assert cz_edit_loras.spec("moss-image-edit") is s          # by adapter name
    assert cz_edit_loras.spec(cz_edit_loras.OUTPAINT_PRESET) is None   # flux3's, not here
    assert cz_edit_loras.speed_names() == []                   # no Lightning LoRA either
    cat = cz_edit_loras.catalog()
    assert cat[0]["name"] == "Moss-Image-Edit" and cat[0]["licence"] == s["licence"]


def test_the_preset_downloads_from_the_hub_under_the_stable_name():
    """The Hugging Face source: hf_hub_download gets repo + file, the copy lands as
    <adapter_name>.safetensors so local_path() and edit_ready() find it next time; a
    failed download is a clean error, no half file."""
    import huggingface_hub
    tmp = tempfile.mkdtemp()
    seen = {}
    src = os.path.join(tmp, "hub_copy.safetensors")
    open(src, "wb").write(b"x" * 10)

    def fake_download(repo, filename, **kw):
        seen["args"] = (repo, filename)
        return src

    real = (huggingface_hub.hf_hub_download, cz_edit_loras.lora_dirs, cz_edit_loras.edit_loras_dir)
    huggingface_hub.hf_hub_download = fake_download
    cz_edit_loras.lora_dirs = lambda: [os.path.join(tmp, "loras")]
    cz_edit_loras.edit_loras_dir = lambda: os.path.join(tmp, "loras", "_hf-edit")
    try:
        assert not cz_edit_loras.is_downloaded("Moss-Image-Edit") and not cz_edit_loras.edit_ready()
        got = cz_edit_loras.resolve("Moss-Image-Edit")
        assert os.path.samefile(got, cz_edit_loras.local_path("Moss-Image-Edit"))
        assert seen["args"] == ("Comfy-Org/MOSS-Image", "loras/moss_image_edit_lora_rank_64_bf16.safetensors")
        assert cz_edit_loras.is_downloaded("Moss-Image-Edit") and cz_edit_loras.edit_ready()
        assert cz_edit_loras.resolve("Moss-Image-Edit") == got      # no second download
        os.remove(got)

        def boom(*a, **k):
            raise OSError("401 gated")
        huggingface_hub.hf_hub_download = boom
        try:
            cz_edit_loras.resolve("Moss-Image-Edit")
        except RuntimeError as e:
            assert "401" in str(e)
        else:
            raise AssertionError("a failed download must raise")
        assert not os.path.exists(cz_edit_loras.local_path("Moss-Image-Edit") + ".tmp")
    finally:
        (huggingface_hub.hf_hub_download, cz_edit_loras.lora_dirs,
         cz_edit_loras.edit_loras_dir) = real
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_config_declared_preset_is_picked_up():
    with _Declared():
        assert "Relight Moss" in cz_edit_loras.names()
        s = cz_edit_loras.spec("Relight Moss")
        assert s["repo"] == SPEC["repo"] and float(s["weight"]) == 0.7
        assert cz_edit_loras.spec("relight-moss") is s    # by adapter_name too
        assert cz_edit_loras.spec("nope") is None
        assert cz_edit_loras.catalog()                  # it is announced to the protocol


def test_a_local_copy_is_used_without_downloading():
    tmp = tempfile.mkdtemp()
    main, extra = os.path.join(tmp, "loras"), os.path.join(tmp, "civitai")
    os.makedirs(main)
    os.makedirs(extra)
    copy = os.path.join(extra, SPEC["weights"])
    open(copy, "wb").close()
    real = (cz_edit_loras.lora_dirs, cz_edit_loras.edit_loras_dir, cz_edit_loras._download)

    def no_download(*a, **kw):
        raise AssertionError("a local copy exists: nothing must be downloaded")

    cz_edit_loras.lora_dirs = lambda: [main, extra]
    cz_edit_loras.edit_loras_dir = lambda: os.path.join(main, "_hf-edit")
    cz_edit_loras._download = no_download
    try:
        with _Declared():
            assert cz_edit_loras.is_downloaded("Relight Moss")
            assert os.path.samefile(cz_edit_loras.resolve("Relight Moss"), copy)
            assert cz_edit_loras.status_label("Relight Moss").endswith("✓")
    finally:
        cz_edit_loras.lora_dirs, cz_edit_loras.edit_loras_dir, cz_edit_loras._download = real
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_flux2_edit_lora_is_refused_by_family():
    import cz_pipeline
    tmp = tempfile.mkdtemp()
    try:
        l2 = os.path.join(tmp, "consistence_edit_flux2.safetensors")
        save_file({"diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight": torch.zeros(4, 3072),
                   "diffusion_model.double_blocks.0.img_attn.proj.lora_B.weight": torch.zeros(3072, 4)}, l2)
        why = cz_pipeline._lora_unsupported(l2)
        assert why and "FLUX.2" in why and "crispz-klein" in why, why
        # a FLUX.3 edit LoRA (the relight / outpaint presets of crispz-flux3) is the other
        # family on Ideogram 4 too, and names its studio
        l3 = os.path.join(tmp, "relight_flux3.safetensors")
        save_file({"transformer.mode_blocks.0.attn.to_q.lora_A.weight": torch.zeros(4, 3072),
                   "transformer.mode_blocks.0.attn.to_q.lora_B.weight": torch.zeros(3072, 4)}, l3)
        why = cz_pipeline._lora_unsupported(l3)
        assert why and "FLUX.3" in why and "crispz-flux3" in why, why
        # the MOSS edit LoRA (bare lora_A / lora_B, moss-edit-lora metadata) passes the
        # guard and is recognised as an edit LoRA (the reference prefix needs one)
        import cz_moss_lora
        lm = os.path.join(tmp, "moss_image_edit.safetensors")
        cfg = {"format": "moss-edit-lora", "format_version": 1,
               "metadata": {"edit_contract": "ref-prefix-only-t0__target-text-original-moss-v1"},
               "modules": {"layers.0.proj.q": {"alpha": 64, "rank": 64}}}
        save_file({"layers.0.proj.q.lora_A": torch.zeros(64, 3072),
                   "layers.0.proj.q.lora_B": torch.zeros(3072, 64),
                   "video_embedder.lora_A": torch.zeros(64, 128)},
                  lm, metadata={"adapter_config": json.dumps(cfg)})
        assert cz_pipeline._lora_unsupported(lm) is None
        assert cz_moss_lora.is_edit_lora(lm) and not cz_moss_lora.is_edit_lora(l3)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_picking_a_preset_sets_its_weight_on_the_slider():
    import cz_ui
    import cz_pipeline
    applied = []
    real = (cz_pipeline.set_edit_loras, cz_edit_loras.is_downloaded, cz_ui._edit_lora_choices)
    cz_pipeline.set_edit_loras = lambda slots: applied.append(list(slots))
    cz_edit_loras.is_downloaded = lambda name, index=None: True
    cz_ui._edit_lora_choices = lambda: ["None", "Relight Moss ✓"]
    try:
        with _Declared():
            dd, note, slider = cz_ui._ui_edit_lora_pick("Relight Moss ✓", 1.0,
                                                        progress=lambda *a, **k: None)
            assert applied == [[("Relight Moss", 0.7)]], applied
            assert slider.get("value") == 0.7, slider
            assert "0.7" in note, note
            # "None": nothing to apply, the slider does not move.
            dd, note, slider = cz_ui._ui_edit_lora_pick("None", 0.8, progress=lambda *a, **k: None)
            assert applied[-1] == [] and "value" not in slider, (applied, slider)
    finally:
        cz_pipeline.set_edit_loras, cz_edit_loras.is_downloaded, cz_ui._edit_lora_choices = real


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"All {len(tests)} edit-LoRA preset tests passed.")
