"""The model profiles and presets of crispz-moss (rewritten from crispz-ideogram's).

The trap this guards: profile_for_model matches by substring, first match wins, and an
unknown name (a local diffusers folder, a renamed mirror, a CivitAI single file) falls
back on the default profile. Inherited from Ideogram, that default was 48 steps at 7.0:
on MOSS-Image a CFG of 7 burns the image. Every path must give 50 / 3.0.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_core  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "OpenMOSS-Team/MOSS-Image-SFT"


def _with_profiles(profiles, fn, default=None):
    old, old_d = cz_core.MODEL_PROFILES, cz_core.DEFAULT_MODEL_PROFILE
    cz_core.MODEL_PROFILES = profiles
    if default is not None:
        cz_core.DEFAULT_MODEL_PROFILE = default
    try:
        return fn()
    finally:
        cz_core.MODEL_PROFILES, cz_core.DEFAULT_MODEL_PROFILE = old, old_d


def test_the_built_in_profiles_give_moss_50_steps_at_3():
    assert cz_core.profile_for_model(REPO) == (50, 3.0)
    assert cz_core.profile_for_model("OpenMOSS-Team/MOSS-Image-Base") == (50, 3.0)
    assert cz_core.profile_for_model("D:/models/MOSS_Image_local") == (50, 3.0)
    assert cz_core.profile_for_model("dirtyMoss_3317613.safetensors") == (50, 3.0)
    # an unknown name (a renamed local folder) falls back on the same profile
    assert cz_core.profile_for_model("D:/models/my-poster-model") == (50, 3.0)


def test_an_empty_default_profile_still_gives_50_at_3():
    """A config.txt whose default_model_profile lacks a field: the code fallback is
    MOSS's, not Ideogram's 48 / 7.0 nor FLUX.3's 16 / 1.0."""
    got = _with_profiles({}, lambda: cz_core.profile_for_model("mystery"), default={})
    assert got == (50, 3.0)
    got = _with_profiles({"moss": {}}, lambda: cz_core.profile_for_model(REPO), default={})
    assert got == (50, 3.0)


def test_config_sample_matches_the_code_and_starts_on_moss():
    with open(os.path.join(ROOT, "config-sample.txt"), encoding="utf-8") as f:
        cfg = json.load(f)
    keys = list(cfg["model_profiles"])
    assert "klein" not in keys and "flux-2" not in keys and "flux2" not in keys
    sample = {k: (int(v["steps"]), float(v["guidance"])) for k, v in cfg["model_profiles"].items()}
    built_in = {k: (int(v["steps"]), float(v["guidance"])) for k, v in cz_core.MODEL_PROFILES.items()}
    assert sample == built_in
    assert cfg["default_model_profile"] == {"steps": 50, "guidance": 3.0}
    presets = cfg["performance_presets"]
    assert cfg["default_performance"] in presets
    assert presets[cfg["default_performance"]][:2] == [50, 3.0]
    assert cfg["default_gen_steps"] == 50 and cfg["default_guidance"] == 3.0
    assert cfg["moss_model"] == REPO and "klein_model" not in cfg and "flux3_model" not in cfg
    assert "ideogram_model" not in cfg and "ideogram_base_repos" not in cfg
    assert cfg["moss_base_repos"][0] == REPO and "flux3_base_repos" not in cfg
    assert cfg["moss_precision"] == "bf16"
    # crispz_flux3_dir is this studio's own setting (the Edit in crispz-flux3 tab, lot 5),
    # not a FLUX.3 leftover
    for k in cfg:
        if k.lstrip("_").startswith("crispz_flux3_dir"):
            continue
        assert "klein_" not in k and "flux3_" not in k and "ideogram_" not in k, k
    with open(os.path.join(ROOT, "preferences.example.json"), encoding="utf-8") as f:
        prefs = json.load(f)
    assert prefs["moss_model"] == REPO and "ideogram_model" not in prefs


def test_the_ui_presets_mirror_the_sample():
    with open(os.path.join(ROOT, "config-sample.txt"), encoding="utf-8") as f:
        cfg = json.load(f)
    import cz_ui
    sample = {k: (int(v[0]), float(v[1])) for k, v in cfg["performance_presets"].items()}
    assert set(sample) >= set(cz_ui.PERFORMANCE) or cz_ui.PERFORMANCE == sample


def test_set_precision_knows_bf16_and_nf4_and_persists_the_moss_key(monkeypatch):
    """The precision is the TEXT ENCODER's on crispz-moss (bf16 or nf4)."""
    import cz_pipeline as P
    lines = []
    monkeypatch.setattr(P, "_log", lambda m: lines.append(str(m)))
    monkeypatch.setattr(P, "PRECISION", "bf16")
    monkeypatch.setattr(P, "ZIMAGE_TRANSFORMER", None)
    assert P.set_precision("bf16") == "bf16"                  # unchanged: nothing said
    assert not lines
    assert P.set_precision("fp8r") == "bf16"                  # unknown -> bf16, said
    assert any("unknown" in m for m in lines)
    assert P.set_precision("nf4") == "nf4"
    # the UI handler persists under the key the engine READS back
    import cz_ui
    saved = {}
    monkeypatch.setattr(cz_ui, "_save_prefs_keys", lambda d: saved.update(d))
    monkeypatch.setattr(P, "PRECISION", "bf16")
    msg = cz_ui._ui_set_precision("nf4")
    assert saved == {"moss_precision": "nf4"} and "nf4" in msg


def test_refine_steps_are_the_measured_ones():
    """Lot 7, measured on the 5090: at the presets' denoise values a refine at 12 steps
    and at 36 give the same image (mean difference 2/255, texts unchanged), 36 being 53 %
    slower. The presets keep the short counts; the slider still reaches 48 and beyond."""
    import cz_ui
    for name, p in cz_ui.PRESETS.items():
        if "steps" in p:
            assert 12 <= p["steps"] <= 16, (name, p["steps"])
    with open(os.path.join(ROOT, "config-sample.txt"), encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["default_refine_steps"] == 12
