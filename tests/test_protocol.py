"""Unit tests for cz_protocol (family CLI protocol v1).
No torch, no GPU: the heavy path (run_gen) is exercised against a FAKE
cz_pipeline injected in sys.modules, and the remote route against a local
HTTP server that mimics the Gradio /gradio_api/call endpoints.

Run:  .venv/Scripts/python tests/test_protocol.py
"""
import os
import io
import sys
import json
import types
import shutil
import tempfile
import threading
import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402

import cz_protocol as cp  # noqa: E402


# ------------------------------------------------------------------ caps ----
def test_caps_shape():
    caps = cp.caps_dict()
    assert caps["ok"] and caps["protocol"] == 1
    assert caps["tool"] == cp.TOOL
    assert "gen" in caps["ops"] and "caps" in caps["ops"]
    for key in ("loras", "refs", "seed", "negative", "arbitrary_size"):
        assert key in caps["supports"]


# ----------------------------------------------------------- validate_spec ---
def test_validate_spec_minimal_and_defaults():
    spec, warnings = cp.validate_spec({"protocol": 1, "prompt": "a cat"})
    assert spec["width"] == 1024 and spec["height"] == 1024
    assert spec["seed"] == -1 and spec["steps"] is None
    assert warnings == []


def test_validate_spec_missing_protocol_is_code_2():
    try:
        cp.validate_spec({"prompt": "a cat"})
    except cp.SpecError as e:
        assert e.code == 2
    else:
        raise AssertionError("missing protocol should raise")


def test_validate_spec_wrong_protocol_is_code_3():
    try:
        cp.validate_spec({"protocol": 99, "prompt": "x"})
    except cp.SpecError as e:
        assert e.code == 3
    else:
        raise AssertionError("protocol 99 should raise")


def test_validate_spec_unknown_field_warns_never_fails():
    spec, warnings = cp.validate_spec(
        {"protocol": 1, "prompt": "x", "qwen_only_knob": 3})
    assert any("qwen_only_knob" in w for w in warnings)


def test_validate_spec_guidance_kept_count_warn():
    # guidance is a per-call CFG on every op since 2026-11-09: kept, no warning
    spec, warnings = cp.validate_spec(
        {"protocol": 1, "prompt": "x", "guidance": 3.5, "count": 4})
    text = " ".join(warnings)
    assert spec["guidance"] == 3.5
    assert "guidance" not in text
    assert "count forced to 1" in text
    _spec, warnings = cp.validate_spec({"protocol": 1, "prompt": "x"})
    assert "guidance" not in _spec
    for bad in ("high", -1, 21):
        try:
            cp.validate_spec({"protocol": 1, "prompt": "x", "guidance": bad})
        except cp.SpecError as e:
            assert e.code == 2
        else:
            raise AssertionError(f"guidance {bad!r} should raise")


def test_validate_spec_bad_values_are_code_2():
    for bad in ({"protocol": 1, "prompt": ""},
                {"protocol": 1, "prompt": "x", "width": "huge"},
                {"protocol": 1, "prompt": "x", "height": 32},
                {"protocol": 1, "prompt": "x", "seed": "abc"}):
        try:
            cp.validate_spec(bad)
        except cp.SpecError as e:
            assert e.code == 2
        else:
            raise AssertionError(f"{bad} should raise")


# ------------------------------------------------------------------ refs v2 ---
@contextlib.contextmanager
def _omni_config(value="some/omni-model"):
    """crispz-moss: editing is available when the Moss-Image-Edit LoRA is on disk
    (cz_edit_loras.edit_ready), not from a config key. The helper keeps its name for the
    family's tests and fakes that presence: a value = on disk, '' = absent."""
    import cz_edit_loras
    old = cz_edit_loras.edit_ready
    cz_edit_loras.edit_ready = lambda index=None: bool(value)
    try:
        yield
    finally:
        cz_edit_loras.edit_ready = old


def test_caps_refs_follow_omni_support_and_config():
    # Families with no omni pipeline (krea/krea2): refs=False WHATEVER the
    # config says - never promise a capability whose loading would raise.
    if not cp.OMNI_SUPPORTED:
        with _omni_config():
            assert cp.caps_dict()["supports"]["refs"] is False
        return
    # LoRA absent -> no refs, no edit, whatever OMNI_DEFAULT (the base repo) says
    with _omni_config(""):
        assert cp.caps_dict()["supports"]["refs"] is False
        assert cp.caps_dict()["supports"]["edit"] is False
    with _omni_config():
        caps = cp.caps_dict()
        assert caps["supports"]["refs"] is True
        assert caps["supports"]["max_refs"] == cp.MAX_REFS


def test_validate_refs_dropped_with_warning_when_unavailable():
    with _omni_config(""):                      # the edit LoRA not on disk
        spec, warnings = cp.validate_spec(
            {"protocol": 1, "prompt": "x", "refs": ["a.png"]})
    assert spec["refs"] == []
    assert any("ref(s) ignored" in w and "MOSS-Image" in w for w in warnings), warnings


def test_validate_refs_missing_file_is_code_2():
    if not cp.OMNI_SUPPORTED:
        return
    with _omni_config():
        try:
            cp.validate_spec({"protocol": 1, "prompt": "x",
                              "refs": ["Z:/nope/missing.png"]})
        except cp.SpecError as e:
            assert e.code == 2 and "not found on disk" in str(e)
        else:
            raise AssertionError("missing ref should raise")


def test_validate_refs_capped_at_max():
    if not cp.OMNI_SUPPORTED:
        return
    d = tempfile.mkdtemp(prefix="cz_refs_")
    try:
        paths = []
        for i in range(cp.MAX_REFS + 2):          # FLUX.3: 10, two too many
            p = os.path.join(d, f"r{i}.png")
            Image.new("RGB", (8, 8)).save(p)
            paths.append(p)
        with _omni_config():
            spec, warnings = cp.validate_spec(
                {"protocol": 1, "prompt": "x", "refs": paths})
        assert len(spec["refs"]) == cp.MAX_REFS
        assert any("keeping the first" in w for w in warnings)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ----------------------------------------------------- run_gen (fake pipe) ---
class _FakePipeline(types.ModuleType):
    LORA_WEIGHT = 1.0
    _LAST_SEED = 424242
    GUIDANCE = 3.0          # the instance's slider (MOSS default)

    def __init__(self):
        super().__init__("cz_pipeline")
        self.calls = {"loras": None, "model": None, "gen": None,
                      "omni": None, "omni_kw": None, "cfg_seen": None}

    def set_loras(self, slots):
        self.calls["loras"] = slots
        self.LORAS = [(p, float(w)) for p, w in slots]

    def set_zimage_model(self, m):
        self.calls["model"] = m

    def _seen(self):
        self.calls["cfg_seen"] = self.GUIDANCE
        self.calls["loras_seen"] = list(self.LORAS)

    def txt2img_run(self, prompt, w, h, steps, seed, negative=""):
        self.calls["gen"] = (prompt, w, h, steps, seed, negative)
        self._seen()
        return Image.new("RGB", (w, h), "#204060"), {"txt2img": 1.5}

    def process_one(self, image, model, factor, denoise, steps, prompt, seed,
                    tile, overlap, do_esrgan=True):
        self._seen()
        return image.copy(), {"refine": 0.5}

    def inpaint_run(self, bg, mask, prompt, steps, denoise, seed):
        self._seen()
        return bg.copy()

    def generate_omni(self, refs, prompt, negative, w, h, steps, seed, **kw):
        # **kw: the optional kwargs some forks pass to the omni call
        # (guidance, honor_size, steps_explicit). A fake with a frozen
        # signature would fail there without saying anything about the real
        # behaviour.
        self.calls["omni"] = (len(refs), prompt, negative, w, h, steps, seed)
        self.calls["omni_kw"] = kw
        return Image.new("RGB", (w, h), "#106040")


@contextlib.contextmanager
def _fake_pipeline():
    fake = _FakePipeline()
    fake.LORAS = [("C:/loras/app_style.safetensors", 0.7)]   # the app's UI slots
    old = sys.modules.get("cz_pipeline")
    sys.modules["cz_pipeline"] = fake
    try:
        yield fake
    finally:
        if old is not None:
            sys.modules["cz_pipeline"] = old
        else:
            sys.modules.pop("cz_pipeline", None)


def test_run_gen_saves_image_and_reports_seed():
    d = tempfile.mkdtemp(prefix="cz_proto_")
    try:
        with _fake_pipeline() as fake:
            spec, warnings = cp.validate_spec(
                {"protocol": 1, "prompt": "a cat", "width": 128, "height": 96,
                 "seed": -1, "loras": ["ink.safetensors:0.8", "flat"],
                 "out_dir": d})
            res = cp.run_gen(spec, warnings)
        assert res["ok"] and res["route"] == "local"
        # seed -1 is resolved to a concrete value BEFORE generating: the seed
        # reported is the one passed to the pipeline (replayable), and _LAST_SEED
        # is set for the UI's 'Reuse last seed'.
        assert res["seed_used"] >= 0
        assert fake.calls["gen"][4] == res["seed_used"]
        assert fake._LAST_SEED == res["seed_used"]
        assert len(res["images"]) == 1 and os.path.isfile(res["images"][0])
        assert res["images"][0].startswith(os.path.abspath(d))
        # default steps = the tool's default_gen_steps (varies per fork)
        from cz_core import CONFIG as _cfg
        assert fake.calls["gen"][1:4] == \
            (128, 96, int(_cfg.get("default_gen_steps", 8)))
        assert fake.calls["loras"] == [("ink.safetensors", 0.8), ("flat", 1.0)]
        with Image.open(res["images"][0]) as im:
            assert im.size == (128, 96)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_run_gen_explicit_seed_is_passed_and_reported():
    d = tempfile.mkdtemp(prefix="cz_proto_s_")
    try:
        with _fake_pipeline() as fake:
            spec, w = cp.validate_spec(
                {"protocol": 1, "prompt": "x", "width": 64, "height": 64,
                 "seed": 42, "out_dir": d})
            res = cp.run_gen(spec, w)
        assert res["seed_used"] == 42
        assert fake.calls["gen"][4] == 42
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_spec_guidance_is_the_cfg_of_that_call_only_on_every_op():
    # crispz-motion drives the Turbo LoRA (CFG 1) through the protocol while
    # the instance slider stays at 3.0: gen, upscale (img2img at factor 1) and
    # inpaint render at the spec's CFG, the slider comes back after each call,
    # and a spec without guidance keeps the slider.
    d = tempfile.mkdtemp(prefix="cz_proto_g_")
    old_esrgan = sys.modules.get("cz_esrgan")
    sys.modules["cz_esrgan"] = types.SimpleNamespace(
        list_esrgan_models=lambda: [], ESRGAN_DIR=d)
    try:
        src = os.path.join(d, "src.png")
        mask = os.path.join(d, "mask.png")
        Image.new("RGB", (64, 64), "#336699").save(src)
        Image.new("L", (64, 64), 255).save(mask)
        base = {"protocol": 1, "prompt": "x", "width": 64, "height": 64,
                "seed": 7, "out_dir": d}
        cases = (("gen", cp.run_gen, {}),
                 ("upscale", cp.run_upscale, {"input": src, "factor": 1.0,
                                              "denoise": 0.35}),
                 ("inpaint", cp.run_inpaint, {"input": src, "mask": mask}))
        with _fake_pipeline() as fake:
            for op, run, extra in cases:
                spec, w = cp.validate_spec(dict(base, op=op, guidance=1.0,
                                                **extra), op=op)
                res = run(spec, w)
                assert res["ok"], (op, res)
                assert fake.calls["cfg_seen"] == 1.0, op
                assert res["guidance"] == 1.0, op
                assert fake.GUIDANCE == 3.0, f"{op}: slider not restored"
                assert not any("guidance" in x for x in res["warnings"]), op
                spec, w = cp.validate_spec(dict(base, op=op, **extra), op=op)
                res = run(spec, w)
                assert fake.calls["cfg_seen"] == 3.0, op
                assert "guidance" not in res, op
        assert cp.caps_dict()["supports"]["guidance"] is True
    finally:
        if old_esrgan is not None:
            sys.modules["cz_esrgan"] = old_esrgan
        else:
            sys.modules.pop("cz_esrgan", None)
        shutil.rmtree(d, ignore_errors=True)


def test_spec_loras_apply_to_every_op_and_the_app_set_comes_back():
    # crispz-motion chains keys through upscale (img2img) with the Turbo LoRA:
    # spec.loras must reach upscale and inpaint, not only gen; a draft preset
    # name resolves to its file like the UI's Performance preset; the app's own
    # slots are back after each call; no spec.loras = the app's set.
    import cz_edit_loras
    d = tempfile.mkdtemp(prefix="cz_proto_l_")
    old_esrgan = sys.modules.get("cz_esrgan")
    sys.modules["cz_esrgan"] = types.SimpleNamespace(
        list_esrgan_models=lambda: [], ESRGAN_DIR=d)
    turbo = os.path.join(d, "moss-turbo-4-steps.safetensors")
    old_resolve = cz_edit_loras.resolve_draft
    cz_edit_loras.resolve_draft = lambda name, **k: turbo
    try:
        src = os.path.join(d, "src.png")
        mask = os.path.join(d, "mask.png")
        Image.new("RGB", (64, 64), "#336699").save(src)
        Image.new("L", (64, 64), 255).save(mask)
        base = {"protocol": 1, "prompt": "x", "width": 64, "height": 64,
                "seed": 7, "out_dir": d}
        cases = (("gen", cp.run_gen, {}),
                 ("upscale", cp.run_upscale, {"input": src, "factor": 1.0,
                                              "denoise": 0.35}),
                 ("inpaint", cp.run_inpaint, {"input": src, "mask": mask}))
        with _fake_pipeline() as fake:
            app = list(fake.LORAS)
            for op, run, extra in cases:
                spec, w = cp.validate_spec(dict(
                    base, op=op, loras=["moss-turbo-4-steps:1", "ink:0.5"],
                    **extra), op=op)
                res = run(spec, w)
                assert res["ok"], (op, res)
                assert fake.calls["loras_seen"] == [(turbo, 1.0), ("ink", 0.5)], op
                assert fake.LORAS == app, f"{op}: the app's LoRAs not restored"
                spec, w = cp.validate_spec(dict(base, op=op, **extra), op=op)
                run(spec, w)
                assert fake.calls["loras_seen"] == app, op
            # a failing render still gives the app its LoRAs back
            fake.process_one = lambda *a, **k: (_ for _ in ()).throw(
                RuntimeError("boom"))
            spec, w = cp.validate_spec(dict(base, op="upscale", loras=["ink"],
                                            input=src, factor=1.0), op="upscale")
            try:
                cp.run_upscale(spec, w)
            except RuntimeError:
                pass
            assert fake.LORAS == app
    finally:
        cz_edit_loras.resolve_draft = old_resolve
        if old_esrgan is not None:
            sys.modules["cz_esrgan"] = old_esrgan
        else:
            sys.modules.pop("cz_esrgan", None)
        shutil.rmtree(d, ignore_errors=True)


def test_lora_slots_weights_and_windows_paths():
    with _fake_pipeline():
        spec = {"loras": ["a.safetensors:0.8", "b", "D:/x/c.safetensors",
                          "D:/x/d.safetensors:0.3"]}
        assert cp._lora_slots(spec, resolve_drafts=False) == [
            ("a.safetensors", 0.8), ("b", 1.0), ("D:/x/c.safetensors", 1.0),
            ("D:/x/d.safetensors", 0.3)]


def test_spec_guidance_is_restored_when_the_render_fails():
    with _fake_pipeline() as fake:
        def boom(*a, **k):
            raise RuntimeError("out of memory")
        fake.txt2img_run = boom
        spec, w = cp.validate_spec({"protocol": 1, "prompt": "x",
                                    "guidance": 1.0, "out_dir": tempfile.gettempdir()})
        try:
            cp.run_gen(spec, w)
        except RuntimeError:
            pass
        else:
            raise AssertionError("the render error must propagate")
        assert fake.GUIDANCE == 3.0


def test_run_gen_with_refs_takes_the_omni_route():
    if not cp.OMNI_SUPPORTED:
        return
    d = tempfile.mkdtemp(prefix="cz_proto_o_")
    try:
        ref = os.path.join(d, "lea.png")
        Image.new("RGB", (16, 16), "#aa3355").save(ref)
        with _omni_config(), _fake_pipeline() as fake:
            spec, w = cp.validate_spec(
                {"protocol": 1, "prompt": "portrait of lea", "width": 64,
                 "height": 64, "seed": 5, "refs": [ref], "out_dir": d})
            res = cp.run_gen(spec, w)
        assert res["ok"] and res["refs_used"] == 1
        assert fake.calls["omni"] is not None and fake.calls["gen"] is None
        n_refs, _pr, _neg, _w, _h, _steps, seed = fake.calls["omni"]
        assert n_refs == 1 and seed == 5
        assert "omni" in res["timings"]
        assert os.path.isfile(res["images"][0])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_handle_gen_json_remote_refuses_model_override():
    d = tempfile.mkdtemp(prefix="cz_proto_m_")
    try:
        with _fake_pipeline() as fake:
            res = cp.handle_gen_json(json.dumps(
                {"protocol": 1, "prompt": "x", "width": 64, "height": 64,
                 "model": "other.safetensors", "out_dir": d}))
        assert res["ok"] and res["route"] == "remote"
        assert fake.calls["model"] is None                 # never applied
        assert any("model override ignored" in w for w in res["warnings"])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_handle_gen_json_bad_spec_is_clean():
    res = cp.handle_gen_json("{not json")
    assert res["ok"] is False and res["exit_code"] == 1
    res = cp.handle_gen_json(json.dumps({"protocol": 1, "prompt": ""}))
    assert res["ok"] is False and res["exit_code"] == 2


# ------------------------------------------------------------ remote route ---
class _GradioMock(BaseHTTPRequestHandler):
    """Mime /gradio_api/call/<name> (POST -> event_id, GET -> flux SSE)."""
    caps_reply = json.dumps({"ok": True, "protocol": 1, "tool": "mock-tool",
                             "version": "9.9", "ops": ["caps", "gen"],
                             "supports": {}})
    gen_reply = json.dumps({"ok": True, "protocol": 1, "tool": "mock-tool",
                            "route": "remote", "images": ["X:/img.png"],
                            "seed_used": 7, "timings": {}, "warnings": []})

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self._json({"event_id": "ev1"})

    def do_GET(self):
        reply = self.caps_reply if "cli_caps" in self.path else self.gen_reply
        body = "data: " + json.dumps([reply]) + "\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def _json(self, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):                             # silence
        pass


@contextlib.contextmanager
def _mock_instance():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _GradioMock)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()


def test_probe_instance_finds_the_mock_and_none_otherwise():
    with _mock_instance() as url:
        caps = cp.probe_instance(url)
        assert caps and caps["tool"] == "mock-tool"
    assert cp.probe_instance("http://127.0.0.1:9", timeout=1) is None


def _main_capture(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cp.main(argv)
    return code, json.loads(buf.getvalue())


def test_main_gen_routes_to_running_instance():
    d = tempfile.mkdtemp(prefix="cz_proto_r_")
    try:
        sp = os.path.join(d, "spec.json")
        with open(sp, "w", encoding="utf-8") as f:
            json.dump({"protocol": 1, "prompt": "a cat"}, f)
        with _mock_instance() as url:
            code, res = _main_capture(["gen", "--spec", sp, "--remote", url])
        assert code == 0 and res["ok"]
        assert res["tool"] == "mock-tool" and res["route"] == "remote"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_main_accepts_utf8_bom_spec():
    # PowerShell 5.1 (Set-Content -Encoding utf8) writes a BOM: the protocol
    # has to accept it, Windows is the main ground here.
    d = tempfile.mkdtemp(prefix="cz_proto_bom_")
    try:
        sp = os.path.join(d, "spec.json")
        with open(sp, "w", encoding="utf-8-sig") as f:
            json.dump({"protocol": 1, "prompt": "a cat"}, f)
        with _mock_instance() as url:
            code, res = _main_capture(["gen", "--spec", sp, "--remote", url])
        assert code == 0 and res["ok"]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_main_forced_remote_unreachable_is_code_4():
    d = tempfile.mkdtemp(prefix="cz_proto_4_")
    try:
        sp = os.path.join(d, "spec.json")
        with open(sp, "w", encoding="utf-8") as f:
            json.dump({"protocol": 1, "prompt": "x"}, f)
        code, res = _main_capture(
            ["gen", "--spec", sp, "--remote", "http://127.0.0.1:9"])
        assert code == 4 and res["ok"] is False
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_ops_is_a_subset_of_the_v1_vocabulary():
    # PROTOCOL_OPS = the v1 vocabulary (frozen, identical in every fork),
    # OPS = what THIS tool implements. The CLI validates the op against the
    # vocabulary, never against OPS: otherwise an op the tool does not
    # implement would come out as an argparse usage (code 2, on stderr)
    # instead of the JSON code 3.
    assert set(cp.OPS) <= set(cp.PROTOCOL_OPS)
    assert "edit" in cp.PROTOCOL_OPS and "inpaint" in cp.PROTOCOL_OPS
    assert cp.caps_dict()["ops"] == list(cp.OPS)


def test_main_unsupported_op_is_code_3():
    # simulate a tool that does not implement 'edit' (OPS reduced): the op
    # stays in the v1 vocabulary, so the CLI accepts it and answers code 3
    # in JSON.
    old = cp.OPS
    cp.OPS = ("caps", "gen")
    try:
        code, res = _main_capture(["edit", "--spec", "-"])
    finally:
        cp.OPS = old
    assert code == 3 and "not supported" in res["error"]


def test_main_gen_without_spec_is_code_2():
    code, res = _main_capture(["gen"])
    assert code == 2


def test_main_caps_reports_instance():
    with _mock_instance() as url:
        code, res = _main_capture(["caps", "--remote", url])
    assert code == 0 and res["ok"]
    assert res["instance"]["running"] and res["instance"]["tool"] == "mock-tool"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"OK {fn.__name__}")
    print(f"All {len(tests)} protocol tests passed.")
