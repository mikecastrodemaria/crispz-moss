"""cz_magic: plain idea -> MOSS layout draft, two local engines (lot 2).

No network, no GPU: Ollama is mocked at its transport, the local engine runs the real
MossPipe.magic_prompt / describe_image on the tiny MOSS (random weights, so the text is
garbage: what is checked is the plumbing, the encoder-only offload and the LM head),
and the one-tensor LM head download is exercised against a fake HTTP server.

Run:  .venv/Scripts/python -m pytest tests/test_magic_prompt.py -q
"""
import json
import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest  # noqa: E402
import torch  # noqa: E402

import cz_caption  # noqa: E402
import cz_magic as M  # noqa: E402
import cz_moss  # noqa: E402

GOOD = {"width": 1024, "height": 1024, "bbox_coord_space": "abs", "bbox_format": "xyxy",
        "summary": {"meta": ["poster"], "canvas": "single",
                    "caption": "A jazz poster on teal, no other text."},
        "items": [{"label": "text", "caption": "Big cream headline.", "text": "JAZZ",
                   "bbox": [100, 80, 900, 300]},
                  {"label": "object", "caption": "A flat saxophone.", "bbox": [300, 400, 700, 900]}]}


@pytest.fixture
def settings(monkeypatch):
    vals = {}
    monkeypatch.setattr(M, "_setting", lambda k, d=None: vals.get(k, d))
    return vals


def test_clean_reads_what_llms_actually_return():
    raw = "<think>planning...</think>\n```json\n" + json.dumps(GOOD) + "\n```"
    text, warnings = M.clean(raw, 1024, 1024)
    assert warnings == [] and json.loads(text)["items"][0]["bbox"] == [100, 80, 900, 300]
    assert json.loads(text)["summary"]["meta"] == ["poster"]
    # prose around the object, a canvas the LLM did not honour: rescaled, said
    other = dict(GOOD, width=512, height=512)
    other["items"] = [dict(GOOD["items"][0], bbox=[50, 40, 450, 150])]
    text, warnings = M.clean("Sure! " + json.dumps(other) + " Enjoy.", 1024, 1024)
    assert any("drew its boxes for 512x512" in w for w in warnings)
    assert json.loads(text)["items"][0]["bbox"] == [100, 80, 900, 300]
    # an answer in thousandths (no width given, boxes under 1000 on a 2048 canvas)
    thou = {"summary": {"caption": "x"}, "items": [{"label": "object", "caption": "c",
                                                     "bbox": [0, 0, 500, 1000]}]}
    text, warnings = M.clean(json.dumps(thou), 2048, 512)
    assert any("thousandths" in w for w in warnings)
    assert json.loads(text)["items"][0]["bbox"] == [0, 0, 1024, 512]
    # a compiled answer (centred boxes) is read back as such
    comp, _, _ = cz_caption.compile(GOOD)
    text, warnings = M.clean(comp, 1024, 1024)
    assert any("compiled" in w for w in warnings)
    assert json.loads(text)["items"][0]["bbox"] == [100, 80, 900, 300]
    with pytest.raises(M.MagicPromptError, match="no JSON object"):
        M.clean("I cannot do that", 1024, 1024)
    with pytest.raises(M.MagicPromptError, match="invalid JSON"):
        M.clean('{"summary": : "broken"}', 1024, 1024)
    # an unterminated string is closed by the repair; the object then lacks items
    with pytest.raises(M.MagicPromptError, match="no 'items'"):
        M.clean('{"summary": "unterminated}', 1024, 1024)
    with pytest.raises(M.MagicPromptError, match="no 'items'"):
        M.clean('{"caption": "a plain caption"}', 1024, 1024)


def test_engine_setting(settings):
    assert M.engine() == "ollama"
    settings["magic_prompt_engine"] = "LOCAL"
    assert M.engine() == "local"
    settings["magic_prompt_engine"] = "ideogram_api"
    assert M.engine() == "ollama"
    assert M.visible_engines() == ("ollama", "local")


def test_ollama_gets_the_brief_the_schema_and_logs_its_url(settings, monkeypatch):
    import prompt_improve
    calls, logs = [], []

    def http(path, payload=None, base=None, timeout=8):
        calls.append((path, payload, base))
        if path == "/api/tags":
            return {"models": [{"name": "qwen3.5:9b"}]}
        return {"response": json.dumps(GOOD)}
    monkeypatch.setattr(prompt_improve, "http", http)
    monkeypatch.setattr(M, "_log", lambda m: logs.append(str(m)))
    text, warnings, used = M.expand("a jazz poster", 1024, 1024, seed=5, which="ollama")
    assert used == "ollama" and warnings == [] and json.loads(text)["items"][0]["text"] == "JAZZ"
    path, payload, base = calls[-1]
    assert path == "/api/generate" and payload["model"] == "qwen3.5:9b"
    assert "LAYOUT DRAFT" in payload["system"] and "no other text" in payload["system"]
    assert "1024 x 1024" in payload["prompt"] and "a jazz poster" in payload["prompt"]
    assert payload["format"] == M.OLLAMA_FORMAT and payload["options"]["seed"] == 5
    assert any("ollama" in m and str(base) in m for m in logs)       # the URL, logged
    settings["magic_prompt_ollama_model"] = "llama3"
    M.expand("a jazz poster", 1920, 320, which="ollama")
    assert calls[-1][1]["model"] == "llama3" and "6:1" in calls[-1][1]["prompt"]


def test_expand_refuses_empty_and_json(settings):
    with pytest.raises(M.MagicPromptError, match="empty"):
        M.expand("  ", 1024, 1024)
    with pytest.raises(M.MagicPromptError, match="already a layout"):
        M.expand(json.dumps(GOOD), 1024, 1024)


def test_local_engine_goes_through_the_pipeline(settings, monkeypatch):
    import cz_pipeline
    seen = []
    monkeypatch.setattr(cz_pipeline, "magic_prompt_local",
                        lambda p, w, h, seed=None: seen.append((p, w, h, seed)) or json.dumps(GOOD))
    text, warnings, used = M.expand("a jazz poster", 1024, 1024, seed=9, which="local")
    assert used == "local" and warnings == [] and seen == [("a jazz poster", 1024, 1024, 9)]


def _tiny_head(pipe, seed=0):
    cfg = pipe._pipe.text_encoder.config.text_config
    torch.manual_seed(seed)
    return torch.randn(cfg.vocab_size, cfg.hidden_size) * 0.02


def test_local_magic_prompt_plumbing_on_the_tiny_pipe():
    """The real MossPipe.magic_prompt: a head handed in (no download here), the generator
    built on the loaded encoder (no second copy), the encoder brought in for the call and
    sent back after under the encoder-only offload, the head too; seeded, so the same
    call gives the same text. Unconstrained (outlines is not the point here)."""
    from moss_tiny import tiny_pipe
    pipe = tiny_pipe()
    pipe.set_lm_head(_tiny_head(pipe))
    gen = pipe._get_generator()
    assert gen.model is pipe._pipe.text_encoder                 # shared, not copied
    assert tuple(gen.lm_head.weight.shape) == (248320, 16)
    pipe.enable_model_cpu_offload(device="cpu")
    moves = []
    te_to = pipe._pipe.text_encoder.to
    pipe._pipe.text_encoder.to = lambda *a, **k: moves.append(str(a[0])) or te_to(*a, **k)
    try:
        a = pipe.magic_prompt("a poster", 256, 256, seed=3, max_new_tokens=6, constrained=False)
        b = pipe.magic_prompt("a poster", 256, 256, seed=3, max_new_tokens=6, constrained=False)
    finally:
        del pipe._pipe.text_encoder.to
    assert isinstance(a, str) and a == b
    assert moves == ["cpu", "cpu", "cpu", "cpu"]        # in + out, twice (dev = cpu here)
    assert not pipe._te_on_gpu
    # the image render still works afterwards (the encoder is the same module)
    out = pipe(prompt="x", width=256, height=256, num_inference_steps=1,
               generator=torch.Generator().manual_seed(0))
    assert out.images[0].size == (256, 256)


def test_describe_plumbing_on_the_tiny_pipe():
    """The real MossPipe.describe_image on the tiny Qwen3.5 (vision tower of depth 1,
    random weights): the image goes through Qwen3.5's real processor and generate, the
    encoder comes and goes under the encoder-only offload. Greedy: deterministic."""
    from PIL import Image
    from moss_tiny import tiny_pipe
    pipe = tiny_pipe()
    pipe.set_lm_head(_tiny_head(pipe))
    pipe.enable_model_cpu_offload(device="cpu")
    img = Image.new("RGB", (640, 360), (200, 30, 30))
    a = pipe.describe_image(img, max_new_tokens=4, constrained=False)
    b = pipe.describe_image(img, max_new_tokens=4, constrained=False)
    assert isinstance(a, str) and a == b
    assert not pipe._te_on_gpu


def test_describe_is_constrained_to_the_draft_schema(monkeypatch):
    """Describe passes the same JSON schema as the magic prompt to the generator (an
    unconstrained description came back as invalid JSON on the card)."""
    from PIL import Image
    from moss_tiny import tiny_pipe
    pipe = tiny_pipe()
    pipe.set_lm_head(_tiny_head(pipe))
    seen = {}

    def fake(self, inputs, max_new_tokens, **kw):
        seen.update(kw)
        return "{}"
    monkeypatch.setattr(cz_moss.MossPipe, "_generate_text", fake)
    pipe.describe_image(Image.new("RGB", (64, 64)), max_new_tokens=4)
    assert seen["schema"] is M.OLLAMA_FORMAT
    pipe.describe_image(Image.new("RGB", (64, 64)), max_new_tokens=4, constrained=False)
    assert seen["schema"] is None


def test_clean_repairs_a_cut_or_malformed_answer():
    """An answer cut by the token budget (open string, dangling key, open arrays and
    objects) or with trailing commas is repaired and said; hopeless text still fails
    with the whole answer kept on the error."""
    full = {"summary": {"caption": "A sign."}, "items": [
        {"label": "text", "caption": "Red.", "text": "OPEN", "bbox": [10, 20, 300, 100]},
        {"label": "object", "caption": "A cup.", "bbox": [10, 120, 300, 400]}]}
    body = json.dumps(full)
    cut_in_string = body[:body.index('"A cup."') + 4]
    cut_at_key = body[:body.index('"bbox": [10, 120')] + '"bbox":'
    cut_in_array = body[:body.index("300, 400]")] + "300,"
    for raw in (cut_in_string, cut_at_key, cut_in_array):
        text, warnings = M.clean(raw, 320, 480)
        d = json.loads(text)
        assert d["items"][0]["text"] == "OPEN" and len(d["items"]) in (1, 2), raw
        assert any("repaired" in w for w in warnings)
    trailing = body.replace('"bbox": [10, 20, 300, 100]}', '"bbox": [10, 20, 300, 100,],}')
    text, warnings = M.clean(trailing, 320, 480)
    assert len(json.loads(text)["items"]) == 2 and any("repaired" in w for w in warnings)
    assert M._repair_json(body) is None
    with pytest.raises(M.MagicPromptError) as e:
        M.clean("{ this is not json at all", 320, 480)
    assert e.value.raw == "{ this is not json at all"


def test_describe_cleans_to_a_draft_at_the_image_size(monkeypatch):
    import cz_pipeline
    from PIL import Image
    raw = json.dumps({"summary": {"caption": "A sign."}, "items": [
        {"label": "text", "bbox": [100, 200, 900, 400], "text": "OPEN", "caption": "Red."}]})
    monkeypatch.setattr(cz_pipeline, "describe_local", lambda img: raw)
    text, warnings, used = M.describe(Image.new("RGB", (1000, 500)), which="local")
    d = json.loads(text)
    assert (d["width"], d["height"]) == (992, 496) and warnings == [] and used == "local"
    assert d["items"][0]["text"] == "OPEN" and d["items"][0]["bbox"][0] == 99
    with pytest.raises(M.MagicPromptError, match="no image"):
        M.describe(None)


def test_lm_head_cache_and_ranged_download(monkeypatch, tmp_path):
    """download_lm_head fetches ONE tensor with HTTP ranges on the shard (a fake server
    over a local shard file here), writes a one-tensor safetensors, and the cache reads
    it back; ensure_lm_head refuses a head of the wrong size."""
    from safetensors.torch import save_file
    shard = tmp_path / "model.safetensors-00001-of-00004.safetensors"
    head = torch.randn(248320 // 100, 16).to(torch.bfloat16)        # a small stand-in
    save_file({"a.weight": torch.zeros(3, 3), "lm_head.weight": head.contiguous(),
               "z.weight": torch.ones(2, 2)}, str(shard))
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"lm_head.weight": shard.name}}))
    data = shard.read_bytes()

    class _Resp:
        def __init__(self, rng):
            a, b = (int(x) for x in rng.replace("bytes=", "").split("-"))
            self.content = data[a:b + 1]
            self.status_code = 206

        def iter_content(self, chunk_size):
            for i in range(0, len(self.content), chunk_size):
                yield self.content[i:i + chunk_size]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    import requests
    monkeypatch.setattr(requests, "get", lambda url, headers=None, **k: _Resp(headers["Range"]))
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda repo, fn, **k: str(index) if fn.endswith(".json") else str(shard))
    monkeypatch.setattr(huggingface_hub, "hf_hub_url", lambda repo, fn: "http://x/" + fn)
    dest = str(tmp_path / "cache" / "head.safetensors")
    cz_moss.download_lm_head("any/repo", dest)
    got = cz_moss.load_lm_head(dest)
    assert torch.equal(got, head)
    # the fallback: the range refused -> the whole shard is read
    monkeypatch.setattr(cz_moss, "_range_fetch_tensor",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("HTTP 200")))
    dest2 = str(tmp_path / "cache" / "head2.safetensors")
    cz_moss.download_lm_head("any/repo", dest2)
    assert torch.equal(cz_moss.load_lm_head(dest2), head)
    # ensure_lm_head: the cache is used, and a wrong size is refused by name
    from moss_tiny import tiny_pipe
    pipe = tiny_pipe()
    with pytest.raises(RuntimeError, match="does not fit"):
        pipe.ensure_lm_head(cache=dest)
    right = str(tmp_path / "right.safetensors")
    save_file({"lm_head.weight": torch.zeros(248320, 16)}, right)
    assert tuple(pipe.ensure_lm_head(cache=right).shape) == (248320, 16)


def test_no_dynamo_without_triton(monkeypatch):
    """No Triton (Windows): torch.compile is off for the generation call, so outlines'
    compiled kernel runs eagerly instead of printing a TritonMissing traceback."""
    import importlib.util
    import torch._dynamo
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda n, *a: None if n == "triton" else real(n, *a))
    with cz_moss._no_compile_without_triton():
        assert torch._dynamo.config.disable is True
    assert torch._dynamo.config.disable is False


def test_schema_processor_is_built_or_said(monkeypatch):
    """With outlines installed a JSON logits processor comes back for the draft schema;
    without it, None and one log line."""
    from moss_tiny import tiny_pipe
    pipe = tiny_pipe()
    pipe.set_lm_head(_tiny_head(pipe))
    procs = cz_moss._schema_processors(pipe._get_generator(), pipe._pipe.tokenizer,
                                       M.OLLAMA_FORMAT)
    try:
        import outlines  # noqa: F401
        assert procs is not None and len(procs) == 1
    except ImportError:
        assert procs is None


def test_describe_through_an_ollama_vision_model(settings, monkeypatch):
    """Engine Ollama: the magic prompt's model is used when /api/show says 'vision', the
    image goes as base64 with the brief (the image's size written in) and the draft
    schema; without any vision model the local engine takes over, said in the log."""
    import cz_ollama
    import cz_pipeline
    import prompt_improve
    from PIL import Image
    settings["magic_prompt_engine"] = "ollama"
    settings["magic_prompt_ollama_model"] = "gemma-vision:latest"
    answer = {"summary": {"caption": "A sign."}, "items": [
        {"label": "text", "bbox": [100, 50, 500, 150], "text": "OPEN", "caption": "Red."}]}
    calls = []

    def http(path, payload=None, base=None, timeout=8):
        calls.append((path, payload))
        if path == "/api/show":
            return {"capabilities": ["completion", "vision"]}
        return {"response": json.dumps(answer)}
    monkeypatch.setattr(prompt_improve, "http", http)
    monkeypatch.setattr(cz_pipeline, "describe_local", lambda img: (_ for _ in ()).throw(
        AssertionError("the local engine must not run")))
    text, warnings, used = M.describe(Image.new("RGB", (640, 320), (10, 20, 30)))
    assert used == "ollama" and json.loads(text)["items"][0]["text"] == "OPEN"
    gen = [pl for path, pl in calls if path == "/api/generate"][0]
    assert gen["model"] == "gemma-vision:latest" and gen["format"] is M.OLLAMA_FORMAT
    assert len(gen["images"]) == 1 and gen["images"][0][:4] == "/9j/"      # a JPEG
    assert "640 x 320" in gen["system"] and gen["prompt"] == M.DESCRIBE_USER
    assert gen["options"]["num_predict"] >= 2048 and gen["options"]["temperature"] == 0.0
    # the configured model has no vision: the first vision model installed is taken
    calls.clear()

    def http2(path, payload=None, base=None, timeout=8):
        calls.append((path, payload))
        if path == "/api/show":
            return {"capabilities": ["vision"] if payload["model"] == "llava:13b" else []}
        if path == "/api/tags":
            return {"models": [{"name": "text-only:latest"}, {"name": "llava:13b"}]}
        return {"response": json.dumps(answer)}
    monkeypatch.setattr(prompt_improve, "http", http2)
    assert M.ollama_vision_model() == "llava:13b"
    # no vision model at all: the local engine, said
    monkeypatch.setattr(prompt_improve, "http", lambda path, payload=None, base=None,
                        timeout=8: {"models": [], "capabilities": []})
    monkeypatch.setattr(cz_pipeline, "describe_local", lambda img: json.dumps(answer))
    said = []
    monkeypatch.setattr(M, "_log", lambda m: said.append(m))
    _, _, used = M.describe(Image.new("RGB", (640, 320)))
    assert used == "local" and any("no Ollama vision model" in m for m in said)
    # an explicit 'local' never touches Ollama
    monkeypatch.setattr(prompt_improve, "http", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("Ollama must not be called")))
    assert M.describe(Image.new("RGB", (64, 64)), which="local")[2] == "local"

