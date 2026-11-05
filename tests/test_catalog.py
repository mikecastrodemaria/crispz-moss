"""The MOSS-Image Character Catalog in the Asset Browser (cz_catalog), no network:
a fake catalog.json.gz stands in for the download. Also: the SPA carries the two tabs
and the "Send to crispz-moss" bridge, the app page listens, the UI shows the accordion.

Run: python -m pytest tests/test_catalog.py -q
"""
import gzip
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cz_catalog as C  # noqa: E402

SUFFIX = (". Zoomed-out full-body view, centered, with the entire subject occupying about 65 "
          "percent of the canvas. No text, logo, border, or watermark.")
ENTRIES = [
    {"id": "aa11", "name": "pikachu", "copyright": "pokemon", "kind": "character",
     "facets": {"character": ["pikachu"], "copyright": ["pokemon"], "hairColor": [],
                "gender": ["Non-human"], "outfit": []},
     "copy_tags": "pikachu, pokemon, no humans", "copy_prompt": "pikachu from pokemon. A non-human character.",
     "reuse_generation_prompt": "A single non-human character, pikachu from pokemon" + SUFFIX,
     "demo_image": "images/aa/aa11.webp", "demo_thumbnail": "thumbnails/aa/aa11.webp",
     "source_count": 11755},
    {"id": "bb22", "name": "jacinthe (pokemon)", "copyright": "pokemon", "kind": "character",
     "facets": {"character": ["jacinthe"], "copyright": ["pokemon"], "hairColor": ["blonde hair"],
                "gender": ["Female"], "outfit": ["dress"]},
     "copy_tags": "jacinthe, pokemon, 1girl, blonde hair", "copy_prompt": "jacinthe from pokemon. A character.",
     "reuse_generation_prompt": "A single character, jacinthe from pokemon. Appearance: blonde hair. Odd ending.",
     "demo_image": "images/bb/bb22.webp", "demo_thumbnail": "thumbnails/bb/bb22.webp",
     "source_count": 40},
    {"id": "cc33", "name": "Watercolor Painting (MOSS caption test)", "copyright": "Picture styles",
     "kind": "style", "facets": {"character": ["Watercolor Painting"], "copyright": ["Picture styles"]},
     "copy_tags": "Watercolor Painting", "copy_prompt": "A transparent watercolor painting.",
     "reuse_generation_prompt": "A transparent watercolor painting" + SUFFIX,
     "demo_image": "images/cc/cc33.webp", "demo_thumbnail": "thumbnails/cc/cc33.webp"},
    {"id": "dd44", "name": "no, comma", "copyright": "x", "kind": "character", "facets": {},
     "copy_tags": "", "copy_prompt": "", "reuse_generation_prompt": "",
     "demo_image": "", "demo_thumbnail": ""},
]


@pytest.fixture
def cached(tmp_path, monkeypatch):
    cache = tmp_path / "cache" / "character_catalog.json.gz"
    cache.parent.mkdir(parents=True)
    with gzip.open(cache, "wt", encoding="utf-8") as f:
        json.dump({"entries": ENTRIES, "version": "2026-10-31-test"}, f)
    monkeypatch.setattr(C, "CACHE", str(cache))
    monkeypatch.setattr(C, "CACHE_DIR", str(cache.parent))
    monkeypatch.setattr(C, "RENDERS_DIR", str(cache.parent / "character_catalog"))
    out = tmp_path / "out"
    out.mkdir()
    return str(out)


def test_browser_entries_shape_and_gen_suffix(cached):
    entries, version = C.load()
    assert len(entries) == 4 and version == "2026-10-31-test"
    tabs, suffix = C.browser_entries(entries, remote_thumbs=False)
    assert suffix == SUFFIX
    assert [c["name"] for c in tabs["characters"]] == ["jacinthe (pokemon)", "pikachu", "no, comma"]
    pika = next(c for c in tabs["characters"] if c["file"] == "aa11")
    assert pika["day"] == "pokemon" and pika["mode"] == "character"
    assert pika["gen"] == "A single non-human character, pikachu from pokemon" and "gen_full" not in pika
    assert pika["facets"] == {"gender": ["Non-human"]}           # empty and identity facets dropped
    assert pika["thumb"] == "" and pika["img"] == "" and pika["hf"] == "images/aa/aa11.webp"
    jac = next(c for c in tabs["characters"] if c["file"] == "bb22")
    assert jac["gen_full"] is True and jac["gen"].endswith("Odd ending.")
    assert [c["name"] for c in tabs["styles"]] == ["Watercolor Painting (MOSS caption test)"]
    tabs, _ = C.browser_entries(entries, remote_thumbs=True)
    pika = next(c for c in tabs["characters"] if c["file"] == "aa11")
    assert pika["thumb"] == C.DATASET_BASE + "thumbnails/aa/aa11.webp"
    assert pika["img"] == C.DATASET_BASE + "images/aa/aa11.webp"
    assert pika["thumb"].startswith("https://huggingface.co/datasets/hide1/")


def test_write_index_then_current_then_stale(cached):
    counts = C.write_index(cached, remote_thumbs=False)
    assert counts == {"characters": 3, "styles": 1}
    idx = os.path.join(cached, "_index")
    m = json.load(open(os.path.join(idx, "characters.json"), encoding="utf-8"))
    assert m["count"] == 3 and m["kind"] == "characters" and m["gen_suffix"] == SUFFIX
    assert m["notice"] == C.NOTICE and m["remote_thumbs"] is False and m["version"] == "2026-10-31-test"
    assert {c["file"] for c in m["images"]} == {"aa11", "bb22", "dd44"}
    assert C.index_is_current(cached, remote_thumbs=False)
    assert not C.index_is_current(cached, remote_thumbs=True)      # another thumbnail choice
    assert C.ensure_index(cached, remote_thumbs=False) is False     # nothing to do
    assert C.ensure_index(cached, remote_thumbs=True) is True       # rewritten
    assert json.load(open(os.path.join(idx, "characters.json"), encoding="utf-8"))["remote_thumbs"] is True
    os.utime(C.CACHE, None)                                         # a newer catalog file
    import time
    time.sleep(1.1)
    os.utime(C.CACHE, None)
    assert not C.index_is_current(cached, remote_thumbs=True)
    assert "characters" in C.status(cached)


def test_tag_rows_and_csv(cached, tmp_path):
    entries, _ = C.load()
    rows = C.tag_rows(entries)
    assert ("pikachu", 4, 11755) in rows and ("jacinthe (pokemon)", 4, 40) in rows
    assert ("pokemon", 3, 2) in rows                                 # the copyright, two characters
    assert ("Watercolor Painting", 1, 0) in rows                     # the style, suffix stripped
    assert not any("," in r[0] for r in rows)                        # a comma would break the CSV
    n = C.write_tags_csv(str(tmp_path / "tags"))
    text = open(tmp_path / "tags" / C.TAGS_CSV, encoding="utf-8").read()
    assert n == 4 and 'pikachu,4,11755,""' in text and text.startswith("#")


def test_download_is_one_said_call_to_huggingface(cached, monkeypatch):
    calls, said = [], []
    monkeypatch.setattr(C, "download_with_progress", lambda url, dst, label=None: calls.append((url, dst)))
    monkeypatch.setattr(C, "_log", lambda m: said.append(m))
    assert C.download() == C.CACHE and calls == []                  # cached: no call
    C.download(force=True)
    assert calls == [(C.CATALOG_URL, C.CACHE)]
    assert calls[0][0].startswith("https://huggingface.co/spaces/hide1/MOSS-Image-Character-Catalog/")
    assert any("CATALOG: downloading" in m and "huggingface.co" in m for m in said)


def test_not_available_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "CACHE", str(tmp_path / "none.gz"))
    assert not C.is_available()
    assert C.ensure_index(str(tmp_path), False) is False
    assert "not downloaded" in C.status(str(tmp_path))


def test_the_spa_and_the_app_page_carry_the_bridge():
    import cz_assets
    html = cz_assets.ASSET_BROWSER_HTML
    assert 'data-s="characters"' in html and 'data-s="styles"' in html
    assert "characters:'_index/characters.json'" in html and "styles:'_index/styles.json'" in html
    assert "function czSend(" in html and "BroadcastChannel('crispz-moss')" in html
    assert "cz_prompt_handoff" in html and "exSend()" in html
    assert "send(\\'gen\\',\\'replace\\')" in html             # the JS string holds escaped quotes
    assert "Send prompt to crispz-moss" in html and "Add triggers to the prompt" in html
    js = cz_assets.CZ_JS
    assert "BroadcastChannel('crispz-moss')" in js and "#cz_prompt textarea" in js
    assert "cz_prompt_handoff" in js and "new Event('input', {bubbles: true})" in js


def _js_one_expression(js):
    """True when `js` is ONE expression: an arrow function whose closing brace is the
    last token. Comments and string literals are skipped when counting braces."""
    i, depth, n, opened = 0, 0, len(js), False
    while i < n:
        c = js[i]
        if js.startswith("//", i):
            i = js.find("\n", i)
            i = n if i < 0 else i
            continue
        if js.startswith("/*", i):
            i = js.find("*/", i + 2) + 2
            continue
        if c in "'\"`":
            j = i + 1
            while j < n and js[j] != c:
                j += 2 if js[j] == "\\" else 1
            i = j + 1
            continue
        if c == "{":
            depth += 1
            opened = True
        elif c == "}":
            depth -= 1
            if depth == 0 and opened:
                return js[i + 1:].strip() == ""
        i += 1
    return False


def test_the_page_js_is_one_expression_gradio_can_evaluate():
    """Gradio evaluates Blocks(js=...) as ONE expression (new AsyncFunction('return (' +
    js + ')')). A second statement appended after the arrow function (the bridge was one,
    2026-11-05) is a SyntaxError that breaks every event handler of the page: no
    Advanced panel, no preset, nothing. Checked structurally, and compiled by node when
    it is installed."""
    import json
    import shutil
    import subprocess
    import cz_assets
    js = cz_assets.CZ_JS.replace("__MAP__", json.dumps({"a": "b"}))
    assert js.lstrip().startswith("() => {")
    assert _js_one_expression(js), "a statement follows the arrow function"
    assert not _js_one_expression("() => { a(); }\n(() => { b(); })();")   # the broken shape
    node = shutil.which("node")
    if node:
        code = ("const js = require('fs').readFileSync(0, 'utf8');"
                "const AF = Object.getPrototypeOf(async function(){}).constructor;"
                "new AF('return (' + js + ')');")
        r = subprocess.run([node, "-e", code], input=js, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


def test_the_ui_offers_the_catalog_accordion():
    import gradio as gr
    import cz_ui
    demo = cz_ui.build_ui()
    labels = {getattr(b, "label", None) for b in demo.blocks.values()}
    assert "Show thumbnails from huggingface.co" in labels
    assert any(isinstance(b, gr.Button) and "Download / update the catalog" in str(b.value)
               for b in demo.blocks.values())


def test_renders_on_disk_come_first(cached, monkeypatch):
    """download_renders fetches the dataset folder through huggingface_hub (mocked here:
    it writes two thumbnails) and says so; the cards then point at the local files, the
    remote URL only fills the gaps when the option is on; the index knows it is stale."""
    import huggingface_hub
    calls, said = [], []
    monkeypatch.setattr(C, "_log", lambda m: said.append(m))

    def fake_snapshot(repo_id, repo_type, allow_patterns, local_dir, max_workers, token=None):
        calls.append((repo_id, repo_type, tuple(allow_patterns), local_dir, max_workers))
        for rel in ("thumbnails/aa/aa11.webp", "thumbnails/bb/bb22.webp"):
            p = os.path.join(local_dir, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "wb").write(b"RIFF")
        return local_dir
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    C.write_index(cached, remote_thumbs=True)
    assert C.index_is_current(cached, True)
    assert C.renders_present("thumbnails") == 0
    n = C.download_renders("thumbnails", max_workers=4)
    assert n == 2 and calls == [(C.SPACE, "dataset", ("thumbnails/*",), C.RENDERS_DIR, 4)]
    assert any("CATALOG: downloading the dataset folder thumbnails/" in m and "huggingface.co" in m
               for m in said)
    assert not C.index_is_current(cached, True)                     # new files on disk: stale
    tabs, _ = C.browser_entries(C.load()[0], remote_thumbs=False)
    pika = next(c for c in tabs["characters"] if c["file"] == "aa11")
    assert pika["thumb"].startswith("/gradio_api/file=") and pika["thumb"].endswith("aa11.webp")
    assert pika["img"] == ""                                         # no full render, option off
    tabs, _ = C.browser_entries(C.load()[0], remote_thumbs=True)
    pika = next(c for c in tabs["characters"] if c["file"] == "aa11")
    assert pika["thumb"].startswith("/gradio_api/file=")             # local first
    assert pika["img"] == C.DATASET_BASE + "images/aa/aa11.webp"      # remote fills the gap
    C.write_index(cached, remote_thumbs=True)
    m = json.load(open(os.path.join(cached, "_index", "characters.json"), encoding="utf-8"))
    assert m["renders"] == {"thumbnails": 2, "images": 0}
    assert "2 thumbnails on disk" in C.status(cached)
    assert C.renders_estimate("thumbnails", 34249) == "34,249 files, about 268 MB"
    assert C.renders_estimate("images", 34249) == "34,249 files, about 1.8 GB"
    with pytest.raises(ValueError):
        C.download_renders("videos")

