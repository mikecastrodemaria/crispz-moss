"""crispz-moss - the MOSS-Image Character and Style Catalog, in the Asset Browser.

hide1/MOSS-Image-Character-Catalog (Hugging Face): 33,935 character and 314 style
samples rendered with MOSS-Image, each with a reusable prompt, its original tags and
facets (copyright, hair, eyes, gender, outfit). The catalog file (catalog.json.gz, about
7 MB; MIT for the data its maintainer authored) is downloaded ONCE, on the user's click,
from huggingface.co, and the log says so; nothing of the user's leaves the machine. The
sample images stay on the Hugging Face dataset: the browser loads their thumbnails from
there only when the user ticks the option (one outbound request per thumbnail shown,
said in the UI). The characters, artworks and generated images are third-party
intellectual property (the catalog's own NOTICE): not cleared for client deliverables,
said in the UI.

What the module writes, from the cached catalog:
  - `<output>/_index/characters.json` and `styles.json`: the Asset Browser manifests
    (the same shape as the LoRAs / Models tabs: a card per entry, grouped by copyright);
  - `tags/moss_catalog.csv`: the names for the prompt autocomplete (characters, their
    copyrights, the styles), read at the next launch like any tag CSV.
No torch, no model: importable by the UI at any time.
"""
import collections
import datetime
import gzip
import json
import os

from cz_core import HERE, _log, download_with_progress

SPACE = "hide1/MOSS-Image-Character-Catalog"
CATALOG_URL = f"https://huggingface.co/spaces/{SPACE}/resolve/main/catalog.json.gz"
DATASET_BASE = f"https://huggingface.co/datasets/{SPACE}/resolve/main/"
DATASET_PAGE = f"https://huggingface.co/datasets/{SPACE}/blob/main/"
CACHE_DIR = os.path.join(HERE, "cache", "moss")
CACHE = os.path.join(CACHE_DIR, "character_catalog.json.gz")
KINDS = {"character": "characters", "style": "styles"}
TAGS_CSV = "moss_catalog.csv"
STYLE_SUFFIX = " (MOSS caption test)"
NOTICE = ("Uncurated synthetic samples from the MOSS-Image Character Catalog (hide1 on Hugging "
          "Face; MIT for the catalog data, not for what it depicts). The characters, artworks "
          "and generated images are third-party intellectual property: not cleared for client "
          "deliverables. Some identities are wrong (the catalog is not curated).")
# The full-scene generation prompt of every entry ends with the same framing sentence:
# stored once per manifest, not 34,000 times (14 MB less to load in the browser).
_GEN_SUFFIX_MARK = ". Zoomed-out"


def is_available():
    return os.path.isfile(CACHE)


def download(force=False):
    """The catalog file into the cache (said in the log: the only network call of this
    module, on the user's click). Returns the cache path."""
    if is_available() and not force:
        return CACHE
    os.makedirs(CACHE_DIR, exist_ok=True)
    _log(f"CATALOG: downloading {CATALOG_URL} (about 7 MB) to {CACHE}")
    download_with_progress(CATALOG_URL, CACHE, label="character catalog")
    return CACHE


def load(path=None):
    """-> (entries, version) from the cached catalog (or `path`)."""
    with gzip.open(path or CACHE, "rt", encoding="utf-8") as f:
        d = json.load(f)
    entries = d.get("entries") if isinstance(d, dict) else d
    version = (d.get("version") or "") if isinstance(d, dict) else ""
    return [e for e in (entries or []) if isinstance(e, dict) and e.get("id")], version


def browser_entries(entries, remote_thumbs=False):
    """The catalog entries in the Asset Browser's card shape, per tab:
    {"characters": [...], "styles": [...]}, plus the generation-prompt suffix factored
    out of the entries (an entry whose prompt does not end with it keeps the whole text
    and says `gen_full`)."""
    gens = []
    for e in entries:
        g = e.get("reuse_generation_prompt") or ""
        i = g.find(_GEN_SUFFIX_MARK)
        gens.append(g[i:] if i > 0 else "")
    common = collections.Counter(s for s in gens if s).most_common(1)
    suffix = common[0][0] if common else ""
    out = {"characters": [], "styles": []}
    for e in entries:
        tab = KINDS.get(e.get("kind"))
        if not tab:
            continue
        gen = e.get("reuse_generation_prompt") or ""
        card = {
            "file": e["id"], "name": e.get("name") or e["id"],
            "day": e.get("copyright") or "(root)", "mode": e["kind"],
            "prompt": e.get("copy_prompt") or "", "tags": e.get("copy_tags") or "",
            "facets": {k: v for k, v in (e.get("facets") or {}).items()
                       if v and k not in ("character", "copyright")},
            "thumb": (DATASET_BASE + e["demo_thumbnail"]) if remote_thumbs and e.get("demo_thumbnail") else "",
            "img": (DATASET_BASE + e["demo_image"]) if remote_thumbs and e.get("demo_image") else "",
            "hf": e.get("demo_image") or "",
        }
        if suffix and gen.endswith(suffix):
            card["gen"] = gen[:-len(suffix)]
        else:
            card["gen"] = gen
            card["gen_full"] = True
        out[tab].append(card)
    for cards in out.values():
        cards.sort(key=lambda c: (c["day"].lower(), c["name"].lower()))
    return out, suffix


def _index_dir(output_dir):
    from cz_assetbrowser import _ab_resolve_dir
    d = os.path.join(_ab_resolve_dir(output_dir), "_index")
    os.makedirs(d, exist_ok=True)
    return d


def _meta_path(output_dir):
    return os.path.join(_index_dir(output_dir), "catalog.meta.json")


def write_index(output_dir, remote_thumbs=False):
    """Writes the two manifests into the output folder's _index. Returns the counts."""
    from cz_assetbrowser import _write_atomic_text
    entries, version = load()
    tabs, suffix = browser_entries(entries, remote_thumbs)
    idx = _index_dir(output_dir)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    counts = {}
    for tab, cards in tabs.items():
        manifest = {"count": len(cards), "kind": tab, "generated": stamp, "version": version,
                    "remote_thumbs": bool(remote_thumbs), "gen_suffix": suffix,
                    "notice": NOTICE, "source": SPACE, "images": cards}
        _write_atomic_text(os.path.join(idx, tab + ".json"),
                           json.dumps(manifest, ensure_ascii=False, separators=(",", ":")))
        counts[tab] = len(cards)
    _write_atomic_text(_meta_path(output_dir), json.dumps({
        "cache_mtime": os.path.getmtime(CACHE), "remote_thumbs": bool(remote_thumbs),
        "version": version, "counts": counts}))
    _log(f"asset-browser catalog: {counts.get('characters', 0)} characters, "
         f"{counts.get('styles', 0)} styles (thumbnails from huggingface.co: "
         f"{'on' if remote_thumbs else 'off'})")
    return counts


def index_is_current(output_dir, remote_thumbs=False):
    """True when the manifests were written from the cached catalog as it is now, with
    the same thumbnail choice."""
    if not is_available():
        return False
    try:
        with open(_meta_path(output_dir), encoding="utf-8") as f:
            meta = json.load(f)
        idx = _index_dir(output_dir)
        if not all(os.path.isfile(os.path.join(idx, t + ".json")) for t in ("characters", "styles")):
            return False
        return (abs(float(meta.get("cache_mtime", -1)) - os.path.getmtime(CACHE)) < 1.0
                and bool(meta.get("remote_thumbs")) == bool(remote_thumbs))
    except Exception:
        return False


def ensure_index(output_dir, remote_thumbs=False):
    """Writes the manifests when the catalog is cached and they are missing or stale.
    Returns True when something was written."""
    if not is_available() or index_is_current(output_dir, remote_thumbs):
        return False
    write_index(output_dir, remote_thumbs)
    return True


def tag_rows(entries):
    """The autocomplete rows (name, category, count): characters (4) with their source
    count, copyrights (3) with the number of their characters, styles (1)."""
    rows, seen = [], set()
    copyrights = collections.Counter()
    for e in entries:
        name = (e.get("name") or "").strip()
        if not name or "," in name or '"' in name:
            continue
        if e.get("kind") == "character":
            if name.lower() not in seen:
                seen.add(name.lower())
                rows.append((name, 4, int(e.get("source_count") or 0)))
            cr = (e.get("copyright") or "").strip()
            if cr and "," not in cr and '"' not in cr:
                copyrights[cr] += 1
        elif e.get("kind") == "style":
            if name.endswith(STYLE_SUFFIX):
                name = name[:-len(STYLE_SUFFIX)]
            if name.lower() not in seen:
                seen.add(name.lower())
                rows.append((name, 1, 0))
    for cr, n in copyrights.items():
        if cr.lower() not in seen:
            seen.add(cr.lower())
            rows.append((cr, 3, n))
    return rows


def write_tags_csv(tags_dir=None):
    """tags/moss_catalog.csv (the tag autocomplete reads every CSV there at launch).
    Returns the number of rows."""
    if tags_dir is None:
        from cz_tags import TAGS_DIR
        tags_dir = TAGS_DIR
    entries, _ = load()
    rows = tag_rows(entries)
    os.makedirs(tags_dir, exist_ok=True)
    path = os.path.join(tags_dir, TAGS_CSV)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write("# MOSS-Image Character Catalog (hide1, Hugging Face): characters (4), "
                "copyrights (3), styles (1). Rebuilt by crispz-moss when the catalog is downloaded.\n")
        for name, cat, n in rows:
            f.write(f"{name},{cat},{n},\"\"\n")
    os.replace(tmp, path)
    return len(rows)


def status(output_dir=None):
    """One line for the UI."""
    if not is_available():
        return ("Catalog not downloaded yet. The button fetches catalog.json.gz (about 7 MB) "
                "from huggingface.co once; the Characters and Styles tabs of the Asset "
                "Browser then work offline.")
    try:
        with open(_meta_path(output_dir), encoding="utf-8") as f:
            meta = json.load(f)
        counts = meta.get("counts") or {}
        return (f"Catalog cached ({meta.get('version') or 'unknown version'}): "
                f"{counts.get('characters', '?')} characters, {counts.get('styles', '?')} styles.")
    except Exception:
        stamp = datetime.datetime.fromtimestamp(os.path.getmtime(CACHE)).strftime("%Y-%m-%d %H:%M")
        return f"Catalog cached ({stamp}); the Asset Browser builds its tabs when opened."
