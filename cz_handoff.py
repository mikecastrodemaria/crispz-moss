"""crispz-moss - hand work over to a sibling studio on this machine (lot 5).

Two gateways, both through the sibling's OWN command line, the family protocol v1
(`cz_protocol.py <op> --spec -`):
  - Edit in crispz-flux3: native multi-reference editing (MOSS-Image edits only
    through the Moss-Image-Edit LoRA, lot 4b);
  - Compare in crispz-ideogram: the SAME prompt (a MOSS layout translated to the
    Ideogram 4 caption schema, cz_caption.to_ideogram) rendered by Ideogram 4 at the
    same size, for an A/B on demand. Ideogram 4 is non-commercial: a comparison, never
    a deliverable (said in the tab and the log).
The sibling decides how to run it:

  - its app is running  -> the edit is queued in that instance (model warm, its queue
                           serializes the GPU);
  - its app is closed   -> it loads FLUX.3 in a separate process (cold). crispz-moss
                           first frees its own VRAM (said in the log): two models of
                           15 to 20 GB do not share a 32 GB card.

Nothing leaves the machine: the image and the references are written as PNG files under
the output folder (`handoff/`) and crispz-flux3 reads them from there (the protocol is
machine-local). Setting: `crispz_flux3_dir`, the crispz-flux3 install folder (env
CRISPZ_FLUX3_DIR). Every hand-over is logged with a `HANDOFF:` line.
"""

import datetime
import json
import os
import subprocess
import sys

from PIL import Image

from cz_core import CONFIG, _log, _prefs

TARGET = "crispz-flux3"
IDEOGRAM = "crispz-ideogram"
TARGETS = {TARGET: {"env": "CRISPZ_FLUX3_DIR", "key": "crispz_flux3_dir"},
           IDEOGRAM: {"env": "CRISPZ_IDEOGRAM_DIR", "key": "crispz_ideogram_dir"}}
MAX_REFS = 3                 # the image itself is reference 0 of the edit
IDEOGRAM_LICENCE = ("Ideogram 4 weights are non-commercial: this render is a comparison, "
                    "never a client deliverable")


class HandoffError(RuntimeError):
    """A hand-over that cannot run or failed, with a message to show as is."""


def target_dir(target=TARGET):
    t = TARGETS[target]
    d = (os.environ.get(t["env"]) or _prefs.get(t["key"]) or CONFIG.get(t["key"], "") or "")
    return os.path.expanduser(str(d).strip().strip('"'))


def flux3_dir():
    return target_dir(TARGET)


def ideogram_dir():
    return target_dir(IDEOGRAM)


def _python(d):
    for p in (os.path.join(d, ".venv", "Scripts", "python.exe"),
              os.path.join(d, ".venv", "bin", "python")):
        if os.path.isfile(p):
            return p
    return None


def status_of(target=TARGET):
    """(usable, reason) from the setting and the folder, without running anything."""
    d = target_dir(target)
    key = TARGETS[target]["key"]
    verb = "edit with" if target == TARGET else "compare with"
    if not d:
        return False, (f"{target} folder not set (`{key}`): set it below to "
                       f"{verb} {target}, on this machine.")
    if not os.path.isdir(d):
        return False, f"{target} folder not found: {d}"
    if not os.path.isfile(os.path.join(d, "cz_protocol.py")):
        return False, f"{d} has no cz_protocol.py: is it a crispz install?"
    if not _python(d):
        return False, f"{d} has no .venv (run its install first)"
    return True, f"{target} at {d}"


def status():
    return status_of(TARGET)


def _czp(args, spec=None, timeout=3600, target=TARGET):
    """Runs the sibling's protocol CLI -> (exit_code, result dict). Replaced in tests."""
    d = target_dir(target)
    cmd = [_python(d), "cz_protocol.py", *args]
    p = subprocess.run(cmd, cwd=d, input=(json.dumps(spec) if spec is not None else None),
                       capture_output=True, text=True, encoding="utf-8", timeout=timeout)
    lines = [ln for ln in (p.stdout or "").splitlines() if ln.strip().startswith("{")]
    if not lines:
        tail = (p.stderr or p.stdout or "").strip()[-300:]
        raise HandoffError(f"{target} gave no answer (exit {p.returncode}): {tail}")
    try:
        return p.returncode, json.loads(lines[-1])
    except ValueError:
        raise HandoffError(f"{target} answer unreadable: {lines[-1][:200]}")


def caps(target=TARGET):
    code, c = _czp(["caps"], timeout=60, target=target)
    if c.get("tool") != target:
        raise HandoffError(f"the folder runs {c.get('tool') or 'an unknown tool'}, not "
                           f"{target}: check `{TARGETS[target]['key']}`")
    return c


def _route(c, target, folder):
    """'instance' when the sibling's app answers (and is really it), else 'cold'."""
    inst = c.get("instance") or {}
    running = bool(inst.get("running"))
    if running and inst.get("tool") not in (None, target):
        # Both studios default to the same port: the sibling's probe can land on THIS app.
        _log(f"HANDOFF: the instance at {inst.get('url')} is {inst.get('tool')}, not "
             f"{target} (same port in both configs?): {target} runs cold instead")
        running = False
    return ("instance" if running else "cold"), inst


def _handoff_dir(output_dir=None):
    from cz_core import DEFAULT_OUTPUT_DIR, HERE
    root = output_dir or DEFAULT_OUTPUT_DIR
    root = root if os.path.isabs(root) else os.path.join(HERE, root)
    d = os.path.join(root, "handoff", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(d, exist_ok=True)
    return d


def _write(img, path):
    img = img if isinstance(img, Image.Image) else Image.open(img)
    img.save(path, "PNG")
    return os.path.abspath(path)


def build_spec(image, instruction, refs=(), seed=-1, output_dir=None):
    """Writes the image and the references as PNG files -> (spec, folder)."""
    instruction = (instruction or "").strip()
    if not instruction:
        raise HandoffError("the edit needs an instruction (what to change)")
    if image is None:
        raise HandoffError("drop the image to edit first")
    refs = [r for r in (refs or []) if r is not None]
    if len(refs) > MAX_REFS:
        _log(f"HANDOFF: {len(refs)} references, {MAX_REFS} kept (the image is the first)")
        refs = refs[:MAX_REFS]
    d = _handoff_dir(output_dir)
    inp = _write(image, os.path.join(d, "input.png"))
    ref_paths = [_write(r, os.path.join(d, f"ref{i + 1}.png")) for i, r in enumerate(refs)]
    if ref_paths:
        # crispz-flux3's `edit` op keeps the input as its ONLY reference (it overwrites
        # spec.refs): with references, `gen` with refs takes the same multi-reference
        # route and keeps them all; the size is pinned to the image's.
        w, h = (image if isinstance(image, Image.Image) else Image.open(image)).size
        spec = {"protocol": 1, "op": "gen", "prompt": instruction,
                "refs": [inp] + ref_paths, "width": int(w), "height": int(h), "out_dir": d}
    else:
        spec = {"protocol": 1, "op": "edit", "prompt": instruction, "input": inp,
                "out_dir": d}
    if seed is not None and int(seed) >= 0:
        spec["seed"] = int(seed)
    return spec, d


def edit(image, instruction, refs=(), seed=-1, output_dir=None, free_vram=None):
    """-> (images, result dict, route). Raises HandoffError."""
    ok, why = status()
    if not ok:
        raise HandoffError(why)
    spec, folder = build_spec(image, instruction, refs, seed, output_dir)
    c = caps()
    route, inst = _route(c, TARGET, folder)
    args = [spec["op"], "--spec", "-"]
    if route == "instance":
        _log(f"HANDOFF: image sent to {TARGET} (its running app, "
             f"{inst.get('url')}), files in {folder}")
    else:
        args.append("--local")
        _log(f"HANDOFF: {TARGET} is not running: it loads FLUX.3 in its own process "
             f"(cold, slower); crispz-moss frees its VRAM first. Files in {folder}")
        if free_vram is not None:
            free_vram()
    code, res = _czp(args, spec)
    if not res.get("ok"):
        raise HandoffError(f"{TARGET}: {res.get('error') or 'failed'} (exit {code})")
    paths = [p for p in res.get("images") or [] if os.path.isfile(p)]
    if not paths:
        raise HandoffError(f"{TARGET} answered without an image file: {str(res)[:200]}")
    imgs = [Image.open(p).copy() for p in paths]
    for w in res.get("warnings") or []:
        _log(f"HANDOFF: {TARGET} warning: {w}")
    _log(f"HANDOFF: {TARGET} edit done ({route}): {', '.join(paths)}")
    with open(os.path.join(folder, "handoff.json"), "w", encoding="utf-8") as f:
        json.dump({"target": TARGET, "route": route, "spec": spec, "result": res}, f,
                  indent=2, ensure_ascii=False)
    return imgs, res, route


def compare(prompt, width, height, seed=-1, steps=None, output_dir=None, free_vram=None):
    """The same prompt rendered by crispz-ideogram at width x height -> (images, result,
    route, ideogram_prompt, notes). A MOSS layout is translated to the Ideogram 4
    caption schema first. Raises HandoffError."""
    import cz_caption
    ok, why = status_of(IDEOGRAM)
    if not ok:
        raise HandoffError(why)
    text = (prompt or "").strip()
    if not text:
        raise HandoffError("the comparison needs a prompt (the main prompt box)")
    w, h = int(width), int(height)
    try:
        caption, notes = cz_caption.to_ideogram(text, w, h)
    except ValueError as e:
        raise HandoffError(f"the layout could not be translated for Ideogram: {e}")
    folder = _handoff_dir(output_dir)
    spec = {"protocol": 1, "op": "gen", "prompt": caption, "width": w, "height": h,
            "out_dir": folder}
    if seed is not None and int(seed) >= 0:
        spec["seed"] = int(seed)
    if steps:
        spec["steps"] = int(steps)
    with open(os.path.join(folder, "ideogram_prompt.json" if caption != text else
                           "ideogram_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(caption)
    for n in notes:
        _log(f"HANDOFF: {n}")
    c = caps(IDEOGRAM)
    route, inst = _route(c, IDEOGRAM, folder)
    args = ["gen", "--spec", "-"]
    if route == "instance":
        _log(f"HANDOFF: prompt sent to {IDEOGRAM} (its running app, {inst.get('url')}) "
             f"for a {w}x{h} comparison, files in {folder}. {IDEOGRAM_LICENCE}")
    else:
        args.append("--local")
        _log(f"HANDOFF: {IDEOGRAM} is not running: it loads Ideogram 4 in its own process "
             f"(cold, slower); crispz-moss frees its VRAM first. Files in {folder}. "
             f"{IDEOGRAM_LICENCE}")
        if free_vram is not None:
            free_vram()
    code, res = _czp(args, spec, target=IDEOGRAM)
    if not res.get("ok"):
        raise HandoffError(f"{IDEOGRAM}: {res.get('error') or 'failed'} (exit {code})")
    paths = [p for p in res.get("images") or [] if os.path.isfile(p)]
    if not paths:
        raise HandoffError(f"{IDEOGRAM} answered without an image file: {str(res)[:200]}")
    imgs = [Image.open(p).copy() for p in paths]
    for wmsg in res.get("warnings") or []:
        _log(f"HANDOFF: {IDEOGRAM} warning: {wmsg}")
    _log(f"HANDOFF: {IDEOGRAM} comparison done ({route}): {', '.join(paths)}")
    with open(os.path.join(folder, "handoff.json"), "w", encoding="utf-8") as f:
        json.dump({"target": IDEOGRAM, "route": route, "spec": spec, "result": res,
                   "notes": notes, "licence": IDEOGRAM_LICENCE}, f, indent=2,
                  ensure_ascii=False)
    return imgs, res, route, caption, notes


def save_setting(folder, target=TARGET):
    """The folder field of a tab -> preferences.json and the running app."""
    from cz_core import _save_prefs_keys
    key = TARGETS[target]["key"]
    folder = (folder or "").strip().strip('"')
    try:
        _save_prefs_keys({key: folder})
    except Exception as e:
        _log(f"{key} not persisted: {e}")
    _prefs[key] = folder
    return status_of(target)


if __name__ == "__main__":       # quick check: python cz_handoff.py
    print(status())
    sys.exit(0)
