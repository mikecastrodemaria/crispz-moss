"""crispz-moss - the MOSS-Image LoRA loader (lot 4).

The vendored MOSSImagePipeline has no LoRA mixin and peft knows none of its modules, so
crispz-moss applies LoRAs itself, as FORWARD HOOKS on the DiT's Linear modules: the base
weights stay pristine, the weight slider re-scales live, unloading removes the hooks.
The LoRA tensors are registered as module buffers, so they follow the DiT between the
card and the RAM (the encode swap of lot 3b) with nothing to do.

Dialects read (real files, headers read on 2026-11-04):
  kohya   diffusion_model.<m>.lora_down.weight / .lora_up.weight / .alpha
          (lodestones/MOSS-Image-SFT-8-steps-dmd lora.safetensors, rank 128, fp32)
  peft    <m>.lora_A.weight / .lora_B.weight, "alpha" in __metadata__
          (Comfy-Org/MOSS-Image loras/)
  bare    <m>.lora_A / .lora_B (no ".weight"), per-module alpha in the
          __metadata__ "adapter_config" JSON (format "moss-edit-lora": chinoll/Moss-Image-Edit,
          Comfy-Org's edit LoRA)
  diff    diffusion_model.<m>.diff / .diff_b: a FULL weight (bias) delta, ComfyUI's, the
          RMSNorm weights included (MATLOWAI/MOSS-Image-SFT-DeBand)
Prefixes stripped: diffusion_model., transformer., base_model.model., model.
ComfyUI's fused `layers.N.proj.qkv` (a 3 x 3072 row LoRA up, or diff) is split into the
DiT's separate q, k, v; the text blocks' `text_embedder.blocks.N.qkv` is fused in the
model itself and stays so. Keys that match no module are COUNTED AND SAID, never dropped
in silence. A file that REPLACES a layer outright (a plain `.weight` without LoRA marker,
e.g. the 384-channel `video_embedder` of Comfy-Org's colour / depth CONTROL LoRA) is
refused by name: it needs conditioning inputs this app does not feed.
"""
import json
import os
import re

import torch

PREFIXES = ("diffusion_model.", "transformer.", "base_model.model.", "model.")
LOW_RANK = ((".lora_down.weight", ".lora_up.weight"),
            (".lora_A.weight", ".lora_B.weight"),
            (".lora.down.weight", ".lora.up.weight"),
            (".lora_A", ".lora_B"))
FUSED_QKV_RE = re.compile(r"^(layers\.\d+\.proj)\.qkv$")


def _log(msg):
    try:
        from cz_core import _log as core_log
        core_log(msg)
    except Exception:
        print(f"[crispz] {msg}")


def strip_prefix(key):
    for p in PREFIXES:
        if key.startswith(p):
            return key[len(p):]
    return key


def classify(key):
    """(kind, module) for a tensor name: kind in down / up / alpha / diff / diff_b /
    full / unknown, module = the DiT module the tensor belongs to."""
    k = strip_prefix(key)
    for d, u in LOW_RANK:
        if k.endswith(d):
            return "down", k[:-len(d)]
        if k.endswith(u):
            return "up", k[:-len(u)]
    if k.endswith(".alpha"):
        return "alpha", k[:-len(".alpha")]
    if k.endswith(".diff_b"):
        return "diff_b", k[:-len(".diff_b")]
    if k.endswith(".diff"):
        return "diff", k[:-len(".diff")]
    if k.endswith(".weight") or k.endswith(".bias"):
        return "full", k.rsplit(".", 1)[0]
    return "unknown", k


def _per_module_alphas(meta):
    """alpha per module from a metadata "adapter_config" JSON (moss-edit-lora), or {}."""
    raw = (meta or {}).get("adapter_config")
    if not raw:
        return {}
    try:
        cfg = json.loads(raw)
    except Exception:
        return {}
    out = {}

    def walk(d):
        if not isinstance(d, dict):
            return
        for k, v in d.items():
            if isinstance(v, dict):
                if "alpha" in v and isinstance(v.get("alpha"), (int, float)):
                    out[strip_prefix(str(k))] = float(v["alpha"])
                else:
                    walk(v)
    walk(cfg)
    return out


def _global_alpha(meta):
    try:
        v = (meta or {}).get("alpha")
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


class LoraFile:
    """What a LoRA file holds, in the DiT's own module names.
    items: module -> {"down", "up", "alpha"} (low rank) and / or {"diff"} / {"diff_b"};
    unknown: tensor names that match nothing; full: modules a file REPLACES."""

    def __init__(self, path):
        self.path = path
        self.items = {}
        self.unknown = []
        self.full = []
        self.meta = {}
        self.split = 0


def read_header(path):
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        return list(f.keys()), (f.metadata() or {})


def inspect_file(path):
    """The file's layout from its header alone (no tensor read): a LoraFile whose items
    hold shapes only (`down`/`up`/`diff` -> True)."""
    keys, meta = read_header(path)
    lf = LoraFile(path)
    lf.meta = meta
    for k in keys:
        kind, module = classify(k)
        if kind == "unknown":
            lf.unknown.append(k)
        elif kind == "full":
            lf.full.append(module)
        else:
            lf.items.setdefault(module, {})[kind] = True
    return lf


_EDIT_CACHE = {}


def edit_contract(path):
    """The `edit_contract` of a moss-edit-lora file (its adapter_config metadata), or
    None when the file is not one. Header only, remembered per path."""
    key = (path, os.path.getmtime(path) if os.path.isfile(path) else None)
    if key in _EDIT_CACHE:
        return _EDIT_CACHE[key]
    out = None
    try:
        _keys, meta = read_header(path)
        raw = (meta or {}).get("adapter_config")
        cfg = json.loads(raw) if raw else {}
        if isinstance(cfg, dict) and cfg.get("format") == "moss-edit-lora":
            out = str((cfg.get("metadata") or {}).get("edit_contract") or "moss-edit-lora")
    except Exception:
        out = None
    _EDIT_CACHE[key] = out
    return out


def is_edit_lora(path):
    """True for a Moss-Image-Edit LoRA (reference-prefix editing, cz_moss.EditPrefix)."""
    return edit_contract(path) is not None


def refusal(path):
    """Why this MOSS-family file cannot be applied, or None: a layer replacement (a
    control LoRA), or no LoRA tensor at all."""
    try:
        lf = inspect_file(path)
    except Exception as e:
        return f"unreadable LoRA ({type(e).__name__}: {e})"
    if lf.full:
        what = ", ".join(sorted(set(lf.full))[:3])
        return (f"replaces whole layers ({what}): a CONTROL LoRA that needs conditioning "
                f"inputs this app does not feed (colour / depth maps), not a LoRA")
    if not any(("down" in it and "up" in it) or "diff" in it or "diff_b" in it
               for it in lf.items.values()):
        return "no LoRA tensors found (lora_A / lora_B, lora_down / lora_up or diff)"
    return None


def read_file(path):
    """The tensors of a LoRA file, grouped per DiT module, the fused qkv split."""
    from safetensors.torch import load_file
    sd = load_file(path)
    _keys, meta = read_header(path)
    lf = LoraFile(path)
    lf.meta = meta
    alphas = {}
    for k, t in sd.items():
        kind, module = classify(k)
        if kind == "unknown":
            lf.unknown.append(k)
        elif kind == "full":
            lf.full.append(module)
        elif kind == "alpha":
            alphas[module] = float(t.float().item()) if t.numel() == 1 else None
        else:
            lf.items.setdefault(module, {})[kind] = t
    g_alpha = _global_alpha(meta)
    cfg_alpha = _per_module_alphas(meta)
    for module, it in list(lf.items.items()):
        a = alphas.get(module)
        if a is None:
            a = cfg_alpha.get(module, g_alpha)
        it["alpha"] = a
    # ComfyUI's fused q / k / v on the DiT blocks -> the model's three modules
    for module in list(lf.items):
        m = FUSED_QKV_RE.match(module)
        if not m:
            continue
        it = lf.items.pop(module)
        base = m.group(1)
        parts = {}
        for name in ("q", "k", "v"):
            parts[name] = {"alpha": it.get("alpha")}
        if "up" in it:
            rows = it["up"].shape[0] // 3
            for i, name in enumerate(("q", "k", "v")):
                parts[name]["up"] = it["up"][i * rows:(i + 1) * rows].contiguous()
                parts[name]["down"] = it["down"]
        for key in ("diff", "diff_b"):
            if key in it:
                rows = it[key].shape[0] // 3
                for i, name in enumerate(("q", "k", "v")):
                    parts[name][key] = it[key][i * rows:(i + 1) * rows].contiguous()
        for name, p in parts.items():
            lf.items[f"{base}.{name}"] = p
        lf.split += 1
    return lf


# --- Application ----------------------------------------------------------------------
def _hook(mod, args, out):
    # The LoRA tensors are read from the module's buffers BY NAME at every call: a
    # module.to(device) (the DiT parked in RAM while the text encoder works, then back
    # on the card) replaces the buffers with copies, and a tensor captured at apply time
    # would stay where it was (seen on the 5090, 2026-11-07: a LoRA applied before the
    # DiT reached the card left its tensors on the CPU, "Expected all tensors to be on
    # the same device"; and a captured copy kept a stale duplicate alive).
    x = args[0]
    for dn, un, s in getattr(mod, "_cz_lr", ()):
        d, u = mod._buffers[dn], mod._buffers[un]
        out = out + ((x.to(d.dtype) @ d.t()) @ u.t()).to(out.dtype) * s
    for dn, s in getattr(mod, "_cz_df", ()):
        d = mod._buffers[dn]
        out = out + (x.to(d.dtype) @ d.t()).to(out.dtype) * s
    for bn, s in getattr(mod, "_cz_db", ()):
        out = out + mod._buffers[bn].to(out.dtype) * s
    return out


def _submodule(root, name):
    try:
        return root.get_submodule(name)
    except AttributeError:
        return None


def unload(transformer):
    """Removes every LoRA hook and buffer, puts the norm weights back. Returns how many
    modules were touched."""
    n = 0
    for mod in transformer.modules():
        h = getattr(mod, "_cz_lora_hook", None)
        if h is not None:
            h.remove()
            del mod._cz_lora_hook
            n += 1
        for attr in ("_cz_lr", "_cz_df", "_cz_db", "_cz_scales"):
            if hasattr(mod, attr):
                delattr(mod, attr)
        for name in [b for b in list(mod._buffers) if b.startswith("_cz_lora_")]:
            del mod._buffers[name]
        orig = getattr(mod, "_cz_orig_weight", None)
        if orig is not None:
            mod.weight.data.copy_(orig.to(mod.weight.device, mod.weight.dtype))
            del mod._cz_orig_weight
            n += 1
    return n


def _buf(mod, name, t, keep_dtype=False):
    """Registers `t` as a buffer of `mod`, on the module's device and in its dtype
    (bf16 on the DiT), or in the tensor's own dtype when keep_dtype (an fp32 LoRA whose
    delta is smaller than bf16's rounding step: the hook then runs that matmul in fp32
    and casts the result)."""
    ref = getattr(mod, "weight", None)
    dev = ref.device if ref is not None else torch.device("cpu")
    if keep_dtype and t.is_floating_point():
        dt = t.dtype
    else:
        dt = ref.dtype if ref is not None and ref.is_floating_point() else torch.float32
    mod.register_buffer(name, t.detach().to(dev, dt), persistent=False)
    return name


def apply(transformer, loras, fp32=()):
    """Applies [(path, weight), ...] on the DiT (the previous set removed first).
    Returns one summary per file: {"path", "weight", "modules", "unknown", "split",
    "diffs", "norms"}. Raises on an unreadable file. Files listed in `fp32` keep their
    tensors in their own dtype instead of the DiT's (said in the log)."""
    unload(transformer)
    reports = []
    fp32 = {os.path.abspath(str(p)) for p in (fp32 or ())}
    for li, (path, weight) in enumerate(loras):
        keep = os.path.abspath(str(path)) in fp32
        lf = read_file(path)
        if lf.full:
            raise ValueError(refusal(path))
        rep = {"path": path, "weight": float(weight), "modules": 0, "unknown": list(lf.unknown),
               "split": lf.split, "diffs": 0, "norms": 0, "missing": []}
        for module, it in lf.items.items():
            mod = _submodule(transformer, module)
            if mod is None:
                rep["missing"].append(module)
                continue
            is_linear = isinstance(mod, torch.nn.Linear)
            touched = False
            if "down" in it and "up" in it and is_linear:
                r = it["down"].shape[0]
                base = (float(it["alpha"]) / r) if it.get("alpha") else 1.0
                idx = len(getattr(mod, "_cz_lr", ()))
                dn = _buf(mod, f"_cz_lora_{li}_{idx}_down", it["down"], keep)
                un = _buf(mod, f"_cz_lora_{li}_{idx}_up", it["up"], keep)
                mod._cz_lr = list(getattr(mod, "_cz_lr", ())) + [(dn, un, base * float(weight))]
                mod._cz_scales = list(getattr(mod, "_cz_scales", ())) + [("lr", idx, li, base)]
                touched = True
            if "diff" in it:
                if is_linear:
                    idx = len(getattr(mod, "_cz_df", ()))
                    dn = _buf(mod, f"_cz_lora_{li}_{idx}_diff", it["diff"], keep)
                    mod._cz_df = list(getattr(mod, "_cz_df", ())) + [(dn, float(weight))]
                    mod._cz_scales = list(getattr(mod, "_cz_scales", ())) + [("df", idx, li, 1.0)]
                    rep["diffs"] += 1
                    touched = True
                elif hasattr(mod, "weight") and tuple(mod.weight.shape) == tuple(it["diff"].shape):
                    if getattr(mod, "_cz_orig_weight", None) is None:
                        mod._cz_orig_weight = mod.weight.detach().clone()
                    mod.weight.data.add_(it["diff"].to(mod.weight.device, mod.weight.dtype)
                                         * float(weight))
                    rep["norms"] += 1
                    touched = True
            if "diff_b" in it and is_linear:        # an output offset, bias or not
                idx = len(getattr(mod, "_cz_db", ()))
                bn = _buf(mod, f"_cz_lora_{li}_{idx}_diffb", it["diff_b"], keep)
                mod._cz_db = list(getattr(mod, "_cz_db", ())) + [(bn, float(weight))]
                mod._cz_scales = list(getattr(mod, "_cz_scales", ())) + [("db", idx, li, 1.0)]
                touched = True
            if touched:
                rep["modules"] += 1
                if is_linear and getattr(mod, "_cz_lora_hook", None) is None:
                    mod._cz_lora_hook = mod.register_forward_hook(_hook)
            else:
                rep["missing"].append(module)
        reports.append(rep)
        name = os.path.basename(path)
        msg = (f"LoRA {name} @ {float(weight):g}: {rep['modules']} module(s)"
               + (f", {rep['split']} fused qkv split" if rep["split"] else "")
               + (f", {rep['diffs']} weight diff(s)" if rep["diffs"] else "")
               + (f", {rep['norms']} norm diff(s)" if rep["norms"] else "")
               + (", tensors kept in their own dtype (fp32)" if keep else ""))
        if rep["unknown"] or rep["missing"]:
            msg += (f"; NOT applied: {len(rep['unknown'])} unknown key(s), "
                    f"{len(rep['missing'])} module(s) not in this DiT"
                    f" (first: {(rep['unknown'] + rep['missing'])[0]})")
        _log(msg)
        if rep["modules"] == 0:
            _log(f"LoRA {name}: nothing applied, the render will look exactly as without it")
    return reports


def rescale(transformer, weights):
    """New weights for the SAME files, in order: the scales change, nothing reloads
    (the norm diffs cannot be rescaled in place: the caller re-applies when a file
    carries any). Returns False when a re-apply is needed."""
    for mod in transformer.modules():
        if getattr(mod, "_cz_orig_weight", None) is not None:
            return False
    for mod in transformer.modules():
        scales = getattr(mod, "_cz_scales", None)
        if not scales:
            continue
        for kind, idx, li, base in scales:
            w = float(weights[li]) if li < len(weights) else 0.0
            if kind == "lr":
                d, u, _s = mod._cz_lr[idx]
                mod._cz_lr[idx] = (d, u, base * w)
            elif kind == "df":
                d, _s = mod._cz_df[idx]
                mod._cz_df[idx] = (d, w)
            elif kind == "db":
                b, _s = mod._cz_db[idx]
                mod._cz_db[idx] = (b, w)
    return True


def applied_modules(transformer):
    return sum(1 for m in transformer.modules()
               if getattr(m, "_cz_lora_hook", None) is not None
               or getattr(m, "_cz_orig_weight", None) is not None)
