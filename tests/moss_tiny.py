"""A tiny random MOSS-Image pipeline for CPU tests (no 27 GB download).

Real architecture, toy sizes: the vendored MOSSImageTransformer (2 blocks, 2 heads of 16,
32 wide, text_dim 16), a 4-layer Qwen3.5 text encoder 16 wide (3 linear-attention
blocks + 1 full-attention block, the real hybrid layout, torch fallbacks for the
kernels), the tiny AutoencoderKLFlux2 of the flux3 tests (32 latent channels -> 128
packed), and the real MOSS-Image tokenizer (public, 20 MB, cached by huggingface_hub).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TOKENIZER_REPO = "OpenMOSS-Team/MOSS-Image-SFT"
HID = 16          # text encoder width = the DiT's text_dim


def tiny_text_encoder(seed=0, hid=HID):
    from transformers import Qwen3_5Config, Qwen3_5Model
    torch.manual_seed(seed)
    cfg = Qwen3_5Config(
        text_config=dict(hidden_size=hid, intermediate_size=32, num_hidden_layers=4,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                         vocab_size=248320, linear_num_value_heads=2, linear_num_key_heads=1,
                         linear_key_head_dim=8, linear_value_head_dim=8,
                         linear_conv_kernel_dim=4),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                           out_hidden_size=hid, patch_size=16, temporal_patch_size=2))
    return Qwen3_5Model(cfg).eval()


def tiny_transformer(seed=0, layers=2):
    import cz_moss
    V = cz_moss._vendored()
    torch.manual_seed(seed)
    return V.MOSSImageTransformer(dim=32, ffn_dim=64, num_heads=2, num_layers=layers,
                                  text_dim=HID, freq_dim=8, latent_channels=128).eval()


def tiny_vae(seed=0):
    from diffusers import AutoencoderKLFlux2
    torch.manual_seed(seed)
    vae = AutoencoderKLFlux2(block_out_channels=[32, 32, 32, 32], latent_channels=32,
                             norm_num_groups=8, layers_per_block=1).eval()
    # Non-trivial bn statistics, like the real VAE's (the preview and the token packing
    # tests must not pass by accident on mean 0 / var 1).
    with torch.no_grad():
        vae.bn.running_mean.copy_(torch.randn(128) * 0.5)
        vae.bn.running_var.copy_(torch.rand(128) * 2 + 0.2)
    return vae


def tiny_components(seed=0):
    from transformers import AutoTokenizer
    import cz_moss
    V = cz_moss._vendored()
    return dict(transformer=tiny_transformer(seed), scheduler=V.MOSSImageScheduler(),
                text_encoder=tiny_text_encoder(seed), vae=tiny_vae(seed),
                tokenizer=AutoTokenizer.from_pretrained(TOKENIZER_REPO, subfolder="tokenizer"))


def tiny_pipe(seed=0, raw=False):
    """A MossPipe (the adapter) on the tiny model; raw=True gives the vendored
    MOSSImagePipeline untouched, for the equality checks."""
    import cz_moss
    V = cz_moss._vendored()
    parts = tiny_components(seed)
    if raw:
        return V.MOSSImagePipeline(**parts)
    return cz_moss.MossPipe(cz_moss.pipeline_class()(**parts), repo="tiny")


def save_tiny_comfy_file(path, transformer, int8=False, group=16):
    """Writes `transformer` as a ComfyUI / CivitAI single file: fused layers.N.proj.qkv,
    rope.bands, optional comfy_quant int8 ConvRot (per-row scales, Hadamard groups of
    `group`) on the DiT's linear weights, plus the comfy_attention config blobs."""
    import json
    from safetensors.torch import save_file
    import cz_moss
    sd = {k: v.detach().clone() for k, v in transformer.state_dict().items()}
    out = {}
    for k, v in sd.items():
        if ".proj.q.weight" in k:
            stem = k[:-len("q.weight")]
            out[stem + "qkv.weight"] = torch.cat(
                [sd[stem + "q.weight"], sd[stem + "k.weight"], sd[stem + "v.weight"]], 0)
        elif ".proj.k.weight" in k or ".proj.v.weight" in k:
            continue
        else:
            out[k] = v
    if int8:
        blob = json.dumps({"format": "int8_tensorwise", "convrot": True,
                           "convrot_groupsize": group}).encode()
        for k in [k for k in out if k.startswith("layers.") and ".proj." in k]:
            w = out.pop(k).float()
            rot = cz_moss.rotate_convrot(w, group)
            scale = rot.abs().amax(1, keepdim=True).clamp(min=1e-8) / 127.0
            q = torch.round(rot / scale).clamp(-127, 127).to(torch.int8)
            out[k] = q
            out[k + "_scale"] = scale.float()
            out[k[:-len(".weight")] + ".comfy_quant"] = torch.frombuffer(
                bytearray(blob), dtype=torch.uint8).clone()
        n = len(transformer.layers)
        for i in range(n):
            out[f"layers.{i}.comfy_attention.config"] = torch.frombuffer(
                bytearray(b'{"attention": "comfy_kitchen_int8"}'), dtype=torch.uint8).clone()
    out = {k: (v.to(torch.bfloat16) if v.is_floating_point() and k != "rope.bands" else v)
           for k, v in out.items()}
    save_file({k: v.contiguous() for k, v in out.items()}, path)
    return path


def save_tiny_comfy_text_encoder(path, encoder, int8=False, group=16):
    """`encoder` (a Qwen3_5Model) as ComfyUI's text-encoder file: 'model.' prefix, an
    lm_head and an mtp head added, optional int8 ConvRot on the linear weights."""
    import json
    from safetensors.torch import save_file
    import cz_moss
    out = {"model." + k: v.detach().clone() for k, v in encoder.state_dict().items()}
    hid = encoder.config.text_config.hidden_size
    out["lm_head.weight"] = torch.randn(64, hid)
    out["mtp.fc.weight"] = torch.randn(hid, 2 * hid)
    out["mtp.norm.weight"] = torch.ones(hid)
    if int8:
        blob = json.dumps({"format": "int8_tensorwise", "convrot": True,
                           "convrot_groupsize": group}).encode()
        for k in [k for k in out if k.endswith("_proj.weight") and out[k].dim() == 2
                  and out[k].shape[1] % group == 0]:
            w = out.pop(k).float()
            rot = cz_moss.rotate_convrot(w, group)
            scale = rot.abs().amax(1, keepdim=True).clamp(min=1e-8) / 127.0
            out[k] = torch.round(rot / scale).clamp(-127, 127).to(torch.int8)
            out[k + "_scale"] = scale.float()
            out[k[:-len(".weight")] + ".comfy_quant"] = torch.frombuffer(
                bytearray(blob), dtype=torch.uint8).clone()
    out = {k: (v.to(torch.bfloat16) if v.is_floating_point() and v.dtype != torch.float32
               else v) for k, v in out.items()}
    save_file({k: v.contiguous() for k, v in out.items()}, path)
    return path
