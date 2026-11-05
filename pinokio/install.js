module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    // Clone crispz-moss into the local app/ folder
    {
      method: "shell.run",
      params: {
        message: [
          "git clone https://github.com/mikecastrodemaria/crispz-moss app"
        ]
      }
    },
    // Install crispz-moss dependencies (core + extras) in a dedicated venv.
    // Python 3.11: diffusers 0.42.1 and Pillow 13 are not published for 3.10, which is
    // Pinokio's default interpreter. requirements.txt pins diffusers==0.42.1 and
    // transformers 5.x (Qwen3_5Model, the text encoder), and carries bitsandbytes (the
    // optional nf4 encoder), outlines (the local magic prompt's JSON) and vtracer
    // (local vectorization). The MOSS-Image pipeline itself is vendored in the app.
    {
      method: "shell.run",
      params: {
        venv: "env",
        venv_python: "3.11",
        // UTF-8 env: packages built from source read UTF-8 files in their
        // setup; the legacy cp1252 codec on Windows fails the build.
        env: {
          PYTHONUTF8: "1",
          PYTHONIOENCODING: "utf-8"
        },
        path: "app",
        message: [
          "uv pip install -r requirements.txt"
        ]
      }
    },
    // Pillow 13, installed apart and without dependencies: gradio 5.50 declares
    // pillow<12, but the Pillow CVEs are only fixed in 13.0 and the app runs on it
    // (install.bat does the same). Resolving both together would refuse.
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install --no-deps pillow==13.0.0"
        ]
      }
    },
    // The extras (rembg for Remove BG, trustmark, c2pa, ultralytics) AFTER Pillow 13:
    // every rembg since 2.0.75 requires pillow>=12.1 while gradio 5.50 requires
    // pillow<12, so asking for both in one resolution fails (seen 2026-10-28). Installed
    // here, rembg resolves against the Pillow 13 already in place.
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          PYTHONUTF8: "1",
          PYTHONIOENCODING: "utf-8"
        },
        path: "app",
        message: [
          "uv pip install -r requirements-extra.txt"
        ]
      }
    },
    // Face Swap tab dependencies (requirements-faceswap.txt).
    // insightface depends on the CPU 'onnxruntime', which shares its files with
    // 'onnxruntime-gpu': whichever is installed last wins. Installing both in one
    // resolution silently leaves the CPU build on top (no CUDAExecutionProvider),
    // so the runtime matching this machine is force-reinstalled last.
    // onnxruntime-gpu has no macOS/ROCm wheels -> plain onnxruntime elsewhere.
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          PYTHONUTF8: "1",
          PYTHONIOENCODING: "utf-8"
        },
        path: "app",
        message: [
          "uv pip install \"insightface>=0.7\"",
          "uv pip install --force-reinstall --no-deps {{gpu === 'nvidia' ? 'onnxruntime-gpu' : 'onnxruntime'}}"
        ]
      }
    },
    // Install PyTorch (CUDA cu128 on NVIDIA / ROCm / MPS / CPU) cross-platform
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app"
        }
      }
    },
    // Verify the installed transformers really exposes Qwen3_5Model (MOSS-Image's text
    // encoder) and diffusers the FLUX.2 VAE, and that outlines and vtracer import.
    // Without them the app starts fine and only fails at the first render, a confusing
    // place to discover a dependency problem.
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "python -c \"from transformers import Qwen3_5Model; from diffusers import AutoencoderKLFlux2; import outlines, vtracer; print('Qwen3.5 encoder, FLUX.2 VAE, outlines, vtracer OK')\""
        ]
      }
    },
    // Deduplicate the venv to save disk space
    {
      method: "fs.link",
      params: {
        venv: "app/env"
      }
    },
    {
      method: "notify",
      params: {
        html: "crispz-moss installed. MOSS-Image is Apache 2.0 (commercial use allowed), public on Hugging Face: no licence to accept, no token. The first render downloads about 27 GB (DiT 10.2 GB, Qwen3.5-9B encoder 16.8 GB, VAE). Then click <b>Start</b>."
      }
    }
  ]
}
