module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        // Force UTF-8 stdout/stderr so tqdm download progress bars don't crash
        // on Windows consoles using the legacy cp1252 codec (UnicodeEncodeError).
        // No offload override here: the app's default_cpu_offload is 'auto', which
        // measures the free VRAM at load time and, under 40 GB, keeps the DiT and the
        // VAE on the card and moves only the Qwen3.5-9B text encoder (9.7 GB resident,
        // 15 GB peak at 1024 once the prompt is cached, 23.5 GB while a new prompt is
        // encoded, measured on an RTX 5090).
        env: {
          PYTHONUTF8: "1",
          PYTHONIOENCODING: "utf-8"
        },
        path: "app",
        message: [
          "python app.py"
        ],
        on: [{
          // Capture the local Gradio URL (e.g. http://127.0.0.1:7860)
          "event": "/(http:\\/\\/[0-9.:]+)/",
          "done": true
        }]
      }
    },
    {
      // Expose the captured URL to pinokio.js (Open Web UI menu item)
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    }
  ]
}
