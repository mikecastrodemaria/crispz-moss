const path = require('path')
module.exports = {
  version: "7.0",
  title: "crispz-moss",
  description: "MOSS-Image studio (5B open weights, Apache 2.0: commercial use allowed), Fooocus-style, 100% local: the text-in-image and design studio of the crispz family (posters, logos, packaging, dense typography). Layout JSON with boxes drawn on a canvas, local magic prompt and Describe (Ollama or the model's own Qwen3.5-9B encoder), 9 design presets, emulated img2img / inpaint / outpaint, ESRGAN upscale, LoRAs in every published layout, editing with references (Moss-Image-Edit LoRA), local vectorization (SVG) and Remove BG, the MOSS-Image Character Catalog, hand-overs to crispz-flux3 and crispz-ideogram. NVIDIA GPU with 24 GB or more advised (27 GB of bf16 weights, encoder-only offload). Public weights: no licence to accept, no token. https://github.com/mikecastrodemaria/crispz-moss",
  icon: "icon.png",
  menu: async (kernel, info) => {
    let installed = info.exists("app/env")
    let running = {
      install: info.running("install.js"),
      start: info.running("start.js"),
      update: info.running("update.js"),
      reset: info.running("reset.js")
    }
    if (running.install) {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Installing",
        href: "install.js",
      }]
    } else if (installed) {
      if (running.start) {
        let local = info.local("start.js")
        if (local && local.url) {
          return [{
            default: true,
            icon: "fa-solid fa-rocket",
            text: "Open Web UI",
            href: local.url,
          }, {
            icon: 'fa-solid fa-terminal',
            text: "Terminal",
            href: "start.js",
          }]
        } else {
          return [{
            default: true,
            icon: 'fa-solid fa-terminal',
            text: "Terminal",
            href: "start.js",
          }]
        }
      } else if (running.update) {
        return [{
          default: true,
          icon: 'fa-solid fa-terminal',
          text: "Updating",
          href: "update.js",
        }]
      } else if (running.reset) {
        return [{
          default: true,
          icon: 'fa-solid fa-terminal',
          text: "Resetting",
          href: "reset.js",
        }]
      } else {
        return [{
          default: true,
          icon: "fa-solid fa-power-off",
          text: "Start",
          href: "start.js",
        }, {
          icon: "fa-solid fa-plug",
          text: "Update",
          href: "update.js",
        }, {
          icon: "fa-solid fa-plug",
          text: "Install",
          href: "install.js",
        }, {
          icon: "fa-regular fa-circle-xmark",
          text: "Reset",
          href: "reset.js",
        }]
      }
    } else {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Install",
        href: "install.js",
      }]
    }
  }
}
