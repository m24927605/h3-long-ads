# H3 Long Ads

Turn one product photo into a **30-second product ad with synchronized sound** in ComfyUI, using
MiniMax H3 and the new `MiniMax H3 Context Windows` node
([Comfy-Org/ComfyUI#17425](https://github.com/Comfy-Org/ComfyUI/pull/17425)).

MiniMax H3 generates video and audio together, but only up to ~15 s per clip. This workflow slides
overlapping 124-frame windows over a 719-frame (~30 s) timeline and fuses the overlaps at every
sampling step, so picture and sound stay continuous across the whole ad.

Built for the Comfy Org *Open Source Higgsfield* hackathon (2026).

## What it does

1. You give it a product photo (first frame) and a timed shot list.
2. ComfyUI runs MiniMax H3 (first-frame mode, 4-step... see `workflows/`) with the context-windows
   node on the model path.
3. Out comes one MP4: 1376×768, 24 fps, ~30 s, with stereo sound (fizz, ice, splash, music beat).

## Run it on Google Colab (one command)

Requirements: a Colab GPU with ≥ 80 GB VRAM (tested on G4 / RTX PRO 6000 Blackwell 96 GB) and the
[Colab CLI](https://github.com/googlecolab/colab-cli), or paste `colab_run.py` into a notebook.

```bash
colab new --session h3-long-ads --gpu G4
colab upload --session h3-long-ads examples/product.png /content/product.png
colab exec --session h3-long-ads --file colab_run.py --timeout 7200
colab download --session h3-long-ads /content/out/h3_long_ad.mp4 h3_long_ad.mp4
```

## Run it in your own ComfyUI

1. Update ComfyUI to a version that includes `MiniMaxH3ContextWindows` (master after 2026-10-30).
2. Download the model files listed in `MODELS.md` into `ComfyUI/models/...`.
3. Load `workflows/h3_long_ad_ui.json` (or queue `workflows/h3_long_ad_api.json` via the API).

## Licenses

- This repository's code and workflows: MIT (see `LICENSE`).
- MiniMax H3 weights are under the **MiniMax H3 Community License** (not covered by MIT): its
  territory excludes the EU, UK, Republic of Korea and USA, commercial products must display
  "MiniMax H3", and impersonating people without consent is prohibited. Read it before use:
  https://huggingface.co/MiniMaxAI/MiniMax-H3
- Turbo LoRA: see its model card (drbaph/MiniMax-H3-Turbo-Lora-ComfyUI).

Outputs are AI-generated; label them as such when you publish them.
