# H3 Ad Recast

**One ad, every protagonist and every product recast — in a single ComfyUI pass, with sound and
dialogue.**

Give it an existing ad video, a photo of the new protagonist and a photo of the new product. Out
comes the same ad — same timing, gestures and camera — starring the new person holding the new
product, speaking the original Mandarin lines in their own voice.

Built on MiniMax H3 in ComfyUI for the Comfy Org *Open Source Higgsfield* hackathon (2026).

![side by side](examples/side_by_side.jpg)

Watch: [demo.mp4](demo.mp4) · [side by side](examples/side_by_side.mp4) · [recast only](examples/recast.mp4) · [original](examples/source_ad.mp4)

| | Original ad | Recast |
|---|---|---|
| Protagonist | dark-haired woman, white t-shirt | silver-blue bob, lime-green turtleneck (from one photo) |
| Product | red soda can | blue WAVE can (from one photo) |
| Motion | raises the can, talks to camera | same timing, same gestures (Canny ControlNet) |
| Dialogue | 「嗨！今天想跟你分享，我每天都離不開的這罐汽水。」(Mandarin) | same line in Mandarin, her own new voice |

## How it works

```
original ad ──► Canny ──► MiniMax H3 Fun ControlNet (union 2.0) ─┐
new person photo ─┐                                              ├─► MiniMax H3 Context Windows ─► sampler ─► video + audio
new product photo ┴─► MiniMax H3 Reference to Video (Ref2VA) ────┘     (124-frame windows, 39 overlap)
script with timed lines ─┘
```

- **Reference to Video (Ref2VA)** puts `<Picture 1>` (person) and `<Picture 2>` (product) into the
  ad; the prompt carries the dialogue with timestamps, so H3 generates the speech in sync.
- **Fun ControlNet** on Canny edges of the original ad keeps its blocking, gestures and timing.
- **Context Windows** ([Comfy-Org/ComfyUI#17425](https://github.com/Comfy-Org/ComfyUI/pull/17425))
  slide overlapping windows over the whole timeline, so one pass can run past H3's ~15 s training
  length (we ran 19 s recasts and a 30 s product ad) without stitching shots by hand. Peak VRAM for
  the 30 s ad was 42 GB — lower than a 15 s clip without windows (53 GB). A 15.8 s recast samples
  and decodes in about 3 minutes on a G4.
- Everything runs in one ComfyUI graph: `workflows/ad_recast_api.json`.

Also in this repo: `workflows/product_ad_30s_windows_api.json` — a 30-second product ad from a
single product photo with the same context windows (first-frame mode).

## Run it on Google Colab

Tested on a Colab **G4** (RTX PRO 6000 Blackwell, 96 GB). Setup (ComfyUI master, VideoHelperSuite,
~41 GB of weights) takes about 3 minutes; a 15–19 s recast samples in about a minute once loaded.

```bash
colab new --session recast --gpu G4
for f in colab/setup.sh colab/run_recast.py examples/prompt.txt examples/person.png examples/product.png; do
  colab upload --session recast $f /content/$(basename $f); done
colab upload --session recast examples/source_ad.mp4 /content/source_ad.mp4   # 24 fps, pad to 17k+5 frames
# on the VM: bash setup.sh, start ComfyUI (cd ComfyUI && python3 main.py --port 8188 &), then
#   python3 run_recast.py recast /content/prompt.txt 1 0.7      # -> /content/recast.mp4
```

## Run it in your own ComfyUI

1. Update ComfyUI to master (needs `MiniMaxH3ContextWindows`, merged 2026-10-30) and install
   ComfyUI-VideoHelperSuite.
2. Download from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3):
   - `diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` (or `fp8_scaled`)
   - `loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors`
   - `model_patches/minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors`
   - `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
   - `vae/minimax_h3_video_vae_fp16.safetensors`, `vae/minimax_h3_audio_vae_fp32.safetensors`
3. Load `workflows/ad_recast_api.json`, point the video and image loaders at your files, write the
   prompt (who is `<Picture 1>`, what is `<Picture 2>`, the lines with their timestamps), queue.

Frame counts must be 17k + 5 (e.g. 379 for ~15.8 s, 464 for ~19.3 s); hold the last frame of the
control video to pad, then trim the output back to the source length.

## Limits

- One person per ad was tested; the background is re-imagined from the prompt rather than kept
  pixel-exact.
- Each context window sees the full prompt, so very long runs can repeat actions (e.g. splashes)
  and overlap regions can show soft cross-fades. The same applies to dialogue: asking for a second,
  different line later in the ad made every window speak the first line again, so for rewritten
  scripts recast shot by shot (one window per line) instead of one windowed pass.
- Lip sync and the start time of speech are approximate (the speech may start earlier than in the
  original).

## Licenses

- This repository's code and workflows: MIT (see `LICENSE`).
- MiniMax H3 weights are under the **MiniMax H3 Community License** (not covered by MIT): its
  territory excludes the EU, UK, Republic of Korea and USA, commercial products must display
  "MiniMax H3", and impersonating people without consent is prohibited. Read it before use:
  https://huggingface.co/MiniMaxAI/MiniMax-H3
- All people in the examples are AI-generated. Only recast people who consented; label outputs as
  AI-generated.
