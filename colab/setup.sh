#!/bin/bash
set -x
T0=$(date +%s)
cd /content
pip install -q -U huggingface_hub hf_xet >/content/pip_hf.log 2>&1
dl() { python3 - "$@" <<'PY'
import sys, time
from huggingface_hub import hf_hub_download
repo, name, local = sys.argv[1:4]
t=time.time()
try:
    p=hf_hub_download(repo_id=repo, filename=name, local_dir=local); print("DL_DONE", p, round(time.time()-t,1), flush=True)
except Exception as e:
    print("DL_FAIL", name, repr(e)[:500], flush=True)
PY
}
[ -d ComfyUI ] || git clone --depth 1 https://github.com/Comfy-Org/ComfyUI.git ComfyUI
M=/content/ComfyUI/models
C=Comfy-Org/MiniMax-H3
(dl $C diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors $M > /content/dl1.log 2>&1) &
(dl $C text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors $M > /content/dl2.log 2>&1) &
(dl $C vae/minimax_h3_video_vae_fp16.safetensors $M > /content/dl3.log 2>&1; dl $C vae/minimax_h3_audio_vae_fp32.safetensors $M >> /content/dl3.log 2>&1) &
(dl $C loras/pika_h3_flex_ref2va_v1_bf16.safetensors $M > /content/dl4.log 2>&1; dl $C loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors $M >> /content/dl4.log 2>&1) &
(dl $C model_patches/minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors $M > /content/dl5.log 2>&1) &
cd /content/ComfyUI && git log -1 --format='%H %cd %s'
pip install -q -r requirements.txt pillow requests > /content/pip_comfy.log 2>&1
[ -d custom_nodes/ComfyUI-VideoHelperSuite ] || git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git custom_nodes/ComfyUI-VideoHelperSuite
pip install -q -r custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt >> /content/pip_comfy.log 2>&1
echo "INSTALL_DONE $(( $(date +%s)-T0 ))s"
wait
echo "SETUP_DONE $(( $(date +%s)-T0 ))s"
